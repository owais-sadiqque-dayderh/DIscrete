"""Query planning (planS3 §2, Rev 3).

The structural plan is authoritative for intent-bearing structure (edit,
aggregate, interval). The LLM plan contributes query strings, entities and a
kind filter, and its intent only from a closed set; it can never switch
retrieval off or supply an interval.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import timedelta

from dateparser.search import search_dates

from .. import textkit as tk
from ..config import T, utility_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..models import QueryPlan
from ..timeutil import day_bounds, from_local, now, to_local

INTENTS = frozenset({"edit", "recall", "temporal", "aggregate", "enumerate", "world", "continuation", "mixed", "why"})
ENTITY_KINDS = frozenset({"person", "organization", "software", "website", "project", "file", "place", "topic"})

_FILE = re.compile(r"[\w.-]+\.(?:py|js|ts|tsx|md|json|yaml|yml|txt|csv|pdf|png|jpg)\b", re.I)
_IDENT = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[\w-]+\.[a-z]{2,}\b")
# Entity roll-up ("which tools have I used") or a count — answered from the entity table.
_AGG = re.compile(r"\b(?:which|what)\b.*\b(?:have i|did i|do i)\b.*\b(?:use|used|visit|visited|try|tried)\b"
                  r"|\bhow many\b|\bhow much\b|\bcount\b", re.I)
# Enumeration ("list all the articles I opened") — the user wants every matching
# record listed, not a synthesized summary or a count (planS3 §17).
_ENUM = re.compile(r"\ball (?:the|of the)\b|\bevery\b|\blist\b|\bshow (?:me |all )\b"
                   r"|\bwhat .*\bdid i (?:open|read|visit|see|watch|look)\b", re.I)
# "Why did I …", "what made me …": answered by following causal trails (trails.py).
_WHY = re.compile(r"^\s*why\b|\bwhy (?:did|do|was|were|have|had) i\b|\bwhat (?:made|led|prompted|caused|got) me\b"
                  r"|\bhow did i (?:end up|come to|get to|find|hear about|discover)\b|\breason (?:i|for my|why i)\b", re.I)
# Questions about the user as a whole, answered best with the profile (process/profile.py).
_ABOUT_USER = re.compile(
    r"\babout me\b|\bmy (?:interests|hobbies|habits|routines?|projects|preferences|work|tools)\b"
    r"|\b(?:what|which) (?:kinds?|sorts?|types?) of\b.*\b(?:i|me)\b"
    r"|\b(?:am i|do i|i'm)\b.*\b(?:interested|into|building|working on|usually|tend to|often|prefer|rely on)\b"
    r"|\bwho am i\b|\bdescribe me\b", re.I)
# "what time on Monday did I…" asks when, not how long: no bare "time on".
_TIME_SPENT = re.compile(r"\b(?:how much time|how long (?:did|do|have) i (?:spend|spent|been)|most (?:of my )?time|"
                         r"time spent|spend(?:ing)? (?:my |the most )?time|spent (?:the )?most)\b", re.I)
_EDIT_VERBS = ("fix", "rewrite", "complete", "summarize", "summarise", "translate", "format", "rephrase", "shorten")
# "Summarize the messages I viewed" asks about the user's past, not about the draft;
# "summarize what I wrote above" still points at the field.
_OWN_PAST = re.compile(r"\bi\s+(?:\w+ed|saw|seen|read|did|had|was|were|wrote|sent|got|went|heard|met|took|made)\b"
                       r"|\b(?:did|have|had)\s+i\b", re.I)
# A request for an answer, not a draft to continue.
_REQUEST = re.compile(r"\s*(?:give|tell|show|list|find|summari[sz]e)\s+(?:me|all|every|the|my)\b", re.I)
_DEICTIC = re.compile(r"\b(?:above|below|this|these|here)\b", re.I)
# Page enumeration lists visited pages. A summary, or a list of things *inside* pages
# (messages, names, prices), is answered from their contents by ordinary search.
_SUMMARY = re.compile(r"\bsummar(?:y|ies|ise|ize|ised|ized)\b|\brecap\b|\bgist\b", re.I)
_ENUM_OBJECT = re.compile(r"\b(?:list|show me|what|all|every)(?:\s+(?:all|every|of|the|my))*\s+([a-z]+)\b", re.I)
_PAGE_NOUNS = frozenset(
    "article articles page pages site sites website websites link links tab tabs url urls video videos post posts "
    "story stories thread threads blog blogs doc docs document documents file files repo repos repositories "
    "app apps window windows".split())
_NOT_NOUNS = frozenset("i you we that what which where when did do have things stuff everything".split())
# Match the user's completion question, not an action mentioned later in a recall question.
_COMPLETED_ACTION = (
    r"(?:book(?:ed|ing)?|order(?:ed|ing)?|buy(?:ing)?|bought|pay(?:ing)?|paid|send(?:ing)?|sent|submit(?:ted|ting)?|"
    r"switch(?:ed|ing)?|install(?:ed|ing)?|subscrib(?:e|ed|ing)|repl(?:y|ied|ying)|purchas(?:e|ed|ing)|"
    r"complete(?:d)? the (?:purchase|order|payment|booking|checkout|subscription))"
)
# "Did I end up buying", "have I already paid", "did I go ahead and submit".
_PREMISE_FILLER = (r"(?:(?:actually|really|already|ever|finally|eventually)\s+)?"
                   r"(?:(?:end(?:ed)? up|go ahead and|get around to|manage(?:d)? to|finish(?:ed)?)\s+)?")
_PREMISE_CLOSED = re.compile(
    r"^\s*(?:did|have|had)\s+i\s+" + _PREMISE_FILLER + _COMPLETED_ACTION + r"\b", re.I)
_PREMISE_ACTUAL = re.compile(
    r"\b(?:did|have|had)\s+i\s+(?:actually|really)\s+" + _COMPLETED_ACTION + r"\b", re.I)
_PREMISE_CHOICE = re.compile(
    r"^\s*(?:which|what)\b[^?]*\b(?:did|have|had)\s+i\s+" + _COMPLETED_ACTION
    + r"\b[^?]*\bor\b", re.I)
_PREMISE_TEMPORAL = re.compile(r"^\s*(?:when\b|what time\b|how (?:long|recently)\b)", re.I)
# The request's own scaffolding. Searching for "list" or "opened" matches any row that
# says so (a todo list), not the pages being asked about (insights.md §7.36).
_ENUM_SCAFFOLD = frozenset(
    "list all every show me open opened visit visited read saw seen see look looked watch watched till until now "
    "ever far article articles page pages site sites link links tab tabs item items thing things "
    "below above here".split())


def strip_scaffold(text: str) -> str:
    return " ".join(t for t in tk.content_tokens(text) if t not in _ENUM_SCAFFOLD)


def page_enumeration(text: str) -> bool:
    """Does this list request ask for visited pages (not a summary or things inside them)?"""
    if _SUMMARY.search(text):
        return False
    if set(tk.tokens(text)) & _PAGE_NOUNS:
        return True
    m = _ENUM_OBJECT.search(text)
    noun = m[1].lower() if m else ""
    return not noun or noun in _PAGE_NOUNS or noun in _NOT_NOUNS


def is_why(text: str, plan: QueryPlan | None = None) -> bool:
    return bool((plan and plan.intent == "why") or _WHY.search(text))


def is_premise_question(text: str) -> bool:
    """Completion checks (including either/or choices), excluding temporal recall."""
    if _PREMISE_TEMPORAL.search(text) or is_why(text):
        return False
    return bool(_PREMISE_CLOSED.search(text) or _PREMISE_ACTUAL.search(text) or _PREMISE_CHOICE.search(text))


KIND_VOCAB = {
    "software": ("tool", "tools", "app", "apps", "software", "program", "programs", "editor", "editors"),
    "website": ("site", "sites", "website", "websites", "domain", "domains", "page", "pages"),
    "person": ("person", "people", "who"),
    "project": ("project", "projects", "repo", "repos", "repository"),
    "file": ("file", "files", "document", "documents"),
}


@dataclass(frozen=True)
class RelativeEvent:
    phrase: str
    direction: str
    last: bool = False


@dataclass
class EventQueryPlan(QueryPlan):
    anchor: RelativeEvent | None = None


def relative_event(text: str) -> RelativeEvent | None:
    # Causal questions have their own resolver; dates and clocks remain intervals.
    if is_why(text):
        return None
    m = re.search(r"\b(?:(?:right|just|immediately)\s+)?(after|before)\s+(.+?)\s*[?.!]*$"
                  r"|\bwhen\s+I\s+(finished\s+.+?)\s*[?.!]*$", text, re.I)
    if not m:
        return None
    phrase = (m[2] or m[3]).strip().rstrip("?.!")
    phrase = re.sub(r"^I\s+(?:had\s+)?", "", phrase, flags=re.I)
    if not phrase or re.fullmatch(r"(?:\d[\d: .apm-]*|today|yesterday|last .+|this .+)", phrase, re.I):
        return None
    # A clock/date ("after midnight on Oct 1") is an interval, not an event.
    if _TEMPORAL_CUE.search(phrase) or re.search(r"\b(?:midnight|noon|lunch|dinner|breakfast)\b", phrase, re.I):
        return None
    # Only the user's own next steps: "what did X reply after I…" asks for content the
    # following moments cannot supply, and an unanswerable must stay a refusal.
    head = text[:m.start()]
    if not re.search(r"\b(?:what|where|which|who)\b.*\b(?:did|was|were|do)\s+I\b", head, re.I):
        return None
    last = bool(re.match(r"(?:finish(?:ed|ing)?|complet(?:ed|ing|e)|done with)\b", phrase, re.I))
    return RelativeEvent(phrase, (m[1] or "after").lower(), last)


# A dateparser match is trusted only if it contains one of these (planS3 Rev 3).
_TEMPORAL_CUE = re.compile(
    r"\d|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day)?\b"
    r"|\b(?:today|tonight|yesterday|morning|afternoon|evening|night|week|month|year|ago|last|since|earlier)\b",
    re.I,
)


_PLAN_PROMPT = """Plan a memory search for this question about the user's own past computer activity.

Question: {q}

Return strict JSON:
{{"intent": "<recall|temporal|aggregate|world|mixed>",
  "queries": ["<2-6 short keyword strings likely to appear in screen text or notes>"],
  "entities": ["<names mentioned in the question>"],
  "entity_kind_filter": "<person|organization|software|website|project|file|place|topic or empty>"}}
"""


async def build_plan(text: str, *, relative_events: bool = True) -> QueryPlan:
    plan = fallback_plan(text, relative_events=relative_events)
    if plan.intent == "edit" or isinstance(plan, EventQueryPlan) or plan.page_operation or not T.QUERY_LLM_PLAN:
        return plan
    ep = utility_endpoint()
    if not ep.configured:
        return plan
    try:
        # The structural plan is always usable, so a slow planner is abandoned, not awaited.
        data = await asyncio.wait_for(
            complete_json(ep, [{"role": "user", "content": _PLAN_PROMPT.format(q=text)}], max_tokens=300),
            timeout=T.PLAN_TIMEOUT_S,
        )
    except EXPECTED_ERRORS:
        return plan
    if not isinstance(data, dict):
        return plan
    # Optional structure: anything not shaped as documented is ignored, never coerced
    # (str() of a dict or a string's characters would become search terms).
    for q in _strings(data.get("queries")):
        q = q.strip()
        if plan.intent == "enumerate":
            q = strip_scaffold(q)
        if q and q not in plan.queries:
            plan.queries.append(q)
    for e in _strings(data.get("entities")):
        e = e.strip()
        if e and e not in plan.entities:
            plan.entities.append(e)
    kind = data.get("entity_kind_filter")
    kind = kind.lower() if isinstance(kind, str) else ""
    if not plan.entity_kind_filter and kind in ENTITY_KINDS:
        plan.entity_kind_filter = kind
    llm_intent = data.get("intent")
    llm_intent = llm_intent.lower() if isinstance(llm_intent, str) else ""
    if plan.intent == "recall" and llm_intent in INTENTS and llm_intent not in ("edit", "continuation"):
        if llm_intent == "temporal" and plan.interval is None:
            llm_intent = "recall"
        plan.intent = llm_intent
    return plan


def _strings(value: object) -> list[str]:
    return [v for v in value if isinstance(v, str)][:8] if isinstance(value, list) else []


def fallback_plan(text: str, *, reference_at: float | None = None, relative_events: bool = True) -> QueryPlan:
    ents = list(dict.fromkeys(m.group(0) for m in _IDENT.finditer(text)))
    files = list(dict.fromkeys(m.group(0) for m in _FILE.finditer(text)))
    terms = tk.content_tokens(text)
    queries = list(dict.fromkeys([*ents, *files, " ".join(terms[:8])])) if terms else [text[:80]]
    queries = [q for q in queries if q]
    low = text.lower().strip()
    intent = "recall"
    summary_of_past = bool(_SUMMARY.search(low[:12]) and _OWN_PAST.search(text) and not _DEICTIC.search(text))
    if any(low.startswith(v) for v in _EDIT_VERBS) and "?" not in text and not summary_of_past:
        intent = "edit"
    elif ("?" not in text and len(text.split()) > 12 and not (set(tk.tokens(text)[:2]) & tk.INTERROGATIVES)
          and not _REQUEST.match(text)):
        intent = "continuation"
    interval = parse_interval(text, reference_at)
    anchor = relative_event(text) if relative_events else None
    kind = ""
    if intent in ("recall", "continuation") and is_why(text):
        intent = "why"
    elif intent == "recall" and anchor:
        intent = "temporal"
    elif intent == "recall" and _AGG.search(text):
        intent = "aggregate"
        words = set(tk.tokens(text))
        kind = next((k for k, vocab in KIND_VOCAB.items() if words & set(vocab)), "")
    elif intent == "recall" and _ENUM.search(text) and page_enumeration(text):
        intent = "enumerate"
        terms = [t for t in terms if t not in _ENUM_SCAFFOLD]
        queries = list(dict.fromkeys(q for q in (strip_scaffold(x) for x in queries) if q))
    elif intent == "recall" and interval:
        intent = "temporal"
    operation, page_search = page_request(text) if T.PAGES and intent not in ("edit", "continuation", "why") else ("", "")
    if operation:
        intent = "enumerate" if operation == "edited" else "aggregate"
    plan_type = EventQueryPlan if anchor and intent == "temporal" else QueryPlan
    plan = plan_type(raw=text, intent=intent, queries=queries, entities=ents + files, terms=terms,
                     interval=interval, entity_kind_filter=kind, needs_store=intent != "edit",
                     about_user=intent not in ("edit", "continuation") and bool(_ABOUT_USER.search(text)),
                     time_spent=intent not in ("edit", "continuation") and bool(_TIME_SPENT.search(text)),
                     page_operation=operation, page_search=page_search,
                     premise_check=intent not in ("edit", "continuation") and is_premise_question(text))
    if isinstance(plan, EventQueryPlan):
        plan.anchor = anchor
    return plan


def page_request(text: str) -> tuple[str, str]:
    """Closed personal activity forms; preference/ownership/causal questions stay on recall."""
    low = text.lower().strip().rstrip("?")
    period = r"(?:\s+(?:today|yesterday|this week|last week|this month|last month))?"
    if re.fullmatch(r"what (?:did|do|have) i read (?:the )?most" + period, low):
        return "read_most", ""
    if re.fullmatch(r"(?:which|what) (?:repos|repositories) (?:do|did|have) i (?:return to|returned to|revisit)" + period, low):
        return "returns", ""
    if re.fullmatch(r"(?:which|what) (?:docs|documents|files) (?:did|do|have) i (?:edit|edited)" + period, low):
        return "edited", ""
    m = re.match(r"^how (?:long|much time)\s+(?:(?:did i spend|do i spend|have i spent)\s+)?on\s+(.+)$", low)
    if m:
        target = re.sub(r"\s+(?:today|yesterday|this (?:week|month|morning|evening)|last (?:week|month))$", "", m[1])
        if target and not re.search(r"\b(?:why|prefer|own|bought|purchase|install|installing|and|or)\b", target):
            target = {"x": "x.com", "twitter": "x.com"}.get(target.strip(), target.strip())
            return "duration", target
    return "", ""


def parse_interval(text: str, ref: float | None = None) -> tuple[float, float] | None:
    low = text.lower()
    ref = ref if ref is not None else now()
    local = to_local(ref)
    today0, today1 = day_bounds(ref)

    def at(hour: int, base: float = today0) -> float:
        return from_local(to_local(base).replace(hour=hour, minute=0, second=0, microsecond=0))

    if "this morning" in low:
        return today0, at(12)
    if "this afternoon" in low:
        return at(12), at(18)
    if "this evening" in low or "tonight" in low:
        return at(17), today1
    if "last night" in low:
        return at(18, today0 - 86400), at(6)
    if "yesterday" in low:
        return today0 - 86400, today0
    if "today" in low:
        return today0, today1
    week0 = from_local((local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0))
    if "this week" in low:
        return week0, today1
    if "last week" in low:
        return week0 - 7 * 86400, week0
    m = re.search(r"\b(\d+|a|an|one|two|three)\s+(day|days|hour|hours)\s+ago\b", low)
    if m:
        n = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))
        if m.group(2).startswith("hour"):
            t = ref - n * 3600
            return t - 3600, t + 3600
        return day_bounds(ref - n * 86400)
    # A clock-only anchor identifies the captured moment across stored days. It
    # must not silently become today's date (or discard yesterday's captures).
    without_clock = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", "", text, flags=re.I)
    if is_why(text) and re.search(r"\b\d{1,2}:\d{2}\b", text) and not _TEMPORAL_CUE.search(without_clock):
        return None
    # No parser result can survive the cue filter without a cue in the input.
    # Avoid language/date detection work on ordinary effect titles.
    if not _TEMPORAL_CUE.search(text):
        return None
    try:
        found = search_dates(text, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": local.replace(tzinfo=None)})
    except Exception:
        found = None
    dates = [d for span, d in (found or []) if _TEMPORAL_CUE.search(span)]
    if dates:
        return day_bounds(from_local(dates[0]))
    return None
