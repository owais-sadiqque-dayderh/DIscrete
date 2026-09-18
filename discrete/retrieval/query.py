"""Query planning (planS3 §2, Rev 3).

The structural plan is authoritative for intent-bearing structure (edit,
aggregate, interval). The LLM plan contributes query strings, entities and a
kind filter, and its intent only from a closed set; it can never switch
retrieval off or supply an interval.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import timedelta

from .. import textkit as tk
from ..config import T, utility_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..timeutil import day_bounds, from_local, now, to_local

INTENTS = frozenset({"edit", "recall", "temporal", "aggregate", "enumerate", "world", "continuation", "mixed"})
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
_EDIT_VERBS = ("fix", "rewrite", "complete", "summarize", "translate", "format", "rephrase", "shorten")
# The request's own scaffolding. Searching for "list" or "opened" matches any row that
# says so (a todo list), not the pages being asked about (insights.md §7.36).
_ENUM_SCAFFOLD = frozenset(
    "list all every show me open opened visit visited read saw seen see look looked watch watched till until now "
    "ever far article articles page pages site sites link links tab tabs item items thing things "
    "below above here".split())


def strip_scaffold(text: str) -> str:
    return " ".join(t for t in tk.content_tokens(text) if t not in _ENUM_SCAFFOLD)

KIND_VOCAB = {
    "software": ("tool", "tools", "app", "apps", "software", "program", "programs", "editor", "editors"),
    "website": ("site", "sites", "website", "websites", "domain", "domains", "page", "pages"),
    "person": ("person", "people", "who"),
    "project": ("project", "projects", "repo", "repos", "repository"),
    "file": ("file", "files", "document", "documents"),
}

# A dateparser match is trusted only if it contains one of these (planS3 Rev 3).
_TEMPORAL_CUE = re.compile(
    r"\d|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day)?\b"
    r"|\b(?:today|tonight|yesterday|morning|afternoon|evening|night|week|month|year|ago|last|since|earlier)\b",
    re.I,
)


@dataclass
class QueryPlan:
    raw: str
    intent: str = "recall"
    queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    interval: tuple[float, float] | None = None
    entity_kind_filter: str = ""
    needs_store: bool = True


_PLAN_PROMPT = """Plan a memory search for this question about the user's own past computer activity.

Question: {q}

Return strict JSON:
{{"intent": "<recall|temporal|aggregate|world|mixed>",
  "queries": ["<2-6 short keyword strings likely to appear in screen text or notes>"],
  "entities": ["<names mentioned in the question>"],
  "entity_kind_filter": "<person|organization|software|website|project|file|place|topic or empty>"}}
"""


async def build_plan(text: str) -> QueryPlan:
    plan = fallback_plan(text)
    if plan.intent == "edit" or not T.QUERY_LLM_PLAN:
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
    for q in data.get("queries") or []:
        q = str(q).strip()
        if plan.intent == "enumerate":
            q = strip_scaffold(q)
        if q and q not in plan.queries:
            plan.queries.append(q)
    for e in data.get("entities") or []:
        e = str(e).strip()
        if e and e not in plan.entities:
            plan.entities.append(e)
    kind = str(data.get("entity_kind_filter") or "").lower()
    if not plan.entity_kind_filter and kind in ENTITY_KINDS:
        plan.entity_kind_filter = kind
    llm_intent = str(data.get("intent") or "").lower()
    if plan.intent == "recall" and llm_intent in INTENTS and llm_intent not in ("edit", "continuation"):
        if llm_intent == "temporal" and plan.interval is None:
            llm_intent = "recall"
        plan.intent = llm_intent
    return plan


def fallback_plan(text: str) -> QueryPlan:
    ents = list(dict.fromkeys(m.group(0) for m in _IDENT.finditer(text)))
    files = list(dict.fromkeys(m.group(0) for m in _FILE.finditer(text)))
    terms = tk.content_tokens(text)
    queries = list(dict.fromkeys([*ents, *files, " ".join(terms[:8])])) if terms else [text[:80]]
    queries = [q for q in queries if q]
    low = text.lower().strip()
    intent = "recall"
    if any(low.startswith(v) for v in _EDIT_VERBS) and "?" not in text:
        intent = "edit"
    elif "?" not in text and len(text.split()) > 12 and not (set(tk.tokens(text)[:2]) & tk.INTERROGATIVES):
        intent = "continuation"
    interval = parse_interval(text)
    kind = ""
    if intent == "recall" and _AGG.search(text):
        intent = "aggregate"
        words = set(tk.tokens(text))
        kind = next((k for k, vocab in KIND_VOCAB.items() if words & set(vocab)), "")
    elif intent == "recall" and _ENUM.search(text):
        intent = "enumerate"
        terms = [t for t in terms if t not in _ENUM_SCAFFOLD]
        queries = list(dict.fromkeys(q for q in (strip_scaffold(x) for x in queries) if q))
    elif intent == "recall" and interval:
        intent = "temporal"
    return QueryPlan(raw=text, intent=intent, queries=queries, entities=ents + files, terms=terms,
                     interval=interval, entity_kind_filter=kind, needs_store=intent != "edit")


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
    try:
        from dateparser.search import search_dates

        found = search_dates(text, settings={"PREFER_DATES_FROM": "past", "RELATIVE_BASE": local.replace(tzinfo=None)})
    except Exception:
        found = None
    dates = [d for span, d in (found or []) if _TEMPORAL_CUE.search(span)]
    if dates:
        return day_bounds(from_local(dates[0]))
    return None
