"""Query planning with temporal and aggregate intents (planS3 §2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import dateparser
from dateparser.search import search_dates

from .. import textkit as tk
from ..config import T, utility_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..timeutil import day_bounds, day_start, from_local, now, to_local

_FILE = re.compile(r"[\w.-]+\.(?:py|js|ts|tsx|md|json|yaml|yml|txt|csv|pdf|png|jpg)\b", re.I)
_IDENT = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[\w.-]+\.[\w.-]+\b")
_AGG = re.compile(r"\b(?:which|what)\b.+\b(?:have i used|did i use|tools|apps|websites)\b", re.I)
_TIMELINE = re.compile(r"\b(?:when|timeline|morning|yesterday|last week|today)\b", re.I)

KIND_VOCAB = {
    "software": ("tool", "tools", "app", "apps", "software", "program"),
    "website": ("site", "sites", "website", "websites", "domain"),
    "person": ("person", "people", "who"),
    "project": ("project", "repo", "repository"),
}


@dataclass
class QueryPlan:
    raw: str
    intent: str = "recall"
    queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    time_hint: str = ""
    interval: tuple[float, float] | None = None
    entity_kind_filter: str = ""
    needs_store: bool = True


async def build_plan(text: str) -> QueryPlan:
    plan = fallback_plan(text)
    if T.QUERY_LLM_PLAN:
        ep = utility_endpoint()
        if ep.configured:
            prompt = (
                "Return JSON {intent, queries[], entities[], time_hint, needs_store, entity_kind_filter}. "
                f"Query: {text}"
            )
            try:
                data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=400)
                if isinstance(data, dict):
                    plan.intent = str(data.get("intent") or plan.intent)
                    plan.queries = list({*plan.queries, *(str(q) for q in (data.get("queries") or []) if q)})
                    plan.entities = list({*plan.entities, *(str(e) for e in (data.get("entities") or []) if e)})
                    plan.time_hint = str(data.get("time_hint") or plan.time_hint)
                    plan.entity_kind_filter = str(data.get("entity_kind_filter") or plan.entity_kind_filter)
                    plan.needs_store = bool(data.get("needs_store", True))
            except EXPECTED_ERRORS:
                pass
    plan.terms = tk.content_tokens(text)
    return plan


def fallback_plan(text: str) -> QueryPlan:
    ents = list({m.group(0) for m in _IDENT.finditer(text)})
    files = list({m.group(0) for m in _FILE.finditer(text)})
    terms = tk.content_tokens(text)
    queries = list({q for q in ents + files + terms[:10] if q}) or terms[:5] or [text[:80]]
    intent = "recall"
    if "?" not in text and len(text.split()) > 12 and not terms[:3]:
        intent = "continuation"
    edit_verbs = ("fix", "rewrite", "complete", "summarize", "translate", "format")
    if any(text.lower().startswith(v) for v in edit_verbs) and "?" not in text:
        intent = "edit"
    interval = _parse_interval(text)
    if interval:
        intent = "temporal" if intent == "recall" else intent
    entity_kind = ""
    if _AGG.search(text):
        intent = "aggregate"
        low = text.lower()
        for kind, words in KIND_VOCAB.items():
            if any(w in low for w in words):
                entity_kind = kind
                break
    if _TIMELINE.search(text) and intent == "recall":
        intent = "temporal"
    return QueryPlan(
        raw=text,
        intent=intent,
        queries=queries,
        entities=ents + files,
        terms=terms,
        interval=interval,
        entity_kind_filter=entity_kind,
    )


def _parse_interval(text: str) -> tuple[float, float] | None:
    low = text.lower()
    ref = to_local(now())
    if "this morning" in low:
        start = from_local(ref.replace(hour=0, minute=0, second=0, microsecond=0))
        end = from_local(ref.replace(hour=12, minute=0, second=0, microsecond=0))
        return start, end
    if "today" in low:
        return day_bounds(now())
    if "yesterday" in low:
        start = day_start(now()) - 86400
        return start, start + 86400
    try:
        parsed = search_dates(text, settings={"PREFER_DATES_FROM": "past"})
    except Exception:
        parsed = None
    if parsed:
        dates = [d for _, d in parsed]
        if len(dates) == 1:
            start = day_start(from_local(dates[0]))
            return start, start + 86400
        if len(dates) >= 2:
            return min(dates[0], dates[1]).timestamp(), max(dates[0], dates[1]).timestamp()
    return None
