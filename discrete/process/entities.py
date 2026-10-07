"""Entities: candidates, recurrence-gated promotion, resolution cascade (planS2 §10)."""

from __future__ import annotations

import logging
import re

from rapidfuzz import fuzz

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, complete_json, try_embed
from ..store import Store
from .prompts import ADJUDICATE_PROMPT

log = logging.getLogger("discrete.process.entities")

_CAP_SPAN = re.compile(r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z0-9]+)+\b")  # never across a line break
_INNER_CAP = re.compile(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*\b|\b[A-Za-z]+\d[A-Za-z0-9]*\b")
_FILE = re.compile(r"\b[\w.-]+\.(?:py|js|ts|tsx|md|json|ya?ml|csv|pdf|ai|psd|fig|sketch)\b", re.I)
_DOMAIN = re.compile(r"\b(?:[\w-]+\.)+(?:com|co|io|org|net|dev|app|ai|design|so|me|xyz)\b", re.I)
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z][\w.]{2,}")
_REPO = re.compile(r"\b[A-Za-z][\w.-]+/[A-Za-z][\w.-]+\b")
_DIGITS = re.compile(r"\d+")

# Document chrome and generic category words are never promoted (planS2 §10.3).
_CHROME = re.compile(r"^(?:untitled|artboard|new tab|new document|document|page|layer|tab|window)\b[\s\d-]*$", re.I)
_GENERIC = frozenset("tech saas ai design app apps tool tools website site home settings search menu help browser".split())
_PIPELINE_WORDS = frozenset("user users".split())
_PATH_ROOTS = frozenset("users documents library applications private tmp var volumes home desktop downloads opt "
                        "system".split())

# Identifier-shaped patterns only. Multi-word Title Case from AX text is mostly the
# app's own sidebar/menu labels, which recur in every episode and pass the recurrence
# gate (insights.md §7.27); real multi-word names arrive via the VLM and claim sources.
_STRUCTURAL = (_INNER_CAP, _FILE, _DOMAIN, _HANDLE, _REPO)


def structural_candidates(text: str) -> list[str]:
    """planS2 §10.2 source 3. Runs on AX text only (Rev 3): OCR errors recur."""
    found: set[str] = set()
    for line in (text or "").splitlines():
        for pat in _STRUCTURAL:
            found.update(m.group(0).strip() for m in pat.finditer(line))
    return sorted(found)


def promotable(name: str) -> bool:
    n = name.strip()
    key = tk.entity_key(n)
    if len(key) < 3 or key.isdigit() or _CHROME.match(n) or tk.norm_key(n) in _GENERIC:
        return False
    if "\n" in n or "\t" in n:
        return False
    toks = n.lower().split()
    if len(toks) > 1 and len(set(toks)) == 1:  # "Unnamed Unnamed": a UI echo, not a name
        return False
    if "/" in n and n.strip("/").split("/")[0].lower() in _PATH_ROOTS:  # filesystem path fragment
        return False
    toks = n.split()
    # A single all-lowercase dictionary-looking word is not a name.
    if len(toks) == 1 and n.islower() and n.isalpha() and "." not in n:
        return False
    # Function words and the pipeline's own vocabulary ("the user") are not names.
    if not [t for t in tk.content_tokens(n) if t not in _PIPELINE_WORDS]:
        return False
    return True


def infer_kind(name: str) -> str:
    """Deterministic kinds only; pages assign the rest (kind is never identity)."""
    if _DOMAIN.fullmatch(name.strip()):
        return "website"
    if _FILE.fullmatch(name.strip()):
        return "file"
    if name.strip().startswith("@"):
        return "person"
    if _REPO.fullmatch(name.strip()):
        return "project"
    return "unknown"


def _distinct(a: str, b: str) -> bool:
    """Names that are never one entity however alike they look: different numbers
    (Atlas 101 / Atlas 102), or two different identifiers of one shape (acme.com /
    acme.org, report.py / report.md). A name and an identifier may still meet (Acme /
    acme.com): that is what the identity key is for."""
    if _DIGITS.findall(a) != _DIGITS.findall(b):
        return True
    kind = infer_kind(a)
    return kind != "unknown" and kind == infer_kind(b) and tk.norm_key(a) != tk.norm_key(b)


async def _adjudicate(a: str, b: str) -> bool:
    ep = insight_endpoint()
    if not ep.configured:
        return False  # no model → no merge; a duplicate is cheaper than a false merge
    try:
        data = await complete_json(ep, [{"role": "user", "content": ADJUDICATE_PROMPT.format(a=a, b=b)}], max_tokens=30)
        return isinstance(data, dict) and data.get("same") is True  # "false" is a truthy string
    except EXPECTED_ERRORS:
        return False


async def resolve_existing(store: Store, name: str, *, use_embeddings: bool = False) -> int | None:
    """Cascade steps 1-5 against existing entities; never creates."""
    nk = tk.norm_key(name)
    if (eid := await store.find_entity_by_alias(nk)) is not None:
        return eid
    key = tk.entity_key(name)
    if not key:
        return None
    if (eid := await store.find_entity_by_key(key)) is not None:
        row = await store.get_entity(eid)
        return None if row and _distinct(name, row["name"]) else eid
    if len(key) >= 5:
        # The alias list is cached by the store until the alias table changes.
        best, best_score = None, 0.0
        for a in await store.all_entity_aliases():
            if len(a["norm_key"]) < 5 or _distinct(name, a["alias"]):
                continue
            s = fuzz.WRatio(key, a["norm_key"]) if abs(len(key) - len(a["norm_key"])) <= 3 else fuzz.ratio(key, a["norm_key"])
            if s > best_score:
                best, best_score = a, s
        if best is not None and best_score >= T.FUZZY_AUTO:
            return best["entity_id"]
        if best is not None and best_score >= T.FUZZY_ADJUDICATE and await _adjudicate(name, best["alias"]):
            return best["entity_id"]
    if use_embeddings:
        vec = await try_embed(name)
        if vec is None:
            return None
        for eid, _sim in await store.dense_search("entity", vec, limit=3, min_sim=T.ENTITY_EMBED_SIM):
            row = await store.get_entity(eid)
            if row and not _distinct(name, row["name"]) and await _adjudicate(name, row["name"]):
                return eid
    return None


async def promote_recurring(store: Store) -> int:
    """Candidates seen in ≥ ENTITY_MIN_EPISODES episodes become entities (or aliases),
    then mentions are back-filled from live claims in one pass."""
    promoted = 0
    resolved: list[tuple[int, str]] = []
    for row in await store.entity_candidates():
        eps = tk.parse_json_list(row["episodes"])
        if len(eps) < T.ENTITY_MIN_EPISODES or not promotable(row["name"]):
            continue
        eid = await resolve_existing(store, row["name"], use_embeddings=True)
        if eid is None:
            eid = await store.create_entity(row["name"], infer_kind(row["name"]), row["name_norm"])
            promoted += 1
        else:
            await store.add_alias(eid, row["name"], row["name_norm"])
        await store.delete_candidate(row["name_norm"])
        resolved.append((eid, row["name"]))
    await _backfill_mentions(store, resolved)
    return promoted


async def _backfill_mentions(store: Store, resolved: list[tuple[int, str]]) -> None:
    """Claims (text or supporting quotes) naming each newly resolved entity. Live
    claims and their quotes are loaded once for the whole batch."""
    if not resolved:
        return
    found = []
    for cid, text, quotes in await store.live_claims_with_quotes():
        haystack = "\n".join([text, *quotes])
        found += [(eid, "claim", cid) for eid, name in resolved if tk.contains_phrase(haystack, name)]
    await store.record_mentions(found)
