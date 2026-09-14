"""Entity extraction and resolution cascade (planS2 §10)."""

from __future__ import annotations

import logging
import re

from rapidfuzz import fuzz, process

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..store import Store

log = logging.getLogger("discrete.process.entities")

_CAP_SPAN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_FILE = re.compile(r"[\w.-]+\.(?:py|js|ts|tsx|md|json|yaml|yml|txt|csv|pdf|png|jpg|svg)\b", re.I)
_DOMAIN = re.compile(r"\b[\w-]+\.(?:com|co|io|org|net|dev|app|ai|design|so|me)\b", re.I)
_HANDLE = re.compile(r"@[\w.-]+")
_REPO = re.compile(r"[\w.-]+/[\w.-]+")


def extract_from_text(text: str) -> list[str]:
    found: set[str] = set()
    for pat in (_CAP_SPAN, _FILE, _DOMAIN, _HANDLE, _REPO):
        found.update(m.group(0).strip() for m in pat.finditer(text or ""))
    for tok in tk.tokens(text):
        if any(c.isdigit() for c in tok) and len(tok) >= 3:
            found.add(tok)
        if len(tok) >= 2 and tok[0].isupper() and tok[1:].islower() and tok.lower() not in tk.STOPWORDS:
            found.add(tok)
    return sorted(found)


def infer_kind(name: str) -> str:
    low = name.lower()
    if _DOMAIN.search(name) or "." in name:
        return "website"
    if _FILE.search(name):
        return "file"
    if "@" in name:
        return "person"
    if any(k in low for k in ("illustrator", "figma", "vscode", "chrome", "safari", "slack", "notion")):
        return "software"
    return "unknown"


async def resolve_entity(store: Store, name: str) -> int:
    nk = tk.norm_key(name)
    existing = await store.find_entity_by_alias(nk)
    if existing is not None:
        return existing

    aliases = await store.all_entity_aliases()
    if aliases:
        choices = {a["alias_norm"]: a["entity_id"] for a in aliases}
        match = process.extractOne(nk, list(choices.keys()), scorer=fuzz.WRatio)
        if match and match[1] >= T.FUZZY_AUTO:
            return choices[match[0]]
        if match and match[1] >= T.FUZZY_ADJUDICATE:
            decided = await _adjudicate(name, match[0])
            if decided:
                return choices[match[0]]

    vec = await store.get_embedding_for_name(name)
    if vec is not None:
        best_id, best_sim = await store.nearest_entity(vec)
        if best_sim >= T.ENTITY_EMBED_SIM and best_id:
            return best_id

    return await store.upsert_entity(name, kind=infer_kind(name))


async def _adjudicate(a: str, b: str) -> bool:
    ep = insight_endpoint()
    if not ep.configured:
        return False
    prompt = f'Are "{a}" and "{b}" the same entity? Return JSON {{"same": true/false}}'
    try:
        data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=50)
        return bool((data or {}).get("same"))
    except EXPECTED_ERRORS:
        return False


async def promote_recurring(store: Store) -> int:
    """Entities seen in >= ENTITY_MIN_EPISODES become real rows."""
    rows = await store.entity_candidates()
    promoted = 0
    for row in rows:
        eps = tk.parse_json_list(row["episodes"])
        if len(eps) < T.ENTITY_MIN_EPISODES:
            continue
        await resolve_entity(store, row["name"])
        promoted += 1
    return promoted
