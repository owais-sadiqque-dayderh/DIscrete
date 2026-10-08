"""Optional LLM listwise rerank (planS3 §6.2)."""

from __future__ import annotations

import logging
import math

from ..config import T, utility_endpoint
from ..llm import EXPECTED_ERRORS, complete_json

log = logging.getLogger("discrete.retrieval.rerank")


async def llm_rerank(query: str, docs: list[str]) -> list[tuple[int, float]] | None:
    if not T.LLM_RERANK or len(docs) < 2:
        return None
    ep = utility_endpoint()
    if not ep.configured:
        return None
    items = [{"id": i, "text": t[:500]} for i, t in enumerate(docs[: T.LLM_RERANK_POOL])]
    prompt = (
        f"Reorder these documents by relevance to the query. Return JSON "
        f'{{"order": [ids], "unusable": [ids]}}. Query: {query}\nDocs: {items}'
    )
    try:
        data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=400)
    except EXPECTED_ERRORS as exc:
        log.debug("llm rerank failed: %s", type(exc).__name__)
        return None
    if not isinstance(data, dict):
        log.debug("llm rerank returned no object")
        return None
    order, unusable = data.get("order"), data.get("unusable") or []
    if not _indices(order, len(items)) or not isinstance(unusable, list) or \
            (unusable and not _indices(unusable, len(items))):
        log.debug("llm rerank returned invalid ids")
        return None
    return [(i, 0.1 if i in unusable else 1.0 - n * 0.01) for n, i in enumerate(order)]


def _indices(ids: object, n: int) -> bool:
    """A nonempty list of distinct in-range integer ids (bools are not ids)."""
    return (isinstance(ids, list) and bool(ids) and len(set(ids)) == len(ids)
            and all(type(i) is int and 0 <= i < n for i in ids))


def valid_order(order: object, n: int) -> list[tuple[int, float]] | None:
    """A reranker's (index, score) list, or None when any entry is malformed:
    a bad index, a repeat or a non-finite score would misplace or duplicate evidence."""
    if not isinstance(order, list):
        return None
    out = []
    for pair in order:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        i, score = pair
        if type(i) is not int or not 0 <= i < n or isinstance(score, bool) or \
                not isinstance(score, (int, float)) or not math.isfinite(score):
            return None
        out.append((i, float(score)))
    return out if len({i for i, _ in out}) == len(out) else None
