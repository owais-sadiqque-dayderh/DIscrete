"""Optional LLM listwise rerank (planS3 §6.2)."""

from __future__ import annotations

import logging

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
        order = [int(x) for x in (data or {}).get("order") or []]
        unusable = set(int(x) for x in (data or {}).get("unusable") or [])
        out = [(i, 0.1 if i in unusable else 1.0 - n * 0.01) for n, i in enumerate(order)]
        return out
    except EXPECTED_ERRORS as exc:
        log.debug("llm rerank failed: %s", exc)
        return None
