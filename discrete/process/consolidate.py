"""Offline consolidation (planS2 §7)."""

from __future__ import annotations

import logging

from ..config import T, consolidate_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..process.entities import promote_recurring, resolve_entity
from ..process.embed_worker import drain_outbox
from ..store import Store
from ..timeutil import day_key, now

log = logging.getLogger("discrete.process.consolidate")

CONSOLIDATE_PROMPT = """Merge or split these related claims into a bounded set.
Return JSON: {{"claims": [{{"text": "...", "kind": "fact|preference", "input_ids": [1,2]}}]}}
Each output ≤240 chars, atomic, count ≤ input count + 2.
Inputs:
{inputs}
"""


async def run_consolidation(store: Store) -> dict:
    report = {"clusters": 0, "digests": 0, "lint": {}, "embedded": 0, "entities": 0}
    report["embedded"] = await drain_outbox(store)
    report["entities"] = await promote_recurring(store)
    await _refresh_entity_pages(store)
    report["digests"] += await _daily_digest(store)
    report["clusters"] = await _cluster_refine(store)
    report["lint"] = await _lint(store)
    await store.set_meta("last_consolidation", str(now()))
    return report


async def _cluster_refine(store: Store) -> int:
    ep = consolidate_endpoint()
    if not ep.configured:
        return 0
    clusters = await store.similar_claim_clusters(T.CONSOLIDATE_SIM, T.CONSOLIDATE_MAX_CLUSTER)
    refined = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        lines = [f"#{r['id']} ({r['kind']}) {r['text']}" for r in cluster]
        prompt = CONSOLIDATE_PROMPT.format(inputs="\n".join(lines))
        try:
            data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=1500)
            outputs = (data or {}).get("claims") or []
            if outputs:
                await store.apply_consolidation(cluster, outputs)
                refined += 1
        except EXPECTED_ERRORS as exc:
            log.warning("consolidation cluster failed: %s", exc)
    return refined


async def _daily_digest(store: Store) -> int:
    ep = consolidate_endpoint()
    if not ep.configured:
        return 0
    dk = day_key(now())
    if await store.get_digest(dk):
        return 0
    eps = await store.episodes_for_day(dk)
    if not eps:
        return 0
    lines = [f"- {e['title']}: {e['summary']}" for e in eps if e["summary"]]
    prompt = f"Write a 3-5 sentence daily digest from these episodes:\n" + "\n".join(lines)
    try:
        from ..llm import complete

        text = await complete(ep, [{"role": "user", "content": prompt}], max_tokens=400)
        await store.save_digest(dk, text.strip())
        return 1
    except EXPECTED_ERRORS:
        return 0


async def _refresh_entity_pages(store: Store) -> None:
    ep = consolidate_endpoint()
    if not ep.configured:
        return
    for row in await store.entities_needing_pages(T.ENTITY_PAGES_PER_RUN):
        claims = await store.claims_for_entity(row["id"])
        if not claims:
            continue
        lines = [f"- {c['text']}" for c in claims[:20]]
        prompt = f"Summarize what we know about entity '{row['name']}' from:\n" + "\n".join(lines)
        try:
            from ..llm import complete

            summary = await complete(ep, [{"role": "user", "content": prompt}], max_tokens=300)
            await store.update_entity_page(row["id"], summary.strip())
        except EXPECTED_ERRORS:
            continue


async def _lint(store: Store) -> dict:
    cov = await store.reconcile_index_coverage()
    unsupported = await store.supersede_unsupported(T.UNSUPPORTED_TTL_DAYS)
    return {"index_gaps_repaired": cov, "unsupported_superseded": unsupported}
