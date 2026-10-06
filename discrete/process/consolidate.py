"""Offline consolidation (planS2 §7). Every step is safe to re-run."""

from __future__ import annotations

import logging

import numpy as np

from .. import textkit as tk
from ..config import T, consolidate_endpoint
from ..llm import EXPECTED_ERRORS, complete, complete_json
from ..models import ClaimDraft
from ..perf import PERF
from ..store import Store
from ..timeutil import now, to_local
from .embed_worker import drain_outbox
from .entities import promote_recurring
from .prompts import CONSOLIDATE_PROMPT, DIGEST_PROMPT, ENTITY_PAGE_PROMPT
from .validation import atomic
from .verification import verify_claims

log = logging.getLogger("discrete.process.consolidate")


async def run_consolidation(store: Store) -> dict:
    with PERF.timer("consolidate.run"):
        return await _consolidate(store)


async def _consolidate(store: Store) -> dict:
    report: dict = {}
    report["embedded"] = await drain_outbox(store)
    report["entities_promoted"] = await promote_recurring(store)
    report["clusters_refined"] = await _cluster_refine(store)
    report["entity_pages"] = await _entity_pages(store)
    report["digests"] = await _digests(store)
    report["embedded"] += await drain_outbox(store)
    report["lint"] = await _lint(store)
    await store.set_meta("last_consolidation", str(now()))
    return report


async def _clusters(store: Store) -> list[list]:
    clusters = []
    for kind in ("fact", "preference"):
        rows = await store.live_claims(kind=kind)
        vecs = await store.vectors("claim", [r["id"] for r in rows])
        used: set[int] = set()
        for r in rows:
            if r["id"] in used or r["id"] not in vecs:
                continue
            members = [r]
            for r2 in rows:
                if r2["id"] == r["id"] or r2["id"] in used or r2["id"] not in vecs:
                    continue
                linked = r2["refines_id"] == r["id"] or r["refines_id"] == r2["id"]
                if linked or float(np.dot(vecs[r["id"]], vecs[r2["id"]])) >= T.CONSOLIDATE_SIM:
                    members.append(r2)
                if len(members) >= T.CONSOLIDATE_MAX_CLUSTER:
                    break
            if len(members) >= 2:
                used.update(m["id"] for m in members)
                clusters.append(members)
    return clusters


async def _cluster_refine(store: Store) -> int:
    ep = consolidate_endpoint()
    if not ep.configured:
        return 0
    refined = 0
    for cluster in await _clusters(store):
        ids = {r["id"] for r in cluster}
        prompt = CONSOLIDATE_PROMPT.format(
            inputs="\n".join(f"[id={r['id']}] ({r['kind']}) {r['text']}" for r in cluster),
            max_out=len(cluster) + 2,
        )
        try:
            data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=1200)
        except EXPECTED_ERRORS as exc:
            log.warning("consolidation cluster failed: %s", exc)
            continue
        if not isinstance(data, dict):
            continue
        outputs = [o for o in data.get("claims") or [] if isinstance(o, dict)]
        # A call failing validation changes nothing (planS2 §7.1).
        covered = {int(i) for o in outputs for i in (o.get("input_ids") or []) if str(i).lstrip("-").isdigit()}
        if not outputs or len(outputs) > len(cluster) + 2 or not ids <= covered:
            continue
        if not all(atomic(str(o.get("text") or "")) for o in outputs):
            continue
        drafts = []
        by_claim = {r["id"]: r for r in cluster}
        for output in outputs:
            inputs = output.get("input_ids") or []
            if not inputs or any(not str(i).isdigit() or int(i) not in ids for i in inputs):
                break
            evidence = []
            for cid in dict.fromkeys(int(i) for i in inputs):
                evidence.extend({"observation_id": r["observation_id"], "quote": r["quote"]}
                                for r in await store.evidence_for_claim(cid) if r["relation"] == "supports")
            drafts.append(ClaimDraft(kind=by_claim[int(inputs[0])]["kind"], text=output["text"], evidence=evidence))
        if len(drafts) != len(outputs):
            continue
        observations = await store.get_observations(list({e["observation_id"] for d in drafts for e in d.evidence}),
                                                   user_only=True)
        try:
            if not all(await verify_claims(drafts, {r["id"]: r for r in observations})):
                continue  # never supersede good inputs with an unsupported refinement
        except EXPECTED_ERRORS as exc:
            log.warning("consolidation verification failed: %s", exc)
            continue
        if await store.consolidate_cluster(cluster, outputs):
            refined += 1
    return refined


async def _entity_pages(store: Store) -> int:
    ep = consolidate_endpoint()
    if not ep.configured:
        return 0
    n = 0
    for row in await store.entities_needing_pages(T.ENTITY_PAGES_PER_RUN):
        claims = await store.claims_for_entity(row["id"], limit=20)
        if not claims:
            continue
        prompt = ENTITY_PAGE_PROMPT.format(name=row["name"], claims="\n".join(f"- {c['text']}" for c in claims))
        try:
            data = await complete_json(ep, [{"role": "user", "content": prompt}], max_tokens=400)
        except EXPECTED_ERRORS:
            continue
        if not isinstance(data, dict):
            continue
        summary = tk.truncate(str(data.get("summary") or ""), 800)
        if not summary:
            continue
        kind = str((data or {}).get("kind") or "").lower()
        await store.update_entity_page(row["id"], summary, kind=kind)
        n += 1
    return n


async def _digests(store: Store) -> int:
    """planS2 §7.4 (Rev 3): (re)written whenever a day's summarized-episode count changes."""
    ep = consolidate_endpoint()
    if not ep.configured:
        return 0
    n = 0
    for day in await store.days_with_episodes():
        eps = [e for e in await store.episodes_for_day(day) if e["status"] == "closed" and e["summary"]]
        if not eps:
            continue
        existing = await store.get_digest(day)
        if existing and existing["n_episodes"] == len(eps):
            continue
        lines = "\n".join(f"- {to_local(e['started_at']):%H:%M} {e['title']}: {e['summary']}" for e in eps)
        try:
            text = (await complete(ep, [{"role": "user", "content": DIGEST_PROMPT.format(episodes=lines)}],
                                   max_tokens=400)).strip()
        except EXPECTED_ERRORS:
            continue
        if text:
            await store.save_digest(day, text, len(eps))
            n += 1
    return n


async def _lint(store: Store) -> dict:
    stats = await store.stats()
    return {
        "index_gaps_repaired": await store.reconcile_index_coverage(),
        "unsupported_superseded": await store.supersede_unsupported(T.UNSUPPORTED_TTL_DAYS),
        "lineage_anomalies": await store.lineage_anomalies(),
        "episodes_backlog": {k: v for k, v in stats["episodes_by_status"].items() if k != "closed"},
        "outbox_pending": stats["outbox_pending"],
    }
