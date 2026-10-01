"""Unified hybrid retrieval (planS3 §3-6): BM25 ‖ dense → RRF → gates → rerank → MMR."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from .. import textkit as tk
from ..confidence import half_life_days
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, rerank, try_embed
from ..models import Hit, QueryPlan, RetrievalResult
from ..perf import PERF, since_ms
from ..store import KINDS, Store
from ..timeutil import age_days, day_key, now, to_local
from .enumerate import enumerate_pages
from .query import build_plan
from .rerank import llm_rerank

log = logging.getLogger("discrete.retrieval.hybrid")


def is_dense_hit(h: Hit) -> bool:
    return h.dense_sim >= T.DENSE_HIT_THRESHOLD


async def materialize(store: Store, kind: str, ids: list[int]) -> dict[int, Hit]:
    """Live rows as Hits; one query per kind (plus one for claim quotes)."""
    out: dict[int, Hit] = {}
    rows = await store.live_rows(kind, ids)
    if kind == "claim":
        quotes = await store.first_quotes([r["id"] for r in rows])
        for r in rows:
            q = quotes.get(r["id"])
            body = r["text"] + (f"\n> {q}" if q else "")
            out[r["id"]] = Hit("claim", r["id"], r["text"][:120], body, modality=r["modality"],
                               confidence=r["confidence"], status=r["status"], occurred_at=r["occurred_from"],
                               occurred_to=r["occurred_to"], claim_kind=r["kind"], untrusted=bool(q))
    elif kind == "observation":
        for r in rows:
            name = r["page_title"] or r["window_title"]
            title = f"{r['app']} — {name}".strip(" —") or r["caption"][:80]
            body = "\n".join(p for p in (r["caption"], tk.truncate(r["screen_text"], 1500)) if p)
            out[r["id"]] = Hit("observation", r["id"], title[:120], body, occurred_at=r["started_at"],
                               occurred_to=r["ended_at"], untrusted=True, url=r["url"] or "",
                               is_listing=bool(r["is_listing"]), navigated_from=r["navigated_from"])
    elif kind == "episode":
        for r in rows:
            out[r["id"]] = Hit("episode", r["id"], r["title"] or "episode", r["summary"],
                               occurred_at=r["started_at"], occurred_to=r["ended_at"])
    elif kind == "entity":
        for r in rows:
            out[r["id"]] = Hit("entity", r["id"], r["name"], r["summary"] or r["name"])
    return out


def coverage(terms: list[str], text: str, df_info: tuple[dict[str, int], int]) -> float:
    """IDF-weighted share of query terms present (token match, prefix for plurals)."""
    df, n = df_info
    if not terms:
        return 0.0
    toks = set(tk.tokens(text))
    num = den = 0.0
    for t in dict.fromkeys(terms):
        w = tk.bm25_idf(df.get(t, 0), n) + 1.0
        den += w
        if t in toks or (len(t) >= 4 and any(x.startswith(t) or t.startswith(x) and len(x) >= 4 for x in toks)):
            num += w
    return num / den if den else 0.0


def _recency(h: Hit) -> float:
    hl = half_life_days(h.claim_kind if h.ref_kind == "claim" else h.ref_kind)
    if not hl:
        return 1.0
    return 0.5 ** (age_days(h.occurred_to or h.occurred_at or now()) / hl)


def _noop(*_a) -> None:
    pass


async def retrieve(store: Store, query: str, *, field_text: str = "", progress=None) -> RetrievalResult:
    """`progress(stage, detail)` is an optional UI hook (Bubble's overlay)."""
    with PERF.timer("retrieve.total") as t:
        result = await _retrieve(store, query, field_text=field_text, progress=progress or _noop)
        t.n = len(result.hits)
    return result


async def _retrieve(store: Store, query: str, *, field_text: str, progress) -> RetrievalResult:
    # The planner and the query embedding are independent API calls: run them together.
    qvec_task = asyncio.create_task(try_embed(query))
    try:
        with PERF.timer("retrieve.plan"):
            plan = await build_plan(query)
    except BaseException:
        qvec_task.cancel()
        raise
    result = RetrievalResult(query=query, plan=plan)
    progress("searching", plan.intent)
    if plan.intent == "edit":
        qvec_task.cancel()
        return result

    if plan.intent == "enumerate":
        with PERF.timer("retrieve.enumerate"):
            await enumerate_pages(store, plan, result, await qvec_task)
        return result

    if plan.interval and plan.interval[1] - plan.interval[0] <= T.EXHAUSTIVE_MAX_SPAN_S and plan.intent in ("temporal", "recall"):
        await _exhaustive(store, plan, result)
        if result.hits:
            qvec_task.cancel()
            return result

    t_search = time.perf_counter()
    cold = await store.observation_count() < T.COLD_START_ROWS
    df_info = await store.df_lookup(plan.terms)
    qvec = await qvec_task
    pool: list[Hit] = []

    if plan.intent == "aggregate":
        agg = await store.aggregate_entities(plan.entity_kind_filter, *(plan.interval or (None, None)))
        for row in agg[: T.TOP_K]:
            pool.append(Hit("entity", row["id"], row["name"], f"{row['summary'] or row['name']} "
                            f"(seen in {row['episodes']} episodes)", score=1.0, strong=True))

    seen = {(h.ref_kind, h.ref_id) for h in pool}
    for kind in KINDS:
        rankings = [await store.fts_ids(kind, q, limit=T.CANDIDATE_POOL) for q in plan.queries[:8]]
        dense: dict[int, float] = {}
        if qvec is not None:
            dense = dict(await store.dense_search(kind, qvec, limit=T.CANDIDATE_POOL))
            rankings.append(list(dense))
        fused = tk.rrf([r for r in rankings if r], k=T.RRF_K)
        ids = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[: T.CANDIDATE_POOL]]
        hits = await materialize(store, kind, ids)
        for rid in ids:
            h = hits.get(rid)
            if h is None or (kind, rid) in seen or not h.in_interval(plan.interval):
                continue
            h.score = fused[rid] * (1.0 if plan.interval else _recency(h))
            h.dense_sim = dense.get(rid, 0.0)
            h.coverage = coverage(plan.terms, h.text, df_info)
            h.echo = tk.jaccard(tk.trigrams(field_text), tk.trigrams(h.body)) if field_text else 0.0
            if h.echo >= T.ECHO_THRESHOLD:
                continue
            if not cold and h.coverage < T.COVERAGE_FLOOR and not is_dense_hit(h):
                continue
            h.strong = is_dense_hit(h) or (not cold and h.coverage >= T.COVERAGE_STRONG)
            pool.append(h)

    pool.sort(key=lambda h: -h.score)
    PERF.sample("retrieve.search", since_ms(t_search), n=len(pool))  # BM25 ‖ dense → fused, gated pool
    progress("ranking", len(pool))
    with PERF.timer("retrieve.rerank", n=len(pool)):
        pool = await _rerank(query, pool, cold)
    with PERF.timer("retrieve.mmr"):
        result.hits = await _mmr(store, pool, T.TOP_K)
    return result


async def _rerank(query: str, pool: list[Hit], cold: bool) -> list[Hit]:
    """planS3 §6.1: order within the included set; never removes a hit."""
    if len(pool) <= 3:
        return pool
    head, tail = pool[: T.RERANK_POOL], pool[T.RERANK_POOL:]
    docs = [tk.truncate(h.text, T.RERANK_MAX_TOKENS * 4) for h in head]
    try:
        order = await rerank(query, docs, len(docs))
    except (NotConfigured, *EXPECTED_ERRORS) as exc:
        log.debug("rerank unavailable: %s", exc)
        order = await llm_rerank(query, docs) or []
    if not order:
        return pool
    ranked = []
    for idx, score in order:
        if 0 <= idx < len(head):
            h = head[idx]
            h.rerank_score = score
            if cold:  # cold start: strength is decided by the reranker (planS3 §5)
                h.strong = score >= T.RERANK_STRONG or (h.strong and h.ref_kind == "entity")
            else:
                h.strong = h.strong or score >= T.RERANK_STRONG
            ranked.append(h)
    placed = {i for i, _ in order}
    missing = [h for i, h in enumerate(head) if i not in placed]
    return ranked + missing + tail


async def _mmr(store: Store, hits: list[Hit], k: int) -> list[Hit]:
    if len(hits) <= k:
        return hits
    vecs: dict[tuple[str, int], np.ndarray] = {}
    for kind in KINDS:
        ids = [h.ref_id for h in hits if h.ref_kind == kind]
        for rid, v in (await store.vectors(kind, ids)).items():
            vecs[(kind, rid)] = v
    n = len(hits)
    rel = [1.0 - i / n for i in range(n)]  # rank-based relevance (post-rerank order)
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < k:
        def score(i: int) -> float:
            v = vecs.get((hits[i].ref_kind, hits[i].ref_id))
            div = max((float(v @ vecs[(hits[j].ref_kind, hits[j].ref_id)]) for j in selected
                       if v is not None and (hits[j].ref_kind, hits[j].ref_id) in vecs), default=0.0)
            return T.MMR_LAMBDA * rel[i] - (1 - T.MMR_LAMBDA) * div

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return [hits[i] for i in selected]


async def _exhaustive(store: Store, plan: QueryPlan, result: RetrievalResult) -> None:
    """planS3 §4.4: every episode in the interval, time-ordered. Episodes not yet
    drafted are represented by their observations' captions (Rev 3)."""
    assert plan.interval is not None
    start, end = plan.interval
    result.exhaustive = True
    for ep in await store.episodes_in_interval(start, end):
        if ep["status"] == "closed" and ep["summary"]:
            title, body = ep["title"], ep["summary"]
        else:
            rows = await store.observations_for_episode(ep["id"])
            caps = list(dict.fromkeys(r["caption"] for r in rows if r["caption"]))[:12]
            if not caps:
                continue
            title = " / ".join(sorted({r["app"] for r in rows if r["app"]})) or "activity"
            body = "\n".join(f"- {c}" for c in caps)
        when = f"{to_local(ep['started_at']):%H:%M}–{to_local(ep['ended_at']):%H:%M}"
        result.hits.append(Hit("episode", ep["id"], f"{when} {title}", body, score=1.0, strong=True,
                               occurred_at=ep["started_at"], occurred_to=ep["ended_at"]))
    digest = await store.get_digest(day_key(start))
    if digest:
        result.digest = digest["text"]


async def brute_verify(store: Store, plan: QueryPlan, exclude: set[tuple[str, int]]) -> str | None:
    """planS3 §8: would any row the pipeline did not return have been strong?"""
    df_info = await store.df_lookup(plan.terms)
    qvec = await try_embed(plan.raw)
    for kind in KINDS:
        ids: list[int] = []
        for q in plan.queries or [plan.raw]:
            ids += await store.fts_ids(kind, q, limit=500)
        dense: dict[int, float] = {}
        if qvec is not None:
            dense = dict(await store.dense_search(kind, qvec, limit=10_000, min_sim=T.DENSE_HIT_THRESHOLD))
        ids = list(dict.fromkeys(ids + list(dense)))
        hits = await materialize(store, kind, [i for i in ids if (kind, i) not in exclude])
        for rid, h in hits.items():
            if dense.get(rid, 0.0) >= T.DENSE_HIT_THRESHOLD or coverage(plan.terms, h.text, df_info) >= T.COVERAGE_STRONG:
                return f"{kind}:{rid}"
    return None
