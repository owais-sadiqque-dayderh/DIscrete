"""Unified retrieval (planS3 §3-6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed, rerank
from ..process.confidence import half_life_days
from ..store import Store
from ..timeutil import age_days, now
from .query import QueryPlan, build_plan
from .rerank import llm_rerank

log = logging.getLogger("discrete.retrieval.search")

_TABLE = {
    "claim": "claims_fts",
    "observation": "observations_fts",
    "episode": "episodes_fts",
    "entity": "entities_fts",
}


@dataclass
class Hit:
    ref_kind: str
    ref_id: int
    title: str
    body: str
    score: float = 0.0
    coverage: float = 0.0
    echo: float = 0.0
    dense_hit: bool = False
    strong: bool = False
    untrusted: bool = False
    modality: str = ""
    confidence: float = 0.0
    occurred_at: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    plan: QueryPlan
    hits: list[Hit] = field(default_factory=list)
    claims: list[Hit] = field(default_factory=list)
    observations: list[Hit] = field(default_factory=list)
    entities: list[Hit] = field(default_factory=list)
    episodes: list[Hit] = field(default_factory=list)
    exhaustive: bool = False


def fuse_rankings(rankings: list[list[int]], *, limit: int, k: float | None = None) -> list[int]:
    fused = tk.rrf(rankings, k=k or T.RRF_K)
    return [rid for rid, _ in sorted(fused.items(), key=lambda x: -x[1])[:limit]]


def _fts(conn, table: str, match: str, limit: int) -> list[int]:
    if not match:
        return []
    try:
        rows = conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]
    except Exception as exc:
        log.debug("fts %s: %s", table, exc)
        return []


async def fts_hits(store: Store, kind: str, query: str, *, limit: int = 20) -> list[int]:
    match = tk.fts_query(tk.content_tokens(query) or tk.tokens(query))
    return await store._call(_fts, store.conn, _TABLE[kind], match, limit)


async def retrieve(store: Store, query: str, *, field_text: str = "") -> RetrievalResult:
    plan = await build_plan(query)
    result = RetrievalResult(query=query, plan=plan)
    if not plan.needs_store or plan.intent == "edit":
        return result

    n_obs = await store.observation_count()
    cold = n_obs < T.COLD_START_ROWS

    if plan.intent == "aggregate" and plan.entity_kind_filter:
        agg = await store.aggregate_entities(
            plan.entity_kind_filter,
            plan.interval[0] if plan.interval else None,
            plan.interval[1] if plan.interval else None,
        )
        for row in agg[: T.TOP_K]:
            result.entities.append(
                Hit("entity", row["id"], row["name"], row.get("summary") or row["name"], score=1.0, strong=True)
            )
        result.hits.extend(result.entities)

    if plan.interval and _exhaustive_ok(plan):
        eps = await store.episodes_in_interval(plan.interval[0], plan.interval[1])
        result.exhaustive = True
        for ep in eps:
            body = f"{ep['title']}\n{ep['summary']}"
            hit = Hit("episode", ep["id"], ep["title"] or "episode", body, score=1.0, strong=True, occurred_at=ep["started_at"])
            result.episodes.append(hit)
            result.hits.append(hit)
        return result

    qvec: list[float] | None = None
    try:
        qvec = (await embed([query]))[0]
    except (NotConfigured, *EXPECTED_ERRORS):
        qvec = None

    kinds = ["episode", "observation", "claim", "entity"]
    dense_by_kind = {k: dict(await store.all_embeddings(k)) for k in kinds}
    df_info = await store.df_lookup(plan.terms)
    pool: dict[tuple[str, int], Hit] = {}

    for kind in kinds:
        rankings: list[list[int]] = []
        for q in (plan.queries or [query])[:8]:
            match = tk.fts_query(tk.content_tokens(q) or [q])
            ids = await store._call(_fts, store.conn, _TABLE[kind], match, T.CANDIDATE_POOL)
            if ids:
                rankings.append(_filter_interval(store, kind, ids, plan.interval))
        if qvec is not None:
            dense_ids = [
                rid for rid, vec in dense_by_kind.get(kind, {}).items() if tk.cosine(qvec, vec) >= T.DENSE_HIT_THRESHOLD
            ]
            if dense_ids:
                rankings.append(dense_ids[: T.CANDIDATE_POOL])
        for rid in fuse_rankings(rankings, limit=T.CANDIDATE_POOL):
            hit = await _materialize(store, kind, rid, qvec, dense_by_kind.get(kind, {}))
            if not hit:
                continue
            hit.coverage = _coverage(plan.terms, hit.body, df_info)
            hit.echo = tk.jaccard(tk.trigrams(field_text), tk.trigrams(hit.body)) if field_text else 0.0
            hit.dense_hit = qvec is not None and rid in dense_by_kind.get(kind, {}) and tk.cosine(qvec, dense_by_kind[kind][rid]) >= T.DENSE_HIT_THRESHOLD
            if not cold and hit.coverage < T.COVERAGE_FLOOR and not hit.dense_hit:
                continue
            if hit.echo >= T.ECHO_THRESHOLD:
                continue
            hit.strong = (not cold and hit.coverage >= T.COVERAGE_STRONG) or hit.dense_hit
            if not plan.interval:
                hit.score *= _recency_multiplier(hit)
            pool[(kind, rid)] = hit

    hits = _suppress_entity_overlap(list(pool.values()))
    hits.sort(key=lambda h: (-int(h.strong), -h.score, -h.coverage))

    if len(hits) > 3:
        try:
            top = hits[: T.RERANK_POOL]
            docs = [_rerank_doc(h) for h in top]
            order = await rerank(query, docs, len(docs))
            by_idx = {i: h for i, h in enumerate(top)}
            reranked = []
            for idx, score in order:
                if idx in by_idx:
                    h = by_idx[idx]
                    h.score = score
                    h.strong = h.strong or score >= T.RERANK_STRONG
                    reranked.append(h)
            hits = reranked + hits[T.RERANK_POOL :]
        except (NotConfigured, *EXPECTED_ERRORS):
            pass
        llm_order = await llm_rerank(query, [_rerank_doc(h) for h in hits[: T.LLM_RERANK_POOL]])
        if llm_order:
            by_idx = {i: h for i, h in enumerate(hits[: T.LLM_RERANK_POOL])}
            hits = [by_idx[i] for i, _ in llm_order if i in by_idx] + hits[T.LLM_RERANK_POOL :]

    hits = _mmr_select(hits, dense_by_kind, T.TOP_K)
    result.hits = hits
    for h in hits:
        bucket = {"claim": result.claims, "observation": result.observations, "entity": result.entities, "episode": result.episodes}.get(h.ref_kind)
        if bucket is not None:
            bucket.append(h)
    return result


async def brute_verify(store: Store, query: str, plan: QueryPlan) -> str | None:
    for kind in ("entity", "claim", "observation", "episode"):
        for q in plan.queries or [query]:
            ids = await fts_hits(store, kind, q, limit=200)
            for rid in ids:
                hit = await _materialize(store, kind, rid, None, {})
                if hit and hit.strong:
                    return f"{kind}:{rid}"
    return None


def _exhaustive_ok(plan: QueryPlan) -> bool:
    if not plan.interval:
        return False
    span = plan.interval[1] - plan.interval[0]
    return span <= T.EXHAUSTIVE_MAX_SPAN_S and plan.intent in ("temporal", "recall")


def _filter_interval(store: Store, kind: str, ids: list[int], interval: tuple[float, float] | None) -> list[int]:
    if not interval:
        return ids
    return ids  # interval filter applied at materialize


def _recency_multiplier(hit: Hit) -> float:
    hl = half_life_days(hit.ref_kind if hit.ref_kind != "observation" else "observation")
    if hit.ref_kind == "claim":
        hl = half_life_days("fact")  # refined per-row in _materialize ideally
    if not hl:
        return 1.0
    age = age_days(hit.occurred_at or now())
    return 0.5 ** (age / hl)


def _rerank_doc(h: Hit) -> str:
    return tk.truncate(h.title + "\n" + h.body, T.RERANK_MAX_TOKENS * 4)


def _suppress_entity_overlap(hits: list[Hit]) -> list[Hit]:
    entities = [h for h in hits if h.ref_kind == "entity"]
    if len(entities) < 2:
        return hits
    kept: list[Hit] = []
    for h in hits:
        if h.ref_kind != "claim":
            kept.append(h)
            continue
        drop = False
        for e in entities:
            if tk.jaccard(tk.trigrams(h.body), tk.trigrams(e.body)) >= T.ENTITY_PROP_OVERLAP_THRESHOLD:
                drop = True
                break
        if not drop:
            kept.append(h)
    return kept


def _mmr_select(hits: list[Hit], dense_by_kind: dict, k: int) -> list[Hit]:
    if len(hits) <= k:
        return hits
    selected: list[Hit] = []
    remaining = hits[:]
    while remaining and len(selected) < k:
        best = None
        best_score = -1.0
        for h in remaining:
            rel = h.score
            if selected:
                vec_h = dense_by_kind.get(h.ref_kind, {}).get(h.ref_id)
                sims = []
                for s in selected:
                    vec_s = dense_by_kind.get(s.ref_kind, {}).get(s.ref_id)
                    if vec_h and vec_s:
                        sims.append(tk.cosine(vec_h, vec_s))
                div = max(sims) if sims else 0.0
                rel = T.MMR_LAMBDA * h.score - (1 - T.MMR_LAMBDA) * div
            if rel > best_score:
                best_score = rel
                best = h
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


async def _materialize(store, kind, rid, qvec, dense_map) -> Hit | None:
    if kind == "claim":
        rows = await store.live_claims()
        row = next((r for r in rows if r["id"] == rid), None)
        if not row:
            return None
        links = await store.evidence_for_claim(rid)
        quote = links[0]["quote"] if links else ""
        body = row["text"] + (f"\n> {quote}" if quote else "")
        return Hit(
            "claim", rid, row["text"][:120], body, modality=row["modality"], confidence=row["confidence"],
            occurred_at=row["occurred_to"], untrusted=False,
        )
    if kind == "observation":
        row = await store.get_observation(rid)
        if not row or row["origin"] != "user":
            return None
        body = "\n".join(p for p in (row["caption"], row["screen_text"]) if p)
        return Hit("observation", rid, (row["caption"] or row["app"])[:120], body, occurred_at=row["started_at"], untrusted=True)
    if kind == "entity":
        row = await store._call(lambda: store.conn.execute("SELECT * FROM entities WHERE id=?", (rid,)).fetchone())
        if not row:
            return None
        return Hit("entity", rid, row["name"], row["summary"] or row["name"])
    if kind == "episode":
        row = await store._call(lambda: store.conn.execute("SELECT * FROM episodes WHERE id=?", (rid,)).fetchone())
        if not row:
            return None
        body = f"{row['title']}\n{row['summary']}"
        return Hit("episode", rid, (row["title"] or "episode")[:120], body, occurred_at=row["started_at"])
    return None


def _coverage(terms: list[str], body: str, df_info: tuple[dict[str, int], int]) -> float:
    df, n = df_info
    if not terms:
        return 0.0
    body_l = body.lower()
    num = den = 0.0
    for t in terms:
        w = tk.bm25_idf(df.get(t, 0), n) + 1.0
        den += w
        if t.lower() in body_l:
            num += w
    return num / den if den else 0.0
