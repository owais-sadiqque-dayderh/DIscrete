"""Unified retrieval (planS3 §3-6)."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed, rerank
from ..process.confidence import half_life_days
from ..store import Store
from ..timeutil import age_days, now, to_local
from .query import QueryPlan, build_plan
from .rerank import llm_rerank

log = logging.getLogger("discrete.retrieval.search")

_OPENED_QUERY = re.compile(r"\b(?:open|opened|read|visit|visited|click|clicked|saw|seen)\b", re.I)

KINDS = ("episode", "observation", "claim", "entity")

# Field-weighted BM25 (planS3 §4.1); argument order = FTS column order in db.py.
_FTS = {
    "observation": ("observations_fts", "bm25(observations_fts, 4.0, 2.0, 3.0, 2.0, 1.0)"),
    "episode": ("episodes_fts", "bm25(episodes_fts, 5.0, 3.0)"),
    "claim": ("claims_fts", "bm25(claims_fts, 6.0, 2.0)"),
    "entity": ("entities_fts", "bm25(entities_fts, 6.0, 6.0, 2.0)"),
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
    dense_sim: float = 0.0
    rerank_score: float | None = None
    strong: bool = False
    untrusted: bool = False
    modality: str = ""
    confidence: float = 0.0
    status: str = ""
    occurred_at: float = 0.0
    occurred_to: float = 0.0
    claim_kind: str = ""
    url: str = ""

    @property
    def dense_hit(self) -> bool:
        return self.dense_sim >= T.DENSE_HIT_THRESHOLD

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


@dataclass
class RetrievalResult:
    query: str
    plan: QueryPlan
    hits: list[Hit] = field(default_factory=list)
    exhaustive: bool = False
    digest: str = ""

    def of(self, kind: str) -> list[Hit]:
        return [h for h in self.hits if h.ref_kind == kind]


def _fts_match(text: str) -> str:
    return tk.fts_query(tk.content_tokens(text) or tk.tokens(text))


def _fts(conn, kind: str, match: str, limit: int) -> list[int]:
    if not match:
        return []
    table, rank = _FTS[kind]
    try:
        rows = conn.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY {rank} LIMIT ?",
                            (match, limit)).fetchall()
    except Exception as exc:  # malformed MATCH expression
        log.debug("fts %s: %s", table, exc)
        return []
    return [int(r[0]) for r in rows]


async def fts_ids(store: Store, kind: str, text: str, *, limit: int = 20) -> list[int]:
    return await store._call(_fts, store.conn, kind, _fts_match(text), limit)


async def _embed_query(text: str) -> list[float] | None:
    try:
        return (await embed([text]))[0]
    except (NotConfigured, *EXPECTED_ERRORS):
        return None


# ------------------------------------------------------------------ materialize

async def _materialize(store: Store, kind: str, ids: list[int]) -> dict[int, Hit]:
    out: dict[int, Hit] = {}
    if not ids:
        return out
    ph = ",".join("?" * len(ids))
    if kind == "claim":
        for r in await store.q(f"SELECT * FROM claims WHERE id IN ({ph}) AND status='live'", ids):
            ev = await store.q1(
                "SELECT quote FROM evidence WHERE claim_id=? AND relation='supports' AND quote!='' LIMIT 1", (r["id"],)
            )
            body = r["text"] + (f"\n> {ev[0]}" if ev else "")
            out[r["id"]] = Hit("claim", r["id"], r["text"][:120], body, modality=r["modality"],
                               confidence=r["confidence"], status=r["status"], occurred_at=r["occurred_from"],
                               occurred_to=r["occurred_to"], claim_kind=r["kind"], untrusted=bool(ev))
    elif kind == "observation":
        for r in await store.q(f"SELECT * FROM observations WHERE id IN ({ph}) AND origin='user'", ids):
            title = f"{r['app']} — {r['window_title']}".strip(" —") or r["caption"][:80]
            body = "\n".join(p for p in (r["caption"], tk.truncate(r["screen_text"], 1500)) if p)
            out[r["id"]] = Hit("observation", r["id"], title[:120], body, occurred_at=r["started_at"],
                               occurred_to=r["ended_at"], untrusted=True, url=r["url"] or "")
    elif kind == "episode":
        for r in await store.q(f"SELECT * FROM episodes WHERE id IN ({ph}) AND status='closed'", ids):
            out[r["id"]] = Hit("episode", r["id"], r["title"] or "episode", r["summary"],
                               occurred_at=r["started_at"], occurred_to=r["ended_at"])
    elif kind == "entity":
        for r in await store.q(f"SELECT * FROM entities WHERE id IN ({ph}) AND merged_into IS NULL", ids):
            out[r["id"]] = Hit("entity", r["id"], r["name"], r["summary"] or r["name"])
    return out


def _in_interval(h: Hit, interval: tuple[float, float] | None) -> bool:
    """planS3 §4.1: episodes, observations and events outside the interval are dropped."""
    if not interval or h.ref_kind == "entity" or (h.ref_kind == "claim" and h.claim_kind != "event"):
        return True
    start, end = interval
    return h.occurred_at < end and (h.occurred_to or h.occurred_at) >= start


def _coverage(terms: list[str], text: str, df_info: tuple[dict[str, int], int]) -> float:
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


# ---------------------------------------------------------------------- recall

def _noop(*_a) -> None:
    pass


async def retrieve(store: Store, query: str, *, field_text: str = "", progress=None) -> RetrievalResult:
    """`progress(stage, detail)` is an optional UI hook (Bubble's overlay)."""
    progress = progress or _noop
    # The planner and the query embedding are independent API calls: run them together.
    qvec_task = asyncio.create_task(_embed_query(query))
    plan = await build_plan(query)
    result = RetrievalResult(query=query, plan=plan)
    progress("searching", plan.intent)
    if plan.intent == "edit":
        qvec_task.cancel()
        return result

    if plan.intent == "enumerate":
        await _enumerate(store, plan, result, await qvec_task)
        return result

    if plan.interval and plan.interval[1] - plan.interval[0] <= T.EXHAUSTIVE_MAX_SPAN_S and plan.intent in ("temporal", "recall"):
        await _exhaustive(store, plan, result)
        if result.hits:
            qvec_task.cancel()
            return result

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
        rankings = [await fts_ids(store, kind, q, limit=T.CANDIDATE_POOL) for q in plan.queries[:8]]
        dense: dict[int, float] = {}
        if qvec is not None:
            dense = dict(await store.dense_search(kind, qvec, limit=T.CANDIDATE_POOL))
            rankings.append(list(dense))
        fused = tk.rrf([r for r in rankings if r], k=T.RRF_K)
        ids = [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[: T.CANDIDATE_POOL]]
        hits = await _materialize(store, kind, ids)
        for rid in ids:
            h = hits.get(rid)
            if h is None or (kind, rid) in seen or not _in_interval(h, plan.interval):
                continue
            h.score = fused[rid] * (1.0 if plan.interval else _recency(h))
            h.dense_sim = dense.get(rid, 0.0)
            h.coverage = _coverage(plan.terms, h.text, df_info)
            h.echo = tk.jaccard(tk.trigrams(field_text), tk.trigrams(h.body)) if field_text else 0.0
            if h.echo >= T.ECHO_THRESHOLD:
                continue
            if not cold and h.coverage < T.COVERAGE_FLOOR and not h.dense_hit:
                continue
            h.strong = h.dense_hit or (not cold and h.coverage >= T.COVERAGE_STRONG)
            pool.append(h)

    pool.sort(key=lambda h: -h.score)
    progress("ranking", len(pool))
    pool = await _rerank(query, pool, cold)
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
    missing = [h for i, h in enumerate(head) if i not in {i for i, _ in order}]
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
    from ..timeutil import day_key

    digest = await store.get_digest(day_key(start))
    if digest:
        result.digest = digest["text"]


def _canon_url(u: str) -> str:
    return u.split("#")[0].rstrip("/") if u else ""


def _wants_opened_items(plan: QueryPlan) -> bool:
    return bool(_OPENED_QUERY.search(plan.raw))


def _is_listing_row(row: dict) -> bool:
    title = (row.get("window_title") or "").lower()
    if tk.is_listing_url(row.get("url") or ""):
        return True
    if title in ("hacker news",) or title.startswith("hacker news "):
        cap = (row.get("caption") or row.get("screen_text") or "").lower()
        if "switched" not in cap and "clicked on a link" not in cap:
            return True
    return False


def _hit_caption(h: Hit) -> str:
    return h.body.split("\n", 1)[1] if "\n" in h.body else ""


def shape_enumerate_items(hits: list[Hit], plan: QueryPlan) -> list[Hit]:
    """Pick the lines that answer a 'list what I opened/read' question (planS3 §17).

    Raw enumerate hits include listing pages (★) and every same-app page within the
    trail window (→). For opened-item questions, drop the listings and pages that are
    clearly not article visits, and relabel generic window titles from captions."""
    if not hits:
        return hits
    labeled: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        cap = _hit_caption(h)
        label = h.title if not tk.is_generic_title(h.title) else tk.page_label(
            h.title, url=h.url, caption=cap, site=h.body.partition(" · ")[0])
        ident = tk.page_identity(h.title, url=h.url, caption=cap)
        if ident in seen:
            continue
        seen.add(ident)
        labeled.append(Hit(h.ref_kind, h.ref_id, label, h.body, score=h.score, strong=h.strong,
                           untrusted=h.untrusted, status=h.status, occurred_at=h.occurred_at,
                           occurred_to=h.occurred_to, url=h.url))

    if not _wants_opened_items(plan):
        return labeled

    listings = [h for h in labeled if _is_listing_row(
        {"url": h.url, "window_title": h.title, "caption": _hit_caption(h), "screen_text": h.body})]

    def opened_item(h: Hit) -> bool:
        cap = _hit_caption(h)
        if h.status == "inferred":
            return True
        return tk.navigation_caption(cap)

    items = [h for h in labeled if opened_item(h)]

    if listings and not items:
        extras: list[Hit] = []
        for h in listings:
            cap = _hit_caption(h)
            if "click" not in cap.lower():
                continue
            for title in tk.hn_headlines(h.body):
                key = tk.norm_key(title)
                if key in seen:
                    continue
                seen.add(key)
                extras.append(Hit("observation", h.ref_id, title, h.body, strong=True, untrusted=True,
                                  status="inferred", occurred_at=h.occurred_at, occurred_to=h.occurred_to))
        if extras:
            return extras

    def keep(h: Hit) -> bool:
        cap = _hit_caption(h)
        low = h.title.lower()
        if low.startswith("untitled document") or low.startswith("arc\n"):
            return False
        if "hacker news" == low or (low == "hacker news" and not h.url):
            return False
        if low in ("exploring local files",):
            return False
        if any(x in low for x in ("terminal window", "finder", ".cache", "credits", "openrouter", "google.com")):
            return False
        if "blueprint" in low and "hacker news" not in cap.lower():
            return False
        if h.title.startswith("The user"):
            return False
        if tk.is_generic_title(h.title) and not tk.navigation_caption(cap):
            return False
        return True

    deduped: list[Hit] = []
    seen_labels: set[str] = set()
    for h in items:
        if not keep(h):
            continue
        key = tk.norm_key(h.title)
        if key in seen_labels:
            continue
        seen_labels.add(key)
        deduped.append(h)
    return deduped or labeled


async def _enumerate(store: Store, plan: QueryPlan, result: RetrievalResult, qvec) -> None:
    """planS3 §17: a time-ordered list of distinct pages/windows, bypassing TOP_K and MMR.

    Matching alone cannot answer "what did I open from X": the pages opened from a
    listing never mention the listing. So the list is the *trail*: every page that
    matched the query (status "matched") plus the pages visited in the same app within
    `ENUMERATE_TRAIL_S` after a match (status "trail"). The writer decides which answer
    the question; the marks tell it how each page got there (insights.md §7.34)."""
    result.exhaustive = True
    ids: list[int] = []
    subject = " ".join(plan.queries[:8]).strip()
    if subject:
        for q in plan.queries[:8]:
            ids += await fts_ids(store, "observation", q, limit=T.ENUMERATE_POOL)
        # Embed the subject, not the raw request: its scaffolding pulls in unrelated rows.
        qvec = await _embed_query(subject) if subject != plan.raw else qvec
        if qvec is not None:
            ids += [i for i, s in await store.dense_search("observation", qvec, limit=T.ENUMERATE_POOL)
                    if s >= T.DENSE_HIT_THRESHOLD]
    elif plan.interval:
        # "List the pages I opened today": no subject, so everything in the interval matches.
        ids = [r["id"] for r in await store.q(
            "SELECT id FROM observations WHERE origin='user' AND started_at >= ? AND started_at < ?",
            plan.interval)]
    hits = await _materialize(store, "observation", list(dict.fromkeys(ids)))
    matched = {i: h for i, h in hits.items() if _in_interval(h, plan.interval)}
    if not matched:
        return
    m_rows = {r["id"]: r for r in await store.q(
        f"SELECT id, app, window_title, url, site FROM observations WHERE id IN ({','.join('?' * len(matched))})",
        list(matched))}
    # A named source ("hacker news", "figma") is matched by page identity: title, site, URL.
    # Captions and screen text mention the previous page too ("switched from Hacker News
    # to…"), so content matches alone marked every page as the source (insights.md §7.36).
    # A topic with no identity matches anywhere ("articles about rust") keeps content matches.
    # Per query string, not their union: the planner adds sibling queries, and the union
    # of all their words is never mostly present in one title.
    wants = [w for w in (set(tk.content_tokens(q)) for q in plan.queries[:8]) if w]
    if wants:
        def ident_cov(r) -> float:
            toks = set(tk.tokens(f"{r['window_title']} {r['site']} {r['url']}"))
            return max(len(w & toks) / len(w) for w in wants)

        by_ident = {i for i, r in m_rows.items() if ident_cov(r) >= 0.5}
        if by_ident:
            matched = {i: h for i, h in matched.items() if i in by_ident}
            m_rows = {i: r for i, r in m_rows.items() if i in by_ident}
    apps = {r["app"] for r in m_rows.values()}

    spans: list[list[float]] = []
    for t in sorted(h.occurred_at for h in matched.values()):
        if spans and t <= spans[-1][1]:
            spans[-1][1] = t + T.ENUMERATE_TRAIL_S
        else:
            spans.append([t, t + T.ENUMERATE_TRAIL_S])
    rows = []
    for a, b in spans:
        b = min(b, plan.interval[1]) if plan.interval else b
        rows += await store.q(
            "SELECT id, app, site, url, window_title, caption, screen_text, started_at, ended_at FROM observations "
            "WHERE origin='user' AND started_at >= ? AND started_at <= ? ORDER BY started_at", (a, b))

    visits: list[dict] = []
    by_ident: dict[str, dict] = {}
    for r in rows:
        is_match = r["id"] in matched
        if not is_match and r["app"] not in apps:
            continue
        url = _canon_url(r["url"] or "")
        title = r["window_title"] or r["app"]
        screen_text = r["screen_text"] or ""
        ident = tk.page_identity(title, url=url, caption=r["caption"] or "", screen_text=screen_text)
        v = by_ident.get(ident)
        if v is None and url:
            v = by_ident.get(url)
            if v is not None and v["url"] and v["url"] != url:
                v = None
        if v is None:
            label = tk.page_label(title, url=url, caption=r["caption"] or "", screen_text=screen_text,
                                  site=r["site"] or "")
            v = {"id": r["id"], "title": label, "raw_title": title, "app": r["app"], "site": r["site"], "url": url,
                 "caption": r["caption"], "screen_text": screen_text,
                 "first": r["started_at"], "last": r["ended_at"], "matched": False}
            visits.append(v)
            by_ident[ident] = v
            if url:
                by_ident[url] = v
        if url and not v["url"]:
            v["url"], v["site"] = url, r["site"]
        v["last"] = max(v["last"], r["ended_at"])
        v["matched"] = v["matched"] or is_match
        if r["caption"] and (not v["caption"] or tk.navigation_caption(r["caption"])):
            v["caption"] = r["caption"]
            v["title"] = tk.page_label(v["raw_title"], url=v["url"], caption=r["caption"],
                                       screen_text=screen_text, site=v["site"] or "")

    for v in visits[: T.ENUMERATE_MAX]:
        body = " · ".join(x for x in (v["site"] or v["app"], v["url"]) if x)
        if v["caption"]:
            body += f"\n{tk.truncate(v['caption'], 160)}"
        result.hits.append(Hit("observation", v["id"], v["title"][:140], body, strong=True, untrusted=True,
                               status="matched" if v["matched"] else "trail", url=v["url"],
                               occurred_at=v["first"], occurred_to=v["last"]))


async def brute_verify(store: Store, plan: QueryPlan, exclude: set[tuple[str, int]]) -> str | None:
    """planS3 §8: would any row the pipeline did not return have been strong?"""
    df_info = await store.df_lookup(plan.terms)
    qvec = await _embed_query(plan.raw)
    for kind in KINDS:
        ids: list[int] = []
        for q in plan.queries or [plan.raw]:
            ids += await fts_ids(store, kind, q, limit=500)
        dense: dict[int, float] = {}
        if qvec is not None:
            dense = dict(await store.dense_search(kind, qvec, limit=10_000, min_sim=T.DENSE_HIT_THRESHOLD))
        ids = list(dict.fromkeys(ids + list(dense)))
        hits = await _materialize(store, kind, [i for i in ids if (kind, i) not in exclude])
        for rid, h in hits.items():
            if dense.get(rid, 0.0) >= T.DENSE_HIT_THRESHOLD or _coverage(plan.terms, h.text, df_info) >= T.COVERAGE_STRONG:
                return f"{kind}:{rid}"
    return None
