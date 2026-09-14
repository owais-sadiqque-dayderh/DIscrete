"""Storage layer — all SQLite access goes through here.

One connection, serialized by an asyncio lock; each public method runs its
SQL in a worker thread. Multi-statement writes use `_tx` (BEGIN IMMEDIATE,
rollback on any error).
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np

from . import textkit as tk
from .config import T, db_path, embedding_endpoint
from .db import connect
from .process.confidence import compute_confidence, modality_of
from .timeutil import day_key, now

log = logging.getLogger("discrete.store")


@dataclass
class ObservationDraft:
    source: str = "screen"
    origin: str = "user"
    event_kind: str = "click"
    session_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    site: str = ""
    screen_text: str = ""
    text_source: str = ""
    caption: str = ""
    entities: list[str] = field(default_factory=list)
    keys: int = 0
    clicks: int = 0
    scrolls: int = 0
    clipboard: int = 0
    frame_phash: str = ""
    frame_path: str = ""
    # Stage 1 pHash tier-0 duplicate: bump the matching row, never insert.
    frame_duplicate: bool = False

    def dedup_key(self) -> str:
        return tk.stable_obs_key(self.app, self.window_title, self.url, self.screen_text)

    def world_text(self) -> str:
        """Text counted in term statistics (planS3 §5): world text only."""
        return "\n".join(p for p in (self.window_title, self.site, self.caption, self.screen_text) if p)


@dataclass
class ClaimDraft:
    kind: str
    text: str
    context: str = ""
    reasoning: str = ""
    modality: str = "passive"
    episode_id: int | None = None
    occurred_from: float = 0.0
    occurred_to: float = 0.0
    entities: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # [{observation_id, quote}]


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = asyncio.Lock()
        self._dense: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        self._dense_version: dict[str, int] = {}

    @classmethod
    async def open(cls, path: str | None = None) -> Store:
        return cls(connect(str(path or db_path())))

    async def close(self) -> None:
        async with self._lock:
            self.conn.close()

    async def _call(self, fn, *args, **kwargs):
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    @contextmanager
    def _tx(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    async def q(self, sql: str, args: Sequence = ()) -> list[sqlite3.Row]:
        return await self._call(lambda: self.conn.execute(sql, args).fetchall())

    async def q1(self, sql: str, args: Sequence = ()) -> sqlite3.Row | None:
        return await self._call(lambda: self.conn.execute(sql, args).fetchone())

    async def execute(self, sql: str, args: Sequence = ()) -> None:
        await self._call(lambda: self.conn.execute(sql, args))

    # --------------------------------------------------------------- helpers

    def _bump_df(self, conn, tokens: Iterable[str], delta: int = 1) -> None:
        for t in {t for t in tokens if len(t) <= 64}:
            conn.execute(
                "INSERT INTO token_df(token, df) VALUES (?, MAX(0, ?)) "
                "ON CONFLICT(token) DO UPDATE SET df = MAX(0, token_df.df + ?)",
                (t, delta, delta),
            )

    def _enqueue(self, conn, ref_kind: str, ref_id: int) -> None:
        conn.execute(
            "INSERT INTO index_outbox(ref_kind, ref_id, enqueued_at, attempts) VALUES (?, ?, ?, 0) "
            "ON CONFLICT(ref_kind, ref_id) DO UPDATE SET enqueued_at=excluded.enqueued_at, attempts=0",
            (ref_kind, ref_id, now()),
        )

    def _drop_vector(self, conn, ref_kind: str, ref_id: int) -> None:
        conn.execute("DELETE FROM embeddings WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))
        conn.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))
        self._dense_version[ref_kind] = self._dense_version.get(ref_kind, 0) + 1

    def _metric(self, conn, name: str, delta: float = 1.0) -> None:
        ts = now()
        conn.execute(
            "INSERT INTO metrics(name, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = metrics.value + ?, updated_at = ?",
            (name, delta, ts, delta, ts),
        )

    async def bump_metric(self, name: str, delta: float = 1.0) -> None:
        await self._call(self._metric, self.conn, name, delta)

    # ------------------------------------------------------------------ meta

    async def get_meta(self, key: str, default: str = "") -> str:
        row = await self.q1("SELECT value FROM meta WHERE key=?", (key,))
        return row[0] if row else default

    async def set_meta(self, key: str, value: str) -> None:
        await self.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    async def stats(self) -> dict[str, Any]:
        def _stats():
            c = self.conn
            one = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
            rng = c.execute("SELECT MIN(started_at), MAX(started_at) FROM observations").fetchone()
            coverage = {}
            for kind, table, live in (
                ("observation", "observations", "origin='user'"),
                ("episode", "episodes", "status='closed'"),
                ("claim", "claims", "status='live'"),
                ("entity", "entities", "merged_into IS NULL"),
            ):
                n = one(f"SELECT COUNT(*) FROM {table} WHERE {live}")
                have = one(
                    f"SELECT COUNT(*) FROM {table} t WHERE {live} AND EXISTS "
                    f"(SELECT 1 FROM embeddings e WHERE e.ref_kind='{kind}' AND e.ref_id=t.id)"
                )
                coverage[kind] = 1.0 if n == 0 else round(have / n, 3)
            return {
                "observations": one("SELECT COUNT(*) FROM observations"),
                "observations_agent": one("SELECT COUNT(*) FROM observations WHERE origin!='user'"),
                "duplicates_collapsed": one("SELECT COALESCE(SUM(dup_count-1),0) FROM observations"),
                "episodes": one("SELECT COUNT(*) FROM episodes"),
                "episodes_by_status": dict(c.execute("SELECT status, COUNT(*) FROM episodes GROUP BY status").fetchall()),
                "claims_live": one("SELECT COUNT(*) FROM claims WHERE status='live'"),
                "claims_by_kind": dict(c.execute("SELECT kind, COUNT(*) FROM claims WHERE status='live' GROUP BY kind").fetchall()),
                "entities": one("SELECT COUNT(*) FROM entities WHERE merged_into IS NULL"),
                "entity_candidates": one("SELECT COUNT(*) FROM entity_candidates"),
                "open_gaps": one("SELECT COUNT(*) FROM gaps WHERE status='open'"),
                "embeddings": one("SELECT COUNT(*) FROM embeddings"),
                "outbox_pending": one("SELECT COUNT(*) FROM index_outbox"),
                "index_coverage": coverage,
                "first_observation": rng[0],
                "last_observation": rng[1],
                "metrics": dict(c.execute("SELECT name, value FROM metrics").fetchall()),
            }

        return await self._call(_stats)

    async def df_lookup(self, toks: Sequence[str]) -> tuple[dict[str, int], int]:
        def _df():
            uniq = list({t for t in toks if t})
            n = self.conn.execute("SELECT COUNT(*) FROM observations WHERE origin='user'").fetchone()[0]
            if not uniq:
                return {}, n
            rows = self.conn.execute(
                f"SELECT token, df FROM token_df WHERE token IN ({','.join('?' * len(uniq))})", uniq
            ).fetchall()
            return {r[0]: r[1] for r in rows}, n

        return await self._call(_df)

    # ----------------------------------------------------------- observations

    async def add_observation(self, draft: ObservationDraft) -> tuple[int | None, bool]:
        """Insert, or collapse into an existing row. Returns (id, is_new)."""

        def _bump(conn, oid: int) -> None:
            conn.execute(
                "UPDATE observations SET dup_count=dup_count+1, ended_at=MAX(ended_at, ?), updated_at=?, "
                "keys=keys+?, clicks=clicks+?, scrolls=scrolls+?, clipboard=clipboard+? WHERE id=?",
                (draft.ended_at or now(), now(), draft.keys, draft.clicks, draft.scrolls, draft.clipboard, oid),
            )
            self._metric(conn, "observations_collapsed")

        def _add():
            with self._tx() as conn:
                if draft.frame_duplicate:
                    row = conn.execute(
                        "SELECT id FROM observations WHERE app=? AND window_title=? AND url=? AND started_at>=? "
                        "ORDER BY id DESC LIMIT 1",
                        (draft.app, draft.window_title, draft.url, (draft.started_at or now()) - 3600),
                    ).fetchone()
                    if row:
                        _bump(conn, row[0])
                        return row[0], False
                    return None, False
                key = draft.dedup_key()
                if not draft.screen_text.strip() and not draft.caption.strip():
                    return None, False
                text_hash = tk.content_hash(key)
                site = draft.site or tk.site_from_url(draft.url)
                # Tier 1: exact on the stable key (planS2 §2.1).
                row = conn.execute(
                    "SELECT id FROM observations WHERE text_hash=? ORDER BY id DESC LIMIT 1", (text_hash,)
                ).fetchone()
                if row:
                    _bump(conn, row[0])
                    return row[0], False
                # Tier 2: near-duplicate within the same app and site (planS2 §2.2).
                mh_new = tk.minhash(key, permutations=T.DEDUP_MINHASH_PERMUTATIONS)
                short = len(tk.word_shingles(key)) < T.DEDUP_SHORT_SHINGLES
                if not short:
                    recent = conn.execute(
                        "SELECT o.id, m.sig FROM observations o JOIN obs_minhash m ON m.obs_id=o.id "
                        "WHERE o.app=? AND o.site=? ORDER BY o.id DESC LIMIT 200",
                        (draft.app, site),
                    ).fetchall()
                    for oid, sig in recent:
                        if tk.minhash_jaccard(mh_new, pickle.loads(sig)) >= T.DEDUP_JACCARD_THRESHOLD:
                            _bump(conn, oid)
                            return oid, False
                ts = draft.started_at or now()
                cur = conn.execute(
                    """INSERT INTO observations(
                      source, origin, event_kind, session_id, started_at, ended_at,
                      app, bundle_id, window_title, url, site, screen_text, text_source,
                      caption, entities, keys, clicks, scrolls, clipboard,
                      frame_phash, frame_path, content_hash, text_hash, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        draft.source, draft.origin, draft.event_kind, draft.session_id, ts, draft.ended_at or ts,
                        draft.app, draft.bundle_id, draft.window_title, draft.url, site, draft.screen_text,
                        draft.text_source, draft.caption, json.dumps(draft.entities), draft.keys, draft.clicks,
                        draft.scrolls, draft.clipboard, draft.frame_phash, draft.frame_path,
                        tk.content_hash(draft.world_text()), text_hash, now(), now(),
                    ),
                )
                oid = cur.lastrowid
                conn.execute("INSERT INTO obs_minhash(obs_id, sig) VALUES (?, ?)", (oid, pickle.dumps(mh_new)))
                if draft.origin == "user":
                    self._bump_df(conn, tk.content_tokens(draft.world_text()))
                    self._enqueue(conn, "observation", oid)
                self._metric(conn, "observations_ingested")
                return oid, True

        return await self._call(_add)

    async def get_observation(self, oid: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM observations WHERE id=?", (oid,))

    async def get_observations(self, ids: Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        rows = await self.q(f"SELECT * FROM observations WHERE id IN ({','.join('?' * len(ids))})", list(ids))
        order = {i: n for n, i in enumerate(ids)}
        return sorted(rows, key=lambda r: order.get(r["id"], 1 << 30))

    async def observations_for_episode(self, episode_id: int, *, user_only: bool = True) -> list[sqlite3.Row]:
        extra = " AND origin='user'" if user_only else ""
        return await self.q(
            f"SELECT * FROM observations WHERE episode_id=?{extra} ORDER BY started_at, id", (episode_id,)
        )

    async def observations_in_interval(self, start: float, end: float, *, limit: int = 200) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM observations WHERE origin='user' AND started_at < ? AND ended_at >= ? "
            "ORDER BY started_at LIMIT ?",
            (end, start, limit),
        )

    async def assign_episode(self, obs_ids: Sequence[int], episode_id: int) -> None:
        if obs_ids:
            await self.execute(
                f"UPDATE observations SET episode_id=? WHERE id IN ({','.join('?' * len(obs_ids))})",
                [episode_id, *obs_ids],
            )

    async def observation_count(self) -> int:
        row = await self.q1("SELECT COUNT(*) FROM observations WHERE origin='user'")
        return row[0]

    # --------------------------------------------------------------- episodes
    # Lifecycle: open → ready → closed (drafted) | failed (after DRAFT_MAX_ATTEMPTS).

    async def open_episode(self, session_id: str, started_at: float) -> int:
        def _open():
            cur = self.conn.execute(
                "INSERT INTO episodes(session_id, started_at, ended_at, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'open', ?, ?)",
                (session_id, started_at, started_at, now(), now()),
            )
            return cur.lastrowid

        return await self._call(_open)

    async def get_episode(self, episode_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM episodes WHERE id=?", (episode_id,))

    async def mark_episode_ready(self, episode_id: int) -> None:
        """Close an episode for drafting; its interval is its observations' span."""
        await self.execute(
            """UPDATE episodes SET status='ready', updated_at=?,
                 started_at=COALESCE((SELECT MIN(started_at) FROM observations WHERE episode_id=episodes.id), started_at),
                 ended_at=COALESCE((SELECT MAX(ended_at) FROM observations WHERE episode_id=episodes.id), ended_at)
               WHERE id=? AND status='open'""",
            (now(), episode_id),
        )

    async def recover_open_episodes(self, *, older_than_s: float = 0.0) -> list[int]:
        """Open episodes whose tracker died with its process become ready."""
        rows = await self.q(
            "SELECT e.id FROM episodes e WHERE e.status='open' AND "
            "COALESCE((SELECT MAX(ended_at) FROM observations o WHERE o.episode_id=e.id), e.ended_at) <= ?",
            (now() - older_than_s,),
        )
        for r in rows:
            await self.mark_episode_ready(r[0])
        return [r[0] for r in rows]

    async def begin_draft(self, episode_id: int) -> bool:
        """Claim a ready episode for drafting (guards against double drafting)."""

        def _claim():
            cur = self.conn.execute(
                "UPDATE episodes SET status='drafting', draft_attempts=draft_attempts+1, updated_at=? "
                "WHERE id=? AND status='ready'",
                (now(), episode_id),
            )
            return cur.rowcount == 1

        return await self._call(_claim)

    async def draft_failed(self, episode_id: int) -> None:
        await self.execute(
            "UPDATE episodes SET status=CASE WHEN draft_attempts>=? THEN 'failed' ELSE 'ready' END, updated_at=? "
            "WHERE id=?",
            (T.DRAFT_MAX_ATTEMPTS, now(), episode_id),
        )

    async def close_episode(self, episode_id: int, *, title: str, summary: str, apps: list[str],
                            sites: list[str], engaged_fraction: float) -> None:
        def _close():
            with self._tx() as conn:
                conn.execute(
                    "UPDATE episodes SET title=?, summary=?, apps=?, sites=?, engaged_fraction=?, "
                    "status='closed', updated_at=? WHERE id=?",
                    (title, summary, json.dumps(apps), json.dumps(sites), engaged_fraction, now(), episode_id),
                )
                self._enqueue(conn, "episode", episode_id)

        await self._call(_close)

    async def episodes_ready_for_draft(self, *, limit: int = 5) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM episodes WHERE status='ready' ORDER BY id LIMIT ?", (limit,))

    async def reset_stuck_drafting(self) -> None:
        await self.execute("UPDATE episodes SET status='ready' WHERE status='drafting'")

    async def episodes_for_day(self, day: str) -> list[sqlite3.Row]:
        from datetime import datetime

        from .timeutil import day_bounds, from_local

        start, end = day_bounds(from_local(datetime.strptime(day, "%Y-%m-%d")))
        return await self.q(
            "SELECT * FROM episodes WHERE started_at >= ? AND started_at < ? ORDER BY started_at", (start, end)
        )

    async def episodes_in_interval(self, start: float, end: float) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM episodes WHERE started_at < ? AND ended_at >= ? ORDER BY started_at", (end, start)
        )

    async def days_with_episodes(self) -> list[str]:
        rows = await self.q("SELECT started_at FROM episodes WHERE status='closed' AND summary!=''")
        return sorted({day_key(r[0]) for r in rows})

    # ------------------------------------------------------------------ claims

    def _recompute_confidence(self, conn, claim_id: int) -> float:
        """planS2 §8: per distinct supporting episode, its best evidence modality."""
        row = conn.execute("SELECT modality FROM claims WHERE id=?", (claim_id,)).fetchone()
        rows = conn.execute(
            "SELECT o.episode_id, o.keys, o.clicks, o.clipboard, o.started_at FROM evidence e "
            "JOIN observations o ON o.id=e.observation_id WHERE e.claim_id=? AND e.relation='supports'",
            (claim_id,),
        ).fetchall()
        best: dict[Any, str] = {}
        days: set[str] = set()
        for r in rows:
            mod = modality_of(r["keys"], r["clicks"], r["clipboard"])
            ep = r["episode_id"] if r["episode_id"] is not None else f"obs{id(r)}"
            if best.get(ep) != "engaged":
                best[ep] = mod
            days.add(day_key(r["started_at"]))
        mods = list(best.values())
        if row and row["modality"] == "stated":
            mods.append("stated")
        conf = compute_confidence(mods, max(1, len(days)))
        modality = "stated" if row and row["modality"] == "stated" else ("engaged" if "engaged" in mods else "passive")
        conn.execute("UPDATE claims SET confidence=?, modality=? WHERE id=?", (conf, modality, claim_id))
        return conf

    def _link(self, conn, claim_id: int, observation_id: int, relation: str, quote: str) -> None:
        conn.execute(
            """INSERT INTO evidence(claim_id, observation_id, relation, quote, created_at) VALUES (?,?,?,?,?)
               ON CONFLICT(claim_id, observation_id) DO UPDATE SET
                 relation=CASE WHEN excluded.relation='supports' THEN 'supports' ELSE evidence.relation END,
                 quote=CASE WHEN excluded.relation='supports' THEN excluded.quote ELSE evidence.quote END""",
            (claim_id, observation_id, relation, quote, now()),
        )

    def _cap_evidence(self, conn, claim_id: int) -> None:
        """planS2 §9: ≤ EVIDENCE_MAX_SUPPORTS supports; cooccurs dropped once ≥3 supports."""
        sup = conn.execute(
            "SELECT observation_id, created_at FROM evidence WHERE claim_id=? AND relation='supports' "
            "ORDER BY created_at DESC",
            (claim_id,),
        ).fetchall()
        if len(sup) >= 3:
            conn.execute("DELETE FROM evidence WHERE claim_id=? AND relation='cooccurs'", (claim_id,))
        if len(sup) > T.EVIDENCE_MAX_SUPPORTS:
            for r in sup[T.EVIDENCE_MAX_SUPPORTS:]:
                conn.execute("DELETE FROM evidence WHERE claim_id=? AND observation_id=?", (claim_id, r[0]))

    def _insert_claim(self, conn, draft: ClaimDraft, *, lineage_id: str, version: int,
                      refines_id: int | None = None, invalidates_id: int | None = None,
                      derived_from: list[int] | None = None) -> int:
        ts = now()
        cur = conn.execute(
            """INSERT INTO claims(lineage_id, version, kind, text, context, reasoning, modality,
                 occurred_from, occurred_to, valid_from, status, episode_id, refines_id, invalidates_id,
                 confidence, derived_from, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,'live',?,?,?,0,?,?,?)""",
            (lineage_id, version, draft.kind, tk.truncate(draft.text, T.CLAIM_MAX_CHARS), draft.context,
             draft.reasoning, draft.modality, draft.occurred_from or ts, draft.occurred_to or ts,
             draft.occurred_from or ts, draft.episode_id, refines_id, invalidates_id,
             json.dumps(derived_from or []), ts, ts),
        )
        cid = cur.lastrowid
        for ev in draft.evidence:
            self._link(conn, cid, int(ev["observation_id"]), "supports" if ev.get("quote") else "cooccurs",
                       str(ev.get("quote") or ""))
        self._cap_evidence(conn, cid)
        self._recompute_confidence(conn, cid)
        self._enqueue(conn, "claim", cid)
        return cid

    async def create_claim(self, draft: ClaimDraft, *, refines_id: int | None = None,
                           invalidates_id: int | None = None) -> int:
        def _create():
            with self._tx() as conn:
                return self._insert_claim(conn, draft, lineage_id=str(uuid4()), version=1,
                                          refines_id=refines_id, invalidates_id=invalidates_id)

        return await self._call(_create)

    async def get_claim(self, claim_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM claims WHERE id=?", (claim_id,))

    async def get_claims(self, ids: Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        return await self.q(f"SELECT * FROM claims WHERE id IN ({','.join('?' * len(ids))})", list(ids))

    async def reinforce_claim(self, claim_id: int, draft: ClaimDraft) -> float:
        """REINFORCE: add verified evidence, extend occurred_to; text untouched (planS2 §6.2)."""

        def _r():
            with self._tx() as conn:
                for ev in draft.evidence:
                    self._link(conn, claim_id, int(ev["observation_id"]), "supports", str(ev.get("quote") or ""))
                conn.execute(
                    "UPDATE claims SET occurred_to=MAX(occurred_to, ?), updated_at=? WHERE id=?",
                    (draft.occurred_to, now(), claim_id),
                )
                self._cap_evidence(conn, claim_id)
                return self._recompute_confidence(conn, claim_id)

        return await self._call(_r)

    async def invalidate_claim(self, claim_id: int, *, by_id: int, at: float) -> None:
        def _inv():
            with self._tx() as conn:
                conn.execute(
                    "UPDATE claims SET status='invalidated', valid_to=?, updated_at=? WHERE id=? AND status='live'",
                    (at, now(), claim_id),
                )
                conn.execute("UPDATE claims SET invalidates_id=? WHERE id=?", (claim_id, by_id))
                self._drop_vector(conn, "claim", claim_id)

        await self._call(_inv)

    async def flag_claims(self, ids: Sequence[int], flag: str) -> None:
        for cid in ids:
            await self.execute("UPDATE claims SET flag=? WHERE id=?", (flag, cid))

    async def live_claims(self, *, kind: str | None = None, limit: int | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM claims WHERE status='live'", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY updated_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return await self.q(sql, args)

    async def evidence_for_claim(self, claim_id: int) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM evidence WHERE claim_id=? ORDER BY relation='supports' DESC, created_at", (claim_id,)
        )

    async def consolidate_cluster(self, cluster: list[sqlite3.Row], outputs: list[dict]) -> list[int]:
        """planS2 §7.1: N inputs → a bounded set. Each output inherits evidence only
        from the inputs it derives from; inputs no output uses stay live."""
        cluster_ids = {r["id"] for r in cluster}
        by_id = {r["id"]: r for r in cluster}

        def _run():
            new_ids: list[int] = []
            with self._tx() as conn:
                replaced: dict[int, int] = {}
                for out in outputs[: len(cluster) + 2]:
                    inputs = [int(x) for x in (out.get("input_ids") or []) if int(x) in cluster_ids]
                    text = str(out.get("text") or "").strip()
                    if not inputs or not text:
                        continue
                    primary = by_id[inputs[0]]
                    ev = conn.execute(
                        f"SELECT observation_id, quote, relation FROM evidence WHERE claim_id IN "
                        f"({','.join('?' * len(inputs))})",
                        inputs,
                    ).fetchall()
                    lineage_version = conn.execute(
                        "SELECT MAX(version) FROM claims WHERE lineage_id=?", (primary["lineage_id"],)
                    ).fetchone()[0] or 1
                    draft = ClaimDraft(
                        kind=primary["kind"], text=text, context=primary["context"],
                        reasoning="consolidated", modality=primary["modality"], episode_id=primary["episode_id"],
                        occurred_from=min(by_id[i]["occurred_from"] for i in inputs),
                        occurred_to=max(by_id[i]["occurred_to"] for i in inputs),
                        evidence=[{"observation_id": r[0], "quote": r[1] if r[2] == "supports" else ""} for r in ev],
                    )
                    nid = self._insert_claim(conn, draft, lineage_id=primary["lineage_id"],
                                             version=lineage_version + 1, derived_from=inputs)
                    self._carry_mentions(conn, inputs, nid)
                    new_ids.append(nid)
                    for i in inputs:
                        replaced.setdefault(i, nid)
                ts = now()
                for old, nid in replaced.items():
                    conn.execute(
                        "UPDATE claims SET status='superseded', valid_to=?, superseded_by=?, updated_at=? WHERE id=?",
                        (ts, nid, ts, old),
                    )
                    self._drop_vector(conn, "claim", old)
            return new_ids

        return await self._call(_run)

    def _carry_mentions(self, conn, old_ids: list[int], new_id: int) -> None:
        for (eid,) in conn.execute(
            f"SELECT DISTINCT entity_id FROM mentions WHERE ref_kind='claim' AND ref_id IN ({','.join('?' * len(old_ids))})",
            old_ids,
        ).fetchall():
            conn.execute(
                "INSERT OR IGNORE INTO mentions(entity_id, ref_kind, ref_id, created_at) VALUES (?, 'claim', ?, ?)",
                (eid, new_id, now()),
            )

    async def supersede_unsupported(self, ttl_days: float) -> int:
        """planS2 §7.5; `stated` claims rest on user testimony and are exempt (Rev 3)."""
        cutoff = now() - ttl_days * 86400

        def _run():
            with self._tx() as conn:
                rows = conn.execute(
                    """SELECT c.id FROM claims c WHERE c.status='live' AND c.modality!='stated' AND c.created_at < ?
                       AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.claim_id=c.id AND e.relation='supports')""",
                    (cutoff,),
                ).fetchall()
                ts = now()
                for (cid,) in rows:
                    conn.execute(
                        "UPDATE claims SET status='superseded', valid_to=?, flag='unsupported', updated_at=? WHERE id=?",
                        (ts, ts, cid),
                    )
                    self._drop_vector(conn, "claim", cid)
                return len(rows)

        return await self._call(_run)

    async def lineage_anomalies(self) -> int:
        row = await self.q1(
            "SELECT COUNT(*) FROM (SELECT lineage_id FROM claims GROUP BY lineage_id HAVING MAX(version) > ?)",
            (T.LINEAGE_MAX_VERSIONS,),
        )
        return row[0]

    # ---------------------------------------------------------------- entities

    async def find_entity_by_alias(self, alias_norm: str) -> int | None:
        row = await self.q1(
            "SELECT a.entity_id FROM entity_aliases a JOIN entities e ON e.id=a.entity_id "
            "WHERE a.alias_norm=? AND e.merged_into IS NULL",
            (alias_norm,),
        )
        return row[0] if row else None

    async def find_entity_by_key(self, key: str) -> int | None:
        row = await self.q1(
            "SELECT a.entity_id FROM entity_aliases a JOIN entities e ON e.id=a.entity_id "
            "WHERE a.norm_key=? AND e.merged_into IS NULL LIMIT 1",
            (key,),
        )
        return row[0] if row else None

    async def all_entity_aliases(self) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT a.alias_norm, a.norm_key, a.entity_id, a.alias FROM entity_aliases a "
            "JOIN entities e ON e.id=a.entity_id WHERE e.merged_into IS NULL"
        )

    async def get_entity(self, entity_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM entities WHERE id=?", (entity_id,))

    def _refresh_entity_fts(self, conn, entity_id: int) -> None:
        row = conn.execute("SELECT name, summary FROM entities WHERE id=?", (entity_id,)).fetchone()
        aliases = " ".join(r[0] for r in conn.execute("SELECT alias FROM entity_aliases WHERE entity_id=?", (entity_id,)))
        conn.execute("DELETE FROM entities_fts WHERE rowid=?", (entity_id,))
        if row:
            conn.execute("INSERT INTO entities_fts(rowid, name, aliases, summary) VALUES (?,?,?,?)",
                         (entity_id, row[0], aliases, row[1]))

    async def create_entity(self, name: str, kind: str, key: str) -> int:
        def _create():
            with self._tx() as conn:
                ts = now()
                eid = conn.execute(
                    "INSERT INTO entities(name, kind, created_at, updated_at) VALUES (?,?,?,?)",
                    (name.strip(), kind, ts, ts),
                ).lastrowid
                conn.execute(
                    "INSERT OR IGNORE INTO entity_aliases(alias_norm, entity_id, alias, norm_key) VALUES (?,?,?,?)",
                    (tk.norm_key(name), eid, name.strip(), key),
                )
                self._refresh_entity_fts(conn, eid)
                self._enqueue(conn, "entity", eid)
                return eid

        return await self._call(_create)

    async def add_alias(self, entity_id: int, alias: str, key: str) -> None:
        def _add():
            with self._tx() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO entity_aliases(alias_norm, entity_id, alias, norm_key) VALUES (?,?,?,?)",
                    (tk.norm_key(alias), entity_id, alias.strip(), key),
                )
                self._refresh_entity_fts(conn, entity_id)

        await self._call(_add)

    async def record_mention(self, entity_id: int, ref_kind: str, ref_id: int) -> None:
        def _rec():
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO mentions(entity_id, ref_kind, ref_id, created_at) VALUES (?,?,?,?)",
                (entity_id, ref_kind, ref_id, now()),
            )
            if cur.rowcount:
                self.conn.execute("UPDATE entities SET updated_at=? WHERE id=?", (now(), entity_id))

        await self._call(_rec)

    async def bump_entity_candidate(self, name: str, key: str, episode_id: int) -> None:
        def _bump():
            ts = now()
            row = self.conn.execute("SELECT episodes FROM entity_candidates WHERE name_norm=?", (key,)).fetchone()
            if row:
                eps = set(tk.parse_json_list(row[0]))
                eps.add(episode_id)
                self.conn.execute("UPDATE entity_candidates SET episodes=?, last_seen=? WHERE name_norm=?",
                                  (json.dumps(sorted(eps)), ts, key))
            else:
                self.conn.execute(
                    "INSERT INTO entity_candidates(name_norm, name, episodes, first_seen, last_seen) VALUES (?,?,?,?,?)",
                    (key, name.strip(), json.dumps([episode_id]), ts, ts),
                )

        await self._call(_bump)

    async def entity_candidates(self) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM entity_candidates")

    async def delete_candidate(self, key: str) -> None:
        await self.execute("DELETE FROM entity_candidates WHERE name_norm=?", (key,))

    async def update_entity_page(self, entity_id: int, summary: str) -> None:
        def _u():
            with self._tx() as conn:
                ts = now()
                conn.execute("UPDATE entities SET summary=?, page_updated_at=? WHERE id=?", (summary, ts, entity_id))
                self._refresh_entity_fts(conn, entity_id)
                self._enqueue(conn, "entity", entity_id)

        await self._call(_u)

    async def entities_needing_pages(self, limit: int) -> list[sqlite3.Row]:
        return await self.q(
            """SELECT e.*, (SELECT COUNT(DISTINCT c.episode_id) FROM mentions m JOIN claims c
                          ON m.ref_kind='claim' AND c.id=m.ref_id WHERE m.entity_id=e.id) AS n_eps
               FROM entities e WHERE e.merged_into IS NULL AND e.page_updated_at < e.updated_at
               ORDER BY n_eps DESC LIMIT ?""",
            (limit,),
        )

    async def claims_for_entity(self, entity_id: int, *, limit: int = 30) -> list[sqlite3.Row]:
        return await self.q(
            """SELECT c.* FROM claims c JOIN mentions m ON m.ref_kind='claim' AND m.ref_id=c.id
               WHERE m.entity_id=? AND c.status='live' ORDER BY c.updated_at DESC LIMIT ?""",
            (entity_id, limit),
        )

    async def aggregate_entities(self, kind_filter: str, start: float | None, end: float | None) -> list[dict]:
        """planS3 §4.5: entities of a kind with mentions in the interval, by distinct episodes."""

        def _get():
            sql = """SELECT e.id, e.name, e.kind, e.summary, COUNT(DISTINCT c.episode_id) AS episodes
                     FROM entities e JOIN mentions m ON m.entity_id=e.id AND m.ref_kind='claim'
                     JOIN claims c ON c.id=m.ref_id
                     WHERE e.merged_into IS NULL"""
            args: list[Any] = []
            if kind_filter:
                sql += " AND e.kind=?"
                args.append(kind_filter)
            if start is not None and end is not None:
                sql += " AND c.occurred_from < ? AND c.occurred_to >= ?"
                args += [end, start]
            sql += " GROUP BY e.id ORDER BY episodes DESC LIMIT 50"
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

        return await self._call(_get)

    # -------------------------------------------------------------- embeddings

    async def reconcile_index_coverage(self) -> int:
        """Any live row without a vector or an outbox entry gets one (planS2 §11)."""

        def _run():
            n = 0
            with self._tx() as conn:
                for kind, table, live in (
                    ("observation", "observations", "origin='user'"),
                    ("claim", "claims", "status='live'"),
                    ("episode", "episodes", "status='closed'"),
                    ("entity", "entities", "merged_into IS NULL"),
                ):
                    rows = conn.execute(
                        f"""SELECT id FROM {table} t WHERE {live}
                            AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.ref_kind=? AND e.ref_id=t.id)
                            AND NOT EXISTS (SELECT 1 FROM index_outbox x WHERE x.ref_kind=? AND x.ref_id=t.id)""",
                        (kind, kind),
                    ).fetchall()
                    for (rid,) in rows:
                        self._enqueue(conn, kind, rid)
                        n += 1
            return n

        return await self._call(_run)

    async def outbox_batch(self, *, limit: int = 64) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM index_outbox WHERE attempts < ? ORDER BY enqueued_at LIMIT ?",
            (T.OUTBOX_MAX_ATTEMPTS, limit),
        )

    async def outbox_failed(self, refs: Sequence[tuple[str, int]]) -> None:
        for kind, rid in refs:
            await self.execute("UPDATE index_outbox SET attempts=attempts+1 WHERE ref_kind=? AND ref_id=?", (kind, rid))

    async def outbox_discard(self, kind: str, rid: int) -> None:
        await self.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (kind, rid))

    async def save_embeddings(self, items: Sequence[tuple[str, int, list[float], str]]) -> None:
        def _save():
            model = embedding_endpoint().model or "unknown"
            with self._tx() as conn:
                for kind, rid, vec, th in items:
                    conn.execute(
                        """INSERT INTO embeddings(ref_kind, ref_id, model, dim, vector, text_hash, created_at)
                           VALUES (?,?,?,?,?,?,?) ON CONFLICT(ref_kind, ref_id) DO UPDATE SET
                           model=excluded.model, dim=excluded.dim, vector=excluded.vector,
                           text_hash=excluded.text_hash, created_at=excluded.created_at""",
                        (kind, rid, model, len(vec), np.asarray(vec, dtype=np.float32).tobytes(), th, now()),
                    )
                    conn.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (kind, rid))
                    self._dense_version[kind] = self._dense_version.get(kind, 0) + 1

        await self._call(_save)

    async def dense_index(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """(ids, L2-normalized matrix) for a kind, cached until its vectors change."""
        version = self._dense_version.get(kind, 0)
        cached = self._dense.get(kind)
        if cached and cached[0] == version:
            return cached[1], cached[2]
        model = embedding_endpoint().model or "unknown"
        rows = await self.q("SELECT ref_id, vector FROM embeddings WHERE ref_kind=? AND model=?", (kind, model))
        if rows:
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        else:
            ids, mat = np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        self._dense[kind] = (version, ids, mat)
        return ids, mat

    async def dense_search(self, kind: str, qvec: Sequence[float], *, limit: int,
                           min_sim: float = -1.0) -> list[tuple[int, float]]:
        """Top rows by cosine, descending."""
        ids, mat = await self.dense_index(kind)
        if not len(ids):
            return []
        q = np.asarray(qvec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = mat @ q
        order = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in order if sims[i] >= min_sim]

    async def vectors(self, kind: str, ids: Sequence[int]) -> dict[int, np.ndarray]:
        all_ids, mat = await self.dense_index(kind)
        pos = {int(i): n for n, i in enumerate(all_ids)}
        return {i: mat[pos[i]] for i in ids if i in pos}

    # ------------------------------------------------------------- retention

    async def clear_old_screen_text(self, *, ttl_days: float) -> int:
        """planS2 §12: screen_text is cleared (row, caption, evidence kept); DF decremented."""
        if ttl_days <= 0:
            return 0
        cutoff = now() - ttl_days * 86400

        def _p():
            with self._tx() as conn:
                rows = conn.execute(
                    "SELECT id, screen_text, origin FROM observations WHERE started_at < ? AND screen_text != ''",
                    (cutoff,),
                ).fetchall()
                for r in rows:
                    if r["origin"] == "user":
                        kept = conn.execute(
                            "SELECT window_title, site, caption FROM observations WHERE id=?", (r["id"],)
                        ).fetchone()
                        before = set(tk.content_tokens(r["screen_text"]))
                        after = set(tk.content_tokens(" ".join(kept)))
                        self._bump_df(conn, before - after, -1)
                    conn.execute("UPDATE observations SET screen_text='', frame_path='' WHERE id=?", (r["id"],))
                    if r["origin"] == "user":
                        self._enqueue(conn, "observation", r["id"])
                return len(rows)

        return await self._call(_p)

    # ----------------------------------------------------------------- digests

    async def save_digest(self, day: str, text: str, n_episodes: int) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO digests(day, text, created_at, n_episodes) VALUES (?,?,?,?)",
            (day, text, now(), n_episodes),
        )

    async def get_digest(self, day: str) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM digests WHERE day=?", (day,))

    # -------------------------------------------------------------------- gaps

    async def record_gap(self, question: str) -> int:
        def _rec():
            return self.conn.execute(
                "INSERT INTO gaps(question, status, created_at) VALUES (?, 'open', ?)", (question, now())
            ).lastrowid

        return await self._call(_rec)

    async def record_retrieval_miss(self, query: str, missed: str, stage: str) -> None:
        await self.execute(
            "INSERT INTO retrieval_misses(query, missed, stage, created_at) VALUES (?,?,?,?)",
            (query, missed, stage, now()),
        )

    async def answer_gap(self, gap_id: int, answer: str) -> sqlite3.Row | None:
        def _ans():
            with self._tx() as conn:
                if not conn.execute("SELECT 1 FROM gaps WHERE id=? AND status='open'", (gap_id,)).fetchone():
                    return None
                conn.execute("UPDATE gaps SET status='answered', answer=?, answered_at=? WHERE id=?",
                             (answer.strip(), now(), gap_id))
                return conn.execute("SELECT * FROM gaps WHERE id=?", (gap_id,)).fetchone()

        return await self._call(_ans)

    async def open_gaps(self) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM gaps WHERE status='open' ORDER BY id")

    async def snapshot_id(self) -> str:
        row = await self.q1(
            "SELECT (SELECT COUNT(*) FROM observations), (SELECT COALESCE(MAX(id),0) FROM observations), "
            "(SELECT COUNT(*) FROM claims WHERE status='live')"
        )
        return f"obs{row[0]}-max{row[1]}-claims{row[2]}"
