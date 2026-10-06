"""Search indexes: field-weighted FTS, embeddings, the embedding outbox, dense search."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence

import numpy as np

from .. import textkit as tk
from ..config import T, embedding_endpoint
from ..timeutil import now
from .base import StoreBase

log = logging.getLogger("discrete.store.index")

KINDS = ("episode", "observation", "claim", "entity")

# Field-weighted BM25 (planS3 §4.1); argument order = FTS column order in db.py.
_FTS = {
    "observation": ("observations_fts", "bm25(observations_fts, 4.0, 2.0, 3.0, 2.0, 1.0, 4.0)"),
    "episode": ("episodes_fts", "bm25(episodes_fts, 5.0, 3.0)"),
    "claim": ("claims_fts", "bm25(claims_fts, 6.0, 2.0)"),
    "entity": ("entities_fts", "bm25(entities_fts, 6.0, 6.0, 2.0)"),
}

# What counts as a live row of each kind: only live rows are indexed or returned.
_LIVE = {
    "observation": ("observations", "origin='user'"),
    "claim": ("claims", "status='live'"),
    "episode": ("episodes", "status='closed'"),
    "entity": ("entities", "merged_into IS NULL"),
}


def fts_match(text: str) -> str:
    return tk.fts_query(tk.content_tokens(text) or tk.tokens(text))


class IndexMixin(StoreBase):
    async def fts_ids(self, kind: str, text: str, *, limit: int = 20) -> list[int]:
        match = fts_match(text)
        if not match:
            return []
        table, rank = _FTS[kind]

        def _run():
            try:
                rows = self.rconn.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY {rank} LIMIT ?",
                                          (match, limit)).fetchall()
            except sqlite3.OperationalError as exc:  # malformed MATCH expression
                log.debug("fts %s: %s", table, exc)
                return []
            return [int(r[0]) for r in rows]

        return await self._read(_run)

    async def live_rows(self, kind: str, ids: Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        table, live = _LIVE[kind]
        return await self.q(f"SELECT * FROM {table} WHERE id IN ({','.join('?' * len(ids))}) AND {live}", list(ids))

    async def reconcile_index_coverage(self) -> int:
        """Any live row without a vector or an outbox entry gets one (planS2 §11)."""

        def _run():
            n = 0
            with self._tx() as conn:
                for kind, (table, live) in _LIVE.items():
                    rows = conn.execute(
                        f"""SELECT id FROM {table} t WHERE {live}
                            AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.ref_kind=? AND e.ref_id=t.id AND e.model=?)
                            AND NOT EXISTS (SELECT 1 FROM index_outbox x WHERE x.ref_kind=? AND x.ref_id=t.id)""",
                        (kind, embedding_endpoint().model or "unknown", kind),
                    ).fetchall()
                    for (rid,) in rows:
                        self._enqueue(conn, kind, rid)
                        n += 1
            return n

        return await self._call(_run)

    async def outbox_batch(self, *, limit: int = 64) -> list[sqlite3.Row]:
        """Oldest first. An observation still waiting for its vision caption is skipped:
        embedding it now would embed the basic caption and then embed it again."""
        return await self.q(
            """SELECT x.* FROM index_outbox x
               LEFT JOIN observations o ON x.ref_kind='observation' AND o.id=x.ref_id
               WHERE x.attempts < ? AND COALESCE(o.enrich_state, '') != 'pending'
               ORDER BY x.enqueued_at LIMIT ?""",
            (T.OUTBOX_MAX_ATTEMPTS, limit),
        )

    async def outbox_failed(self, refs: Sequence[tuple[str, int]]) -> None:
        def _run():
            with self._tx() as conn:
                conn.executemany("UPDATE index_outbox SET attempts=attempts+1, enqueued_at=? WHERE ref_kind=? AND ref_id=?",
                                 [(now(), kind, rid) for kind, rid in refs])

        if refs:
            await self._call(_run)

    async def retry_failed_outbox(self, *, older_than_s: float = 0.0) -> None:
        await self.execute(
            "UPDATE index_outbox SET attempts=0, enqueued_at=? WHERE attempts>=? AND enqueued_at<=?",
            (now(), T.OUTBOX_MAX_ATTEMPTS, now() - older_than_s),
        )

    async def outbox_discard(self, kind: str, rid: int) -> None:
        await self.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (kind, rid))

    async def save_embeddings(self, items: Sequence[tuple[str, int, Sequence[float], str]]) -> None:
        def _save():
            with self._tx() as conn:
                for kind, rid, vec, th in items:
                    self._write_vector(conn, kind, rid, vec, th)

        if items:
            await self._call(_save)

    async def get_vector(self, kind: str, rid: int, text_hash: str) -> np.ndarray | None:
        """The stored vector for exactly this text, if it is already embedded."""
        row = await self.q1("SELECT vector FROM embeddings WHERE ref_kind=? AND ref_id=? AND text_hash=? AND model=?",
                            (kind, rid, text_hash, embedding_endpoint().model or "unknown"))
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    async def dense_index(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """(ids, L2-normalized matrix) for a kind, cached until its vectors change."""
        model = embedding_endpoint().model or "unknown"
        # The stored version also changes on writes by another process (the daemon
        # while the CLI or an eval run reads); the model, when EMBED_LM changes.
        stored = await self.q1("SELECT value FROM meta WHERE key=?", (f"vectors:{kind}",))
        version = (self._dense_version.get(kind, 0), model, stored[0] if stored else "0")
        cached = self._dense.get(kind)
        if cached and cached[0] == version:
            return cached[1], cached[2]
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
