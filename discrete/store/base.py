"""Connections, locking and shared write helpers for every store piece.

Two connections: a writer, serialized by `_lock`, and a query-only reader with
its own lock. WAL lets the reader run while the writer is mid-transaction, so
Bubble's retrieval never queues behind ingest, drafting or cleanup. Each call
runs its SQL in a worker thread. Multi-statement writes use `_tx` (BEGIN
IMMEDIATE, rollback on any error).
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..config import db_path, embedding_endpoint
from ..timeutil import now
from .db import connect


def rowid(cur: sqlite3.Cursor) -> int:
    """The id of the row an INSERT just created."""
    assert cur.lastrowid is not None
    return cur.lastrowid


class StoreBase:
    def __init__(self, conn: sqlite3.Connection, rconn: sqlite3.Connection | None = None):
        self.conn = conn
        self.rconn = rconn or conn
        self._lock = asyncio.Lock()
        self._rlock = asyncio.Lock() if rconn is not None else self._lock
        self._dense: dict[str, tuple[tuple[int, str, str], np.ndarray, np.ndarray]] = {}
        self._dense_version: dict[str, int] = {}
        self._dirty_kinds: set[str] = set()
        self._alias_cache: tuple[int, list[sqlite3.Row]] | None = None
        self._alias_version = 0

    @classmethod
    async def open(cls, path: str | None = None):
        path = str(path or db_path())
        conn = connect(path)
        try:
            rconn = None if path == ":memory:" else connect(path, read_only=True)
        except BaseException:
            conn.close()
            raise
        return cls(conn, rconn)

    async def close(self) -> None:
        async with self._lock:
            if self.rconn is not self.conn:
                async with self._rlock:
                    self.rconn.close()
            self.conn.close()

    async def _call(self, fn, *args, **kwargs):
        """Run `fn` on the writer connection."""
        async with self._lock:
            return await self._thread(fn, *args, **kwargs)

    async def _read(self, fn, *args, **kwargs):
        """Run `fn` (reads only) on the reader connection."""
        async with self._rlock:
            return await self._thread(fn, *args, **kwargs)

    @staticmethod
    async def _thread(fn, *args, **kwargs):
        # Cancelling to_thread does not stop SQLite. Keep the connection lock
        # until the worker finishes, including a transaction already committing.
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    @contextmanager
    def _tx(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            from .pages import sync_pages

            sync_pages(self.conn)
        except BaseException:
            self.conn.execute("ROLLBACK")
            self._dirty_kinds.clear()
            raise
        self.conn.execute("COMMIT")
        self._publish_vectors()

    def _publish_vectors(self) -> None:
        """Invalidate dense caches only after commit: a reader that rebuilt the cache
        between the change and the commit would otherwise keep stale vectors."""
        for kind in self._dirty_kinds:
            self._dense_version[kind] = self._dense_version.get(kind, 0) + 1
        self._dirty_kinds.clear()

    async def q(self, sql: str, args: Sequence = ()) -> list[sqlite3.Row]:
        return await self._read(lambda: self.rconn.execute(sql, args).fetchall())

    async def q1(self, sql: str, args: Sequence = ()) -> sqlite3.Row | None:
        return await self._read(lambda: self.rconn.execute(sql, args).fetchone())

    async def scalar(self, sql: str, args: Sequence = ()) -> Any:
        """First column of an aggregate query's single row."""
        row = await self.q1(sql, args)
        assert row is not None, sql
        return row[0]

    async def execute(self, sql: str, args: Sequence = ()) -> None:
        def run():
            with self._tx() as conn:
                conn.execute(sql, args)
        await self._call(run)

    # ------------------------------------------------------ shared write helpers

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

    def _vectors_changed(self, conn, ref_kind: str) -> None:
        """Bump the stored per-kind vector version, so a dense cache in another process
        (the CLI, an eval run beside the daemon) notices the change too."""
        conn.execute("INSERT INTO meta(key, value) VALUES (?, '1') "
                     "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1", (f"vectors:{ref_kind}",))
        self._dirty_kinds.add(ref_kind)

    def _drop_vector(self, conn, ref_kind: str, ref_id: int) -> None:
        conn.execute("DELETE FROM embeddings WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))
        conn.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))
        self._vectors_changed(conn, ref_kind)
        if not conn.in_transaction:
            self._publish_vectors()

    def _write_vector(self, conn, ref_kind: str, ref_id: int, vec: Sequence[float], text_hash: str) -> None:
        conn.execute(
            """INSERT INTO embeddings(ref_kind, ref_id, model, dim, vector, text_hash, created_at)
               VALUES (?,?,?,?,?,?,?) ON CONFLICT(ref_kind, ref_id) DO UPDATE SET
               model=excluded.model, dim=excluded.dim, vector=excluded.vector,
               text_hash=excluded.text_hash, created_at=excluded.created_at""",
            (ref_kind, ref_id, embedding_endpoint().model or "unknown", len(vec),
             np.asarray(vec, dtype=np.float32).tobytes(), text_hash, now()),
        )
        conn.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))
        self._vectors_changed(conn, ref_kind)
        if not conn.in_transaction:
            self._publish_vectors()

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
