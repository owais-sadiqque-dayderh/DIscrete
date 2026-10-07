"""Episodes and day digests. Lifecycle: open → ready → drafting → closed | failed; ready → merged."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..config import T, embedding_endpoint
from ..timeutil import day_bounds, day_key, from_local, now
from .base import StoreBase, rowid
from .tasks import cosine

# A split predecessor can wait briefly for its live successor to become ready.
# A module constant keeps the hold independent of model/retry configuration.
EPISODE_DRAFT_GRACE_S = 30.0
_WAITING_SUCCESSOR = """
    e.draft_attempts=0 AND COALESCE(e.ready_since, e.updated_at) > ?
    AND e.boundary_locked=0
    AND EXISTS (
      SELECT 1 FROM episodes successor WHERE successor.id=(
        SELECT id FROM episodes WHERE status!='merged' AND (started_at,id)>(e.started_at,e.id)
        ORDER BY started_at,id LIMIT 1)
      AND successor.status='open' AND successor.session_id=e.session_id
      AND successor.started_at-e.ended_at <= ?)
"""


@dataclass
class _ObservationCentroid:
    count: int
    vector: np.ndarray | None


class EpisodesMixin(StoreBase):
    async def open_episode(self, session_id: str, started_at: float) -> int:
        def _open():
            cur = self.conn.execute(
                "INSERT INTO episodes(session_id, started_at, ended_at, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'open', ?, ?)",
                (session_id, started_at, started_at, now(), now()),
            )
            return rowid(cur)

        return await self._call(_open)

    async def get_episode(self, episode_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM episodes WHERE id=?", (episode_id,))

    async def get_episodes(self, ids: list[int], *, closed_only: bool = False) -> list[sqlite3.Row]:
        if not ids:
            return []
        extra = " AND status='closed'" if closed_only else ""
        return await self.q(f"SELECT * FROM episodes WHERE id IN ({','.join('?' * len(ids))}){extra}", ids)

    async def mark_episode_ready(self, episode_id: int) -> None:
        """Close an episode for drafting; its interval is its observations' span."""
        def _ready() -> None:
            with self._tx() as conn:
                ts = now()
                conn.execute(
                    """UPDATE episodes SET status='ready', updated_at=?, ready_since=?,
                 started_at=COALESCE((SELECT MIN(started_at) FROM observations WHERE episode_id=episodes.id), started_at),
                 ended_at=COALESCE((SELECT MAX(ended_at) FROM observations WHERE episode_id=episodes.id), ended_at)
               WHERE id=? AND status='open'""",
                    (ts, ts, episode_id),
                )
        await self._call(_ready)

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
            with self._tx() as conn:
                self._merge_ready_episode(conn, episode_id)
                if conn.execute(f"SELECT 1 FROM episodes e WHERE e.id=? AND ({_WAITING_SUCCESSOR})",
                                (episode_id, now() - EPISODE_DRAFT_GRACE_S, T.EPISODE_MERGE_GAP_S)).fetchone():
                    return False
                cur = conn.execute(
                    "UPDATE episodes SET status='drafting', draft_attempts=draft_attempts+1, updated_at=? "
                    "WHERE id=? AND status='ready'",
                    (now(), episode_id),
                )
                return cur.rowcount == 1

        return await self._call(_claim)

    def _merge_ready_episode(self, conn: sqlite3.Connection, episode_id: int) -> None:
        """Merge only adjacent closed-for-drafting inputs, atomically with claiming.

        Tombstones keep IDs stable. A prior draft attempt/checkpoint/claim is a
        hard barrier, as are sessions, time gaps and the tracker size caps.
        Redrafted copies lock existing boundaries while resetting the retry budget.
        """
        def eligible(ep: sqlite3.Row | None) -> bool:
            return (ep is not None and ep['status'] == 'ready' and ep['draft_attempts'] == 0
                    and not ep['title'] and not ep['summary']
                    and not ep['boundary_locked']
                    and not conn.execute("SELECT 1 FROM episode_draft_chunks WHERE episode_id=?", (ep['id'],)).fetchone()
                    and not conn.execute("SELECT 1 FROM claims WHERE episode_id=?", (ep['id'],)).fetchone()
                    and not conn.execute("SELECT 1 FROM episode_tasks WHERE episode_id=?", (ep['id'],)).fetchone())

        def observations(ep: sqlite3.Row) -> _ObservationCentroid:
            rows = conn.execute(
                "SELECT o.id, v.vector FROM observations o LEFT JOIN embeddings v "
                "ON v.ref_kind='observation' AND v.ref_id=o.id AND v.model=? "
                "WHERE o.episode_id=? AND o.origin='user'",
                (embedding_endpoint().model or 'unknown', ep['id'])).fetchall()
            # Partial coverage can make a mixed-topic episode appear homogeneous.
            if not rows or any(r['vector'] is None for r in rows):
                return _ObservationCentroid(len(rows), None)
            vectors = [np.frombuffer(r['vector'], dtype=np.float32) for r in rows]
            if len({v.shape for v in vectors}) != 1:
                return _ObservationCentroid(len(rows), None)
            vectors = [v / max(float(np.linalg.norm(v)), 1e-9) for v in vectors]
            return _ObservationCentroid(len(rows), np.mean(vectors, axis=0))

        while True:
            ep = conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
            if not eligible(ep):
                return
            assert ep is not None
            near = []
            for comparison, order in (('<', 'DESC'), ('>', 'ASC')):
                neighbor = conn.execute(
                    f"SELECT * FROM episodes WHERE status!='merged' AND (started_at, id){comparison}(?,?) "
                    f"ORDER BY started_at {order}, id {order} LIMIT 1", (ep['started_at'], episode_id)).fetchone()
                if neighbor is not None:
                    near.append(neighbor)
            own = observations(ep)
            if own.vector is None:
                return
            merged = False
            for other in near:
                if not eligible(other) or other['session_id'] != ep['session_id']:
                    continue
                gap = max(ep['started_at'], other['started_at']) - min(ep['ended_at'], other['ended_at'])
                start, end = min(ep['started_at'], other['started_at']), max(ep['ended_at'], other['ended_at'])
                if gap > T.EPISODE_MERGE_GAP_S or end - start > T.EPISODE_MAX_S:
                    continue
                candidate = observations(other)
                if own.count + candidate.count > T.EPISODE_MAX_OBS or candidate.vector is None:
                    continue
                if cosine(own.vector, candidate.vector) < T.EPISODE_MERGE_SIM:
                    continue
                # Merging updates updated_at, but must not restart the grace timer.
                timers = [ts for ts in (ep['ready_since'], other['ready_since']) if ts is not None]
                ready_since = min(timers) if timers else min(ep['updated_at'], other['updated_at'])
                conn.execute("UPDATE observations SET episode_id=? WHERE episode_id=?", (episode_id, other['id']))
                conn.execute("UPDATE episodes SET started_at=?, ended_at=?, ready_since=?, updated_at=? WHERE id=?",
                             (start, end, ready_since, now(), episode_id))
                conn.execute("UPDATE episodes SET status='merged', updated_at=? WHERE id=?", (now(), other['id']))
                self._drop_vector(conn, 'episode', other['id'])
                self._metric(conn, 'episodes_merged')
                merged = True
                break
            if not merged:
                return

    async def defer_draft(self, episode_id: int) -> None:
        """Back to ready without spending an attempt: nothing was tried."""
        await self.execute(
            "UPDATE episodes SET status='ready', draft_attempts=MAX(0, draft_attempts-1), updated_at=? "
            "WHERE id=? AND status='drafting'", (now(), episode_id))

    async def draft_failed(self, episode_id: int) -> None:
        await self.execute(
            "UPDATE episodes SET status=CASE WHEN draft_attempts>=? THEN 'failed' ELSE 'ready' END, updated_at=? "
            "WHERE id=? AND status='drafting'",
            (T.DRAFT_MAX_ATTEMPTS, now(), episode_id),
        )

    async def draft_chunks(self, episode_id: int, input_hash: str) -> dict[int, dict]:
        """Read checkpoints for this exact episode input; discard obsolete results."""
        def _load():
            with self._tx() as conn:
                conn.execute("DELETE FROM episode_draft_chunks WHERE episode_id=? AND input_hash!=?",
                             (episode_id, input_hash))
                return {r["chunk_index"]: json.loads(r["result"]) for r in conn.execute(
                    "SELECT chunk_index, result FROM episode_draft_chunks WHERE episode_id=? AND input_hash=?",
                    (episode_id, input_hash))}

        return await self._call(_load)

    async def save_draft_chunk(self, episode_id: int, chunk_index: int, input_hash: str, result: dict) -> None:
        """Commit a successful model response before requesting the next chunk."""
        await self.execute(
            "INSERT INTO episode_draft_chunks(episode_id, chunk_index, input_hash, result) VALUES (?,?,?,?) "
            "ON CONFLICT(episode_id, chunk_index) DO UPDATE SET input_hash=excluded.input_hash, result=excluded.result",
            (episode_id, chunk_index, input_hash, json.dumps(result)),
        )

    async def retry_failed_drafts(self, *, older_than_s: float | None = None) -> None:
        """Give failed episodes another chance after a provider outage.

        With `older_than_s` (the daemon): one more attempt per round, waiting
        older_than_s, then twice as long each round (capped at 64x), so an episode
        that can never be drafted costs a handful of calls a day, not a steady stream.
        Without it (`discrete process --recover`): every failed episode, attempts reset."""
        if older_than_s is None:
            await self.execute("UPDATE episodes SET status='ready', draft_attempts=0, updated_at=? "
                               "WHERE status='failed'", (now(),))
            return
        await self.execute(
            "UPDATE episodes SET status='ready', updated_at=? WHERE status='failed' "
            "AND updated_at <= ? - ? * (1 << MIN(MAX(draft_attempts - ?, 0), 6))",
            (now(), now(), older_than_s, T.DRAFT_MAX_ATTEMPTS),
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
                self._assign_episode_task(conn, episode_id)

        await self._call(_close)

    async def episodes_ready_for_draft(self, *, limit: int = 5) -> list[sqlite3.Row]:
        return await self.q(f"SELECT e.* FROM episodes e WHERE e.status='ready' AND NOT ({_WAITING_SUCCESSOR}) "
                            "ORDER BY e.started_at, e.id LIMIT ?",
                            (now() - EPISODE_DRAFT_GRACE_S, T.EPISODE_MERGE_GAP_S, limit))

    async def reset_stuck_drafting(self) -> None:
        await self.execute("UPDATE episodes SET status='ready' WHERE status='drafting'")

    async def episodes_for_day(self, day: str) -> list[sqlite3.Row]:
        start, end = day_bounds(from_local(datetime.strptime(day, "%Y-%m-%d")))
        return await self.q(
            "SELECT * FROM episodes WHERE status!='merged' AND started_at >= ? AND started_at < ? ORDER BY started_at", (start, end)
        )

    async def episodes_in_interval(self, start: float, end: float) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM episodes WHERE status!='merged' AND started_at < ? AND ended_at >= ? ORDER BY started_at", (end, start)
        )

    async def days_with_episodes(self) -> list[str]:
        rows = await self.q("SELECT started_at FROM episodes WHERE status='closed' AND summary!=''")
        return sorted({day_key(r[0]) for r in rows})

    # ----------------------------------------------------------------- digests

    async def save_digest(self, day: str, text: str, n_episodes: int) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO digests(day, text, created_at, n_episodes) VALUES (?,?,?,?)",
            (day, text, now(), n_episodes),
        )

    async def get_digest(self, day: str) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM digests WHERE day=?", (day,))
