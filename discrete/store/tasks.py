"""Deterministic task grouping, using the episode embeddings already indexed.

Before the outbox has embedded a title/summary, token cosine is a conservative
fallback. Shared named entities and a bounded time gap are required either way.
No task title or embedding requires an extra model call.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter

import numpy as np

from .. import textkit as tk
from ..config import T, embedding_endpoint
from .base import StoreBase, rowid


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den else 0.0


def token_cosine(a: str, b: str) -> float:
    av, bv = Counter(tk.content_tokens(a)), Counter(tk.content_tokens(b))
    den = math.sqrt(sum(v * v for v in av.values()) * sum(v * v for v in bv.values()))
    return sum(v * bv[k] for k, v in av.items()) / den if den else 0.0


class TasksMixin(StoreBase):
    def _task_entities(self, conn: sqlite3.Connection, ep: sqlite3.Row, names: list[str]) -> set[str]:
        rows = conn.execute("SELECT entities, app, site FROM observations WHERE episode_id=? AND origin='user'",
                            (ep['id'],)).fetchall()
        ignored = {tk.entity_key(x) for r in rows for x in (r['app'], r['site']) if x}
        names = [*names, *(x for r in rows for x in tk.parse_json_list(r['entities']) if isinstance(x, str))]
        names += [r[0] for r in conn.execute(
            "SELECT DISTINCT e.name FROM mentions m JOIN entities e ON e.id=m.entity_id "
            "WHERE e.merged_into IS NULL AND ((m.ref_kind='episode' AND m.ref_id=?) OR "
            "(m.ref_kind='claim' AND m.ref_id IN (SELECT id FROM claims WHERE episode_id=?)))",
            (ep['id'], ep['id']))]
        return {k for x in names if isinstance(x, str) and (k := tk.entity_key(x)) and k not in ignored}

    def _assign_episode_task(self, conn: sqlite3.Connection, episode_id: int, *, names: list[str] | None = None) -> int | None:
        ep = conn.execute("SELECT * FROM episodes WHERE id=? AND status='closed'", (episode_id,)).fetchone()
        if ep is None:
            return None
        own = conn.execute("SELECT * FROM episode_tasks WHERE episode_id=?", (episode_id,)).fetchone()
        entities = self._task_entities(conn, ep, names or [])
        if own:
            entities |= set(tk.parse_json_list(own['entities']))
            # Established groups stay stable when the outbox refreshes a member.
            if conn.execute("SELECT COUNT(*) FROM episode_tasks WHERE task_id=?", (own['task_id'],)).fetchone()[0] > 1:
                return own['task_id']
        vector = conn.execute("SELECT vector, text_hash FROM embeddings WHERE ref_kind='episode' AND ref_id=? AND model=?",
                              (episode_id, embedding_endpoint().model or 'unknown')).fetchone()
        candidates = conn.execute(
            "SELECT e.*, et.task_id, et.entities AS task_entities, v.vector AS task_vector, v.text_hash AS task_hash "
            "FROM episode_tasks et JOIN episodes e ON e.id=et.episode_id JOIN tasks t ON t.id=et.task_id "
            "LEFT JOIN embeddings v ON v.ref_kind='episode' AND v.ref_id=e.id AND v.model=? "
            "WHERE e.status='closed' AND t.status='active' AND e.id!=? AND e.ended_at>=? AND e.started_at<=? "
            "ORDER BY e.started_at, e.id",
            (embedding_endpoint().model or 'unknown', episode_id, ep['started_at'] - T.TASK_WINDOW_S,
             ep['ended_at'] + T.TASK_WINDOW_S)).fetchall()
        best, best_sim = None, -1.0
        text = tk.truncate(f"{ep['title']}\n{ep['summary']}".strip(), T.EMBED_TEXT_CHARS)
        if vector is not None and vector['text_hash'] != tk.content_hash(text):
            vector = None
        for other in candidates:
            if own and other['task_id'] == own['task_id']:
                continue
            shared = entities & set(tk.parse_json_list(other['task_entities']))
            overlap = len(shared) / max(1, min(len(entities), len(tk.parse_json_list(other['task_entities']))))
            if not shared or overlap < T.TASK_ENTITY_OVERLAP:
                continue
            other_text = tk.truncate(f"{other['title']}\n{other['summary']}".strip(), T.EMBED_TEXT_CHARS)
            if (vector is not None and other['task_vector'] is not None
                    and other['task_hash'] == tk.content_hash(other_text)):
                sim = cosine(np.frombuffer(vector[0], dtype=np.float32),
                             np.frombuffer(other['task_vector'], dtype=np.float32))
            else:
                sim = token_cosine(text, other_text)
            if sim >= T.TASK_MATCH_SIM and sim > best_sim:
                best, best_sim = other['task_id'], sim
        if best is None:
            if own:
                return own['task_id']
            best = rowid(conn.execute("INSERT INTO tasks(title, first_seen, last_seen) VALUES (?,?,?)",
                                      (ep['title'].strip() or f"Episode {episode_id}", ep['started_at'], ep['ended_at'])))
        conn.execute("INSERT INTO episode_tasks(episode_id, task_id, entities) VALUES (?,?,?) "
                     "ON CONFLICT(episode_id) DO UPDATE SET task_id=excluded.task_id, entities=excluded.entities",
                     (episode_id, best, json.dumps(sorted(entities))))
        conn.execute("UPDATE tasks SET first_seen=MIN(first_seen, ?), last_seen=MAX(last_seen, ?) WHERE id=?",
                     (ep['started_at'], ep['ended_at'], best))
        if own and own['task_id'] != best:
            conn.execute("DELETE FROM tasks WHERE id=? AND NOT EXISTS "
                         "(SELECT 1 FROM episode_tasks WHERE task_id=tasks.id)", (own['task_id'],))
        return best

    async def backfill_tasks(self) -> int:
        """Group historical drafted episodes without re-drafting or embedding them."""
        def _fill():
            with self._tx() as conn:
                rows = conn.execute("SELECT id FROM episodes WHERE status='closed' AND NOT EXISTS "
                                    "(SELECT 1 FROM episode_tasks et WHERE et.episode_id=episodes.id) "
                                    "ORDER BY started_at, id").fetchall()
                for r in rows:
                    self._assign_episode_task(conn, r[0])
                return len(rows)
        return await self._call(_fill)

    async def list_tasks(self, *, limit: int = 20) -> list[sqlite3.Row]:
        return await self.q("SELECT t.*, COUNT(et.episode_id) AS n_episodes FROM tasks t "
                            "JOIN episode_tasks et ON et.task_id=t.id GROUP BY t.id "
                            "ORDER BY t.last_seen DESC, t.id DESC LIMIT ?", (max(0, limit),))

    async def task_siblings(self, episode_ids: list[int], *, limit: int,
                            interval: tuple[float, float] | None = None) -> list[sqlite3.Row]:
        if not episode_ids or limit <= 0:
            return []
        slots = ','.join('?' * len(episode_ids))
        time_filter = ' AND e.started_at < ? AND e.ended_at >= ?' if interval else ''
        time_args = [interval[1], interval[0]] if interval else []
        return await self.q(
            f"SELECT e.*, et.task_id FROM episodes e JOIN episode_tasks et ON et.episode_id=e.id "
            f"WHERE e.status='closed' AND e.id NOT IN ({slots}) AND et.task_id IN "
            f"(SELECT task_id FROM episode_tasks WHERE episode_id IN ({slots})){time_filter} "
            "ORDER BY e.started_at, e.id LIMIT ?", [*episode_ids, *episode_ids, *time_args, limit])
