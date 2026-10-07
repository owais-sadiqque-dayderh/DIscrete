"""Entities, aliases, mentions and recurrence candidates (planS2 §10)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from .. import textkit as tk
from ..timeutil import now
from .base import StoreBase, rowid


class EntitiesMixin(StoreBase):
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
        """Every live alias, cached until the alias table changes (checked with a
        one-row fingerprint query, so another process's writes are seen too)."""
        fp = await self.q1("SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM entity_aliases")
        key = hash(tuple(fp or ()))
        if self._alias_cache and self._alias_cache[0] == key:
            return self._alias_cache[1]
        rows = await self.q(
            "SELECT a.alias_norm, a.norm_key, a.entity_id, a.alias FROM entity_aliases a "
            "JOIN entities e ON e.id=a.entity_id WHERE e.merged_into IS NULL"
        )
        self._alias_cache = (key, rows)
        return rows

    async def get_entity(self, entity_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM entities WHERE id=?", (entity_id,))

    async def get_entities(self, ids: Sequence[int], *, live_only: bool = False) -> list[sqlite3.Row]:
        if not ids:
            return []
        extra = " AND merged_into IS NULL" if live_only else ""
        return await self.q(f"SELECT * FROM entities WHERE id IN ({','.join('?' * len(ids))}){extra}", list(ids))

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
                eid = rowid(conn.execute(
                    "INSERT INTO entities(name, kind, created_at, updated_at) VALUES (?,?,?,?)",
                    (name.strip(), kind, ts, ts),
                ))
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
        await self.record_mentions([(entity_id, ref_kind, ref_id)])

    async def record_mentions(self, items: Sequence[tuple[int, str, int]]) -> int:
        """Insert many mentions in one transaction; returns how many were new."""
        if not items:
            return 0

        def _rec():
            n = 0
            with self._tx() as conn:
                ts = now()
                for eid, kind, rid in items:
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO mentions(entity_id, ref_kind, ref_id, created_at) VALUES (?,?,?,?)",
                        (eid, kind, rid, ts),
                    )
                    if cur.rowcount:
                        conn.execute("UPDATE entities SET updated_at=? WHERE id=?", (ts, eid))
                        n += 1
            return n

        return await self._call(_rec)

    async def entity_candidates(self) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM entity_candidates")

    async def delete_candidate(self, key: str) -> None:
        await self.execute("DELETE FROM entity_candidates WHERE name_norm=?", (key,))

    async def update_entity_page(self, entity_id: int, summary: str, *, kind: str = "") -> None:
        """New page text; `kind` replaces an 'unknown' kind only."""

        def _u():
            with self._tx() as conn:
                ts = now()
                conn.execute("UPDATE entities SET summary=?, page_updated_at=? WHERE id=?", (summary, ts, entity_id))
                if kind:
                    conn.execute("UPDATE entities SET kind=? WHERE id=? AND kind='unknown'", (kind, entity_id))
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
        return [dict(r) for r in await self.q(sql, args)]

    async def top_entities(self, limit: int = 8) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT e.name, COUNT(DISTINCT m.ref_id) n FROM entities e JOIN mentions m ON m.entity_id=e.id "
            "WHERE e.merged_into IS NULL GROUP BY e.id ORDER BY n DESC LIMIT ?", (limit,))
