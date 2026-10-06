"""Typed links between memories: causal trails ("this post led to that download")
and, later, any relation the graph needs. A link points from a source (an
observation, a claim or an entity) to a destination, with a confidence, a
one-sentence reason and the quotes it rests on. Links written by one decision carry
its event id, so reverting the event retires them."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from ..timeutil import now
from .claims import ClaimsMixin


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


class LinksMixin(ClaimsMixin):
    async def record_trail(self, claim_id: int, links: list[dict], *, reason: str, data: dict | None = None) -> int:
        """Store the causal trail found for an action claim (possibly empty) as one event."""

        def _run() -> int:
            with self._tx() as conn:
                eid = self._log_event(conn, "trail", "trail", claim_id=claim_id, reason=reason,
                                      data={"links": links, **(data or {})})
                for link in links:
                    conn.execute(
                        "INSERT INTO links(src_kind, src_id, dst_kind, dst_id, relation, confidence, reason, evidence, "
                        "status, event_id, created_at) VALUES ('observation', ?, 'claim', ?, ?, ?, ?, ?, 'live', ?, ?) "
                        "ON CONFLICT(src_kind, src_id, dst_kind, dst_id, relation) DO UPDATE SET "
                        "confidence=excluded.confidence, reason=excluded.reason, evidence=excluded.evidence, "
                        "status='live', event_id=excluded.event_id",
                        (link["observation_id"], claim_id, link["relation"], float(link["confidence"]),
                         link["explanation"], json.dumps([{"observation_id": link["observation_id"],
                                                           "quote": link["quote"]}]), eid, now()))
                return eid

        return await self._call(_run)

    async def actions_without_trail(self, since: float, *, limit: int = 20) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM claims c WHERE c.status='live' AND c.kind='event' AND c.action != '' AND c.created_at >= ? "
            "AND NOT EXISTS (SELECT 1 FROM memory_events e WHERE e.op='trail' AND e.claim_id=c.id) "
            "ORDER BY c.occurred_from LIMIT ?", (since, limit))

    async def links_into(self, dst_kind: str, ids: Sequence[int]) -> list[sqlite3.Row]:
        """Live links pointing at these memories, strongest first."""
        if not ids:
            return []
        return await self.q(
            f"SELECT * FROM links WHERE status='live' AND dst_kind=? AND dst_id IN ({_ph(ids)}) "
            f"ORDER BY relation='led_to' DESC, confidence DESC", (dst_kind, *ids))
