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
            f"AND src_kind='observation' AND relation IN ('led_to', 'step') "
            f"ORDER BY relation='led_to' DESC, confidence DESC", (dst_kind, *ids))

    @staticmethod
    def _pair_revision(row) -> dict:
        return {k: row[k] for k in ("id", "updated_at", "text", "context", "status", "flag", "modality",
                                    "occurred_from", "occurred_to", "confidence", "derived_from")}

    async def checked_pairs(self) -> set[tuple[int, int]]:
        """Only validated verdicts for the current revisions suppress another sweep."""
        out = set()
        for link in await self.q("SELECT src_id, dst_id, evidence FROM links WHERE relation='checked' "
                                 "AND src_kind='claim' AND status='live'"):
            rows = await self.get_claims([link["src_id"], link["dst_id"]])
            current = [self._pair_revision(r) for r in sorted(rows, key=lambda r: r["id"])]
            try:
                saved = json.loads(link["evidence"] or "{}")
            except (ValueError, TypeError):
                continue
            if isinstance(saved, dict) and saved.get("revisions") == current and len(current) == 2:
                out.add((min(link["src_id"], link["dst_id"]), max(link["src_id"], link["dst_id"])))
        return out

    async def mark_pair_checked(self, a: int, b: int, verdict: str, reason: str, *,
                                revisions: list[dict] | None = None) -> bool:
        if verdict not in {"compatible", "conflict"}:
            return False
        lo, hi = min(a, b), max(a, b)

        def _run():
            with self._tx() as conn:
                rows = conn.execute("SELECT * FROM claims WHERE id IN (?,?) ORDER BY id", (lo, hi)).fetchall()
                current = [self._pair_revision(r) for r in rows]
                if len(current) != 2 or (revisions is not None and current != revisions):
                    return False
                conn.execute(
                    "INSERT INTO links(src_kind, src_id, dst_kind, dst_id, relation, reason, evidence, status, created_at) "
                    "VALUES ('claim', ?, 'claim', ?, 'checked', ?, ?, 'live', ?) "
                    "ON CONFLICT(src_kind,src_id,dst_kind,dst_id,relation) DO UPDATE SET "
                    "reason=excluded.reason,evidence=excluded.evidence,status='live',created_at=excluded.created_at",
                    (lo, hi, f"{verdict}: {reason}", json.dumps({"revisions": current}), now()))
                return True

        return await self._call(_run)
