"""The memory change log: every change to a claim, who made it, why, and how to undo it.

Claims are never deleted: a replaced claim keeps its row with a status
(superseded, invalidated, retired, forgotten, reverted). That already kept the
*data* of every version. This log adds what it lacked:

* **what changed and why**: one append-only `memory_events` row per decision
  (an episode draft adding or invalidating, a merge, a dream proposal, a cleanup,
  a user's forget/correct), with its actor, a reason and, for every claim it touched,
  the state before and after;
* **undo**: `revert_event` restores the before-state of every claim an event
  changed and retires the claims it created. A revert is itself an event, so it
  can be reverted (redo). A revert refuses when a later event changed the same
  claims, unless forced.

The state an event records is `TRACKED`: the claim fields any decision changes.
Text never changes in place (a new text is a new version), so it needs no tracking.

Why a log and not git (Cognition's memory repo): memory here is rows with vectors
and full-text indexes written continuously by capture, and per-claim history with
reasons is what review and undo need. `discrete export` mirrors the live memory
into a Markdown folder under git for reading and diffs (export.py).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from ..timeutil import now
from .base import StoreBase, rowid

TRACKED = ("status", "valid_to", "superseded_by", "flag")
#: States a claim can be retired into; none of them is searched.
DEAD = ("superseded", "invalidated", "retired", "forgotten", "reverted")


class RevertConflict(RuntimeError):
    """A later change touched the same claims; revert those first, or force."""


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


class HistoryMixin(StoreBase):
    # ------------------------------------------------------------ writing (in a tx)

    def _set_state(self, conn, claim_id: int, *, only_if_live: bool = False, **after: Any) -> dict | None:
        """Change tracked fields of one claim; returns the change for the event, or
        None when nothing changed (missing claim, or not live under only_if_live)."""
        assert after and set(after) <= set(TRACKED), after
        row = conn.execute(f"SELECT {', '.join(TRACKED)} FROM claims WHERE id=?", (claim_id,)).fetchone()
        if row is None or (only_if_live and row["status"] != "live"):
            return None
        before = {f: row[f] for f in after}
        if before == after:
            return None
        sets = ", ".join(f"{f}=?" for f in after)
        conn.execute(f"UPDATE claims SET {sets}, updated_at=? WHERE id=?", (*after.values(), now(), claim_id))
        self._index_for_status(conn, claim_id, before.get("status", row["status"]), after.get("status", row["status"]))
        return {"id": claim_id, "before": before, "after": after}

    def _index_for_status(self, conn, claim_id: int, old: str, new: str) -> None:
        """Only live claims are searchable: drop the vector on retire, re-embed on revive."""
        if old == "live" and new != "live":
            self._drop_vector(conn, "claim", claim_id)
        elif old != "live" and new == "live":
            self._enqueue(conn, "claim", claim_id)

    def _log_event(self, conn, op: str, actor: str, *, claim_id: int | None = None,
                   created: Sequence[int] = (), changes: Sequence[dict | None] = (), reason: str = "",
                   run_id: str = "", data: dict | None = None) -> int:
        changes = [c for c in changes if c]
        cur = conn.execute(
            "INSERT INTO memory_events(at, op, actor, claim_id, created, changes, reason, run_id, data) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (now(), op, actor, claim_id, json.dumps(list(created)), json.dumps(changes), reason[:500], run_id,
             json.dumps(data or {})),
        )
        eid = rowid(cur)
        refs = {*(created or ()), *(c["id"] for c in changes), *([claim_id] if claim_id is not None else [])}
        conn.executemany("INSERT OR IGNORE INTO memory_event_refs(event_id, claim_id) VALUES (?, ?)",
                         [(eid, cid) for cid in refs])
        return eid

    # ------------------------------------------------------------------ reading

    async def events_for_claim(self, claim_id: int) -> list[sqlite3.Row]:
        """Every event touching any version of the claim's lineage, oldest first."""
        return await self.q(
            "SELECT DISTINCT e.* FROM memory_events e JOIN memory_event_refs r ON r.event_id=e.id "
            "WHERE r.claim_id IN (SELECT id FROM claims WHERE lineage_id=(SELECT lineage_id FROM claims WHERE id=?)) "
            "OR r.claim_id=? ORDER BY e.id", (claim_id, claim_id))

    async def recent_events(self, *, since: float = 0.0, actor: str = "", op: str = "",
                            limit: int = 50) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM memory_events WHERE at >= ?", [since]
        if actor:
            sql += " AND actor=?"
            args.append(actor)
        if op:
            sql += " AND op=?"
            args.append(op)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return await self.q(sql, args)

    async def get_event(self, event_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM memory_events WHERE id=?", (event_id,))

    async def lineage(self, claim_id: int) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM claims WHERE lineage_id=(SELECT lineage_id FROM claims WHERE id=?) "
                            "ORDER BY version, id", (claim_id,))

    # ------------------------------------------------------------------- revert

    async def revert_event(self, event_id: int, *, force: bool = False, reason: str = "") -> int:
        """Undo one event; returns the id of the revert event."""

        def _run() -> int:
            with self._tx() as conn:
                ev = conn.execute("SELECT * FROM memory_events WHERE id=?", (event_id,)).fetchone()
                if ev is None:
                    raise KeyError(f"no event {event_id}")
                if ev["undone_by"] is not None:
                    raise RevertConflict(f"event {event_id} was already reverted by event {ev['undone_by']}")
                created = json.loads(ev["created"])
                changes = json.loads(ev["changes"])
                touched = list({*created, *(c["id"] for c in changes)})
                if touched and not force:
                    later = conn.execute(
                        f"SELECT DISTINCT e.id, e.op FROM memory_events e JOIN memory_event_refs r ON r.event_id=e.id "
                        f"WHERE e.id > ? AND e.undone_by IS NULL AND e.op != 'revert' "
                        f"AND r.claim_id IN ({_ph(touched)})", (event_id, *touched)).fetchall()
                    if later:
                        raise RevertConflict("later changes touch the same claims: "
                                             + ", ".join(f"#{r['id']} {r['op']}" for r in later))
                out: list[dict | None] = []
                for change in changes:
                    cid = change["id"]
                    for obs_id in change.get("evidence_added", []):
                        conn.execute("DELETE FROM evidence WHERE claim_id=? AND observation_id=?", (cid, obs_id))
                    if change.get("evidence_added"):
                        self._recompute_after_revert(conn, cid)
                    if change.get("before"):
                        out.append(self._set_state(conn, cid, **change["before"]))
                for cid in created:
                    out.append(self._set_state(conn, cid, status="reverted", valid_to=now()))
                # A forget's suppression goes with it.
                conn.execute("DELETE FROM suppressions WHERE event_id=?", (event_id,))
                rid = self._log_event(conn, "revert", "user", claim_id=ev["claim_id"], changes=out,
                                      reason=reason or f"revert of #{event_id} ({ev['op']})",
                                      data={"reverts": event_id, "evidence_removed": [
                                          {"id": c["id"], "observations": c["evidence_added"]}
                                          for c in changes if c.get("evidence_added")]})
                conn.execute("UPDATE memory_events SET undone_by=? WHERE id=?", (rid, event_id))
                return rid

        return await self._call(_run)

    def _recompute_after_revert(self, conn, claim_id: int) -> None:
        """Hook for ClaimsMixin: confidence follows evidence."""
