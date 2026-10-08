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
        withdrawn = {c["id"] for c in changes if c.get("after", {}).get("status") in DEAD
                     or c.get("after", {}).get("flag") == "contested"}
        if withdrawn:
            changes += self._withdraw_dependents(conn, withdrawn)
            if op != "merge":  # a merge only rewords; everything else may leave a page quoting a withdrawn claim
                self._dirty_entity_pages(conn, {c["id"] for c in changes
                                                if c.get("after", {}).get("status") in DEAD
                                                or c.get("after", {}).get("flag") == "contested"})
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

    def _dirty_entity_pages(self, conn, claim_ids: set[int]) -> None:
        """Generated entity pages are not authority: clear any page that may quote a withdrawn claim
        (and its search rows), so it is rebuilt from the claims that remain."""
        for (eid,) in conn.execute(
                f"SELECT DISTINCT m.entity_id FROM mentions m JOIN entities e ON e.id=m.entity_id "
                f"WHERE m.ref_kind='claim' AND m.ref_id IN ({_ph(sorted(claim_ids))}) AND e.summary != ''",
                sorted(claim_ids)).fetchall():
            conn.execute("UPDATE entities SET summary='', page_updated_at=0, updated_at=? WHERE id=?", (now(), eid))
            self._refresh_entity_fts(conn, eid)
            self._enqueue(conn, "entity", eid)

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

    # -------------------------------------------------------------- evidence/fields

    def _evidence_rows(self, conn, claim_id: int) -> dict[int, dict]:
        return {r["observation_id"]: dict(r) for r in conn.execute(
            "SELECT observation_id, relation, quote, field, created_at FROM evidence WHERE claim_id=?", (claim_id,))}

    def _evidence_change(self, conn, claim_id: int, before: dict[int, dict], *, fields_before: dict | None = None,
                         fields_after: dict | None = None) -> dict:
        """The evidence rows a change added and evicted (in full, so a revert can put them
        back), plus untracked columns it changed (occurred_to, derived_from, flag)."""
        after = self._evidence_rows(conn, claim_id)
        # A row updated in place is both removed (its old form) and added (its new one).
        changed = {o for o in set(after) & set(before) if after[o] != before[o]}
        change: dict = {"id": claim_id,
                        "evidence_added_rows": [after[o] for o in sorted((set(after) - set(before)) | changed)],
                        "evidence_removed_rows": [before[o] for o in sorted((set(before) - set(after)) | changed)]}
        if fields_before is not None:
            change["fields_before"], change["fields_after"] = fields_before, fields_after or {}
        return change

    def _invert(self, conn, change: dict) -> dict | None:
        """Undo one recorded change; returns the change that undoes the undo (for redo)."""
        cid = change["id"]
        inv: dict = {"id": cid}
        added = change.get("evidence_added_rows") or [{"observation_id": o} for o in change.get("evidence_added", [])]
        if added:
            have = self._evidence_rows(conn, cid)
            inv["evidence_removed_rows"] = [have[r["observation_id"]] for r in added if r["observation_id"] in have]
            for r in added:
                conn.execute("DELETE FROM evidence WHERE claim_id=? AND observation_id=?", (cid, r["observation_id"]))
        if removed := change.get("evidence_removed_rows"):
            for r in removed:
                conn.execute("INSERT OR REPLACE INTO evidence(claim_id, observation_id, relation, quote, field, "
                             "created_at) VALUES (?,?,?,?,?,?)", (cid, r["observation_id"], r["relation"], r["quote"],
                                                                  r.get("field", ""), r["created_at"]))
            inv["evidence_added_rows"] = removed
        if fields := change.get("fields_before"):
            row = conn.execute(f"SELECT {', '.join(fields)} FROM claims WHERE id=?", (cid,)).fetchone()
            inv["fields_before"], inv["fields_after"] = dict(row) if row else {}, fields
            conn.execute(f"UPDATE claims SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                         (*fields.values(), cid))
        if added or removed:
            self._recompute_after_revert(conn, cid)
        if change.get("before"):
            if (c := self._set_state(conn, cid, **change["before"])) is not None:
                inv["before"], inv["after"] = c["before"], c["after"]
        return inv if len(inv) > 1 else None

    # ------------------------------------------------------------------- revert

    def _set_profile(self, conn, profile: dict | None) -> None:
        if profile is None:
            conn.execute("DELETE FROM meta WHERE key='profile'")
        else:
            conn.execute("INSERT INTO meta(key, value) VALUES ('profile', ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(profile),))

    def _current_profile(self, conn) -> dict | None:
        row = conn.execute("SELECT value FROM meta WHERE key='profile'").fetchone()
        return json.loads(row[0]) if row else None

    async def revert_event(self, event_id: int, *, force: bool = False, reason: str = "") -> int:
        """Undo one event; returns the id of the event that records the undo.

        Reverting a `revert` re-applies the original (op `redo`) and makes the original
        active again, so it can later be reverted the ordinary way. A redo itself is not
        reverted; revert the original event again instead."""

        def _run() -> int:
            with self._tx() as conn:
                ev = conn.execute("SELECT * FROM memory_events WHERE id=?", (event_id,)).fetchone()
                if ev is None:
                    raise KeyError(f"no event {event_id}")
                if ev["undone_by"] is not None:
                    raise RevertConflict(f"event {event_id} was already reverted by event {ev['undone_by']}")
                data = json.loads(ev["data"] or "{}")
                if data.get("erased"):
                    raise RevertConflict(f"event {event_id} was erased with the memory it changed")
                if ev["op"] == "redo":
                    raise RevertConflict(f"#{event_id} re-applied #{data.get('reapplies')}; revert that event instead")
                changes = json.loads(ev["changes"])
                created = [] if ev["op"] == "revert" else json.loads(ev["created"])
                touched = list({*created, *(c["id"] for c in changes)})
                if not force:
                    later = conn.execute(
                        f"SELECT DISTINCT e.id, e.op FROM memory_events e JOIN memory_event_refs r ON r.event_id=e.id "
                        f"WHERE e.id > ? AND e.undone_by IS NULL AND e.op != 'revert' "
                        f"AND NOT (e.op = 'redo' AND json_extract(e.data, '$.reapplies') = ?) "
                        f"AND r.claim_id IN ({_ph(touched)})", (event_id, event_id, *touched)).fetchall() \
                        if touched else []
                    if ev["op"] == "profile" or "profile_before" in data:
                        later += conn.execute("SELECT id, op FROM memory_events WHERE id > ? AND op='profile' "
                                              "AND undone_by IS NULL", (event_id,)).fetchall()
                    if later:
                        raise RevertConflict("later changes touch the same memories: "
                                             + ", ".join(f"#{r['id']} {r['op']}" for r in later))
                out: list[dict | None] = [self._invert(conn, c) for c in changes]
                out += [self._set_state(conn, cid, status="reverted", valid_to=now()) for cid in created]
                if ev["op"] == "revert":
                    # Redo: put back the side effects the revert took away.
                    original = data["reverts"]
                    for r in data.get("suppressions", []):
                        conn.execute("INSERT OR REPLACE INTO suppressions(id, kind, text, norm_key, vector, claim_id, "
                                     "event_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
                                     (r["id"], r["kind"], r["text"], r["norm_key"],
                                      bytes.fromhex(r["vector"]) if r["vector"] else None, r["claim_id"],
                                      r["event_id"], r["created_at"]))
                    if data.get("links"):
                        conn.execute(f"UPDATE links SET status='live' WHERE id IN ({_ph(data['links'])})",
                                     data["links"])
                    for r in data.get("links_overwritten", []):
                        self._restore_link(conn, r["id"], r["after"])
                    if "profile_was" in data:
                        if not force and self._current_profile(conn) != data["profile_was"]:
                            raise RevertConflict("the profile was rewritten after the undo; redo would discard it")
                        self._set_profile(conn, data["profile_set"])
                    rid = self._log_event(conn, "redo", "user", claim_id=ev["claim_id"], changes=out,
                                          reason=reason or f"re-applied #{original} (reverted by #{event_id})",
                                          data={"reapplies": original, "undoes": event_id})
                    conn.execute("UPDATE memory_events SET undone_by=NULL WHERE id=?", (original,))
                    conn.execute("UPDATE memory_events SET undone_by=? WHERE id=?", (rid, event_id))
                    return rid
                side: dict = {"reverts": event_id}
                # A forget's suppression, an event's links and a profile rewrite go with it.
                sup = conn.execute("SELECT * FROM suppressions WHERE event_id=?", (event_id,)).fetchall()
                side["suppressions"] = [dict(r) | {"vector": r["vector"].hex() if r["vector"] else ""} for r in sup]
                conn.execute("DELETE FROM suppressions WHERE event_id=?", (event_id,))
                links = [r[0] for r in conn.execute("SELECT id FROM links WHERE event_id=? AND status='live'",
                                                    (event_id,))]
                # A link this event overwrote goes back to its predecessor rather than being retired.
                before_rows = {r["id"]: r for r in data.get("links_before", [])}
                side["links_overwritten"] = []
                for lid in [x for x in links if x in before_rows]:
                    current = conn.execute("SELECT * FROM links WHERE id=?", (lid,)).fetchone()
                    side["links_overwritten"].append({"id": lid, "after": dict(current)})
                    self._restore_link(conn, lid, before_rows[lid])
                links = [x for x in links if x not in before_rows]
                side["links"] = links
                if links:
                    conn.execute(f"UPDATE links SET status='reverted' WHERE id IN ({_ph(links)})", links)
                if ev["op"] == "profile" or "profile_before" in data:
                    previous = data.get("previous") if ev["op"] == "profile" else data["profile_before"]
                    side["profile_was"], side["profile_set"] = previous, self._current_profile(conn)
                    self._set_profile(conn, previous)
                rid = self._log_event(conn, "revert", "user", claim_id=ev["claim_id"], changes=out,
                                      reason=reason or f"revert of #{event_id} ({ev['op']})", data=side)
                conn.execute("UPDATE memory_events SET undone_by=? WHERE id=?", (rid, event_id))
                return rid

        return await self._call(_run)

    @staticmethod
    def _restore_link(conn, link_id: int, row: dict) -> None:
        cols = ("confidence", "reason", "evidence", "status", "event_id", "created_at")
        conn.execute(f"UPDATE links SET {', '.join(f'{c}=?' for c in cols)} WHERE id=?",
                     (*(row[c] for c in cols), link_id))

    def _recompute_after_revert(self, conn, claim_id: int) -> None:
        """Hook for ClaimsMixin: confidence follows evidence."""
