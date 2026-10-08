"""The user's corrections: forget a memory, or correct its wording (mem0's and
Honcho's user feedback; Claude Code's dream gathers "user corrections" first).

A forgotten claim is retired *and* suppressed: a later episode that drafts the same
claim again (the screen still shows it) drops it instead of re-learning it. A
correction is a new version of the claim in its lineage with modality `stated`,
which outranks screen evidence everywhere confidence is used. Both are events in
the change log, so both can be reverted.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from uuid import uuid4

import numpy as np

from .. import textkit as tk
from ..config import T
from ..models import ClaimDraft
from ..timeutil import now
from .claims import ClaimsMixin, _ph


class CorrectionsMixin(ClaimsMixin):
    def _feedback_claims(self, conn, claim_ids: Sequence[int], *, live_only: bool = False) -> list[int]:
        ids = sorted(set(claim_ids))
        for cid in ids:
            row = conn.execute("SELECT status FROM claims WHERE id=?", (cid,)).fetchone()
            if row is None or (live_only and row[0] != "live"):
                raise KeyError(f"no {'live ' if live_only else ''}claim {cid}")
        return ids

    async def state_claim(self, kind: str, text: str, *, context: str = "",
                          derived_from: Sequence[int] = (), reason: str = "") -> tuple[int, int]:
        """A new lineage in the user's words, with copied supporting provenance."""
        def _run() -> tuple[int, int]:
            with self._tx() as conn:
                ids = self._feedback_claims(conn, derived_from)
                cid = self._insert_claim(conn, ClaimDraft(kind=kind, text=text.strip(), context=context,
                    modality="stated", reasoning="stated by the user"), lineage_id=str(uuid4()), version=1,
                    derived_from=ids, allow_suppressed=True)
                if ids:
                    for ev in conn.execute(f"SELECT * FROM evidence WHERE relation='supports' "
                                           f"AND claim_id IN ({_ph(ids)}) ORDER BY created_at", ids).fetchall():
                        self._link(conn, cid, ev["observation_id"], "supports", ev["quote"], ev["field"])
                    self._cap_evidence(conn, cid)
                    self._recompute_confidence(conn, cid)
                    self._carry_mentions(conn, ids, cid)
                eid = self._log_event(conn, "state", "user", claim_id=cid, created=[cid],
                                      reason=reason or "the user stated this", data={"derived_from": ids})
                return cid, eid

        return await self._call(_run)

    async def confirm_claims(self, claim_ids: Sequence[int], *, context: str = "", reason: str = "") -> int:
        """Record applicability; observation-based belief strength stays unchanged."""
        def _run() -> int:
            with self._tx() as conn:
                ids = self._feedback_claims(conn, claim_ids, live_only=True)
                eid = self._log_event(conn, "confirm", "user", reason=reason or "the user confirmed these memories",
                                      data={"claim_ids": ids, "context": context})
                conn.executemany("INSERT INTO memory_event_refs(event_id, claim_id) VALUES (?,?)",
                                 [(eid, cid) for cid in ids])
                return eid

        return await self._call(_run)

    async def retire_claim(self, claim_id: int, *, reason: str = "") -> int:
        """Retire every live version of a lineage without suppressing future learning."""
        def _run() -> int:
            with self._tx() as conn:
                self._feedback_claims(conn, [claim_id])
                live = conn.execute("SELECT id FROM claims WHERE lineage_id="
                                    "(SELECT lineage_id FROM claims WHERE id=?) AND status='live'",
                                    (claim_id,)).fetchall()
                if not live:
                    raise KeyError(f"claim {claim_id} has no live version to retire")
                ts = now()
                changes = [self._set_state(conn, r[0], status="retired", valid_to=ts) for r in live]
                return self._log_event(conn, "retire", "user", claim_id=claim_id, changes=changes,
                                       reason=reason or "the user said this is no longer true")

        return await self._call(_run)

    async def reject_inference(self, text: str, *, kind: str = "preference",
                               claim_ids: Sequence[int] = (), reason: str = "") -> int:
        """Suppress an inference's wording, leaving its source memories untouched."""
        def _run() -> int:
            with self._tx() as conn:
                ids = self._feedback_claims(conn, claim_ids)
                wording = text.strip()
                # Only reuse an embedding of this exact wording, never of a different premise.
                vec = conn.execute("SELECT e.vector FROM embeddings e JOIN claims c ON c.id=e.ref_id "
                                   "WHERE e.ref_kind='claim' AND c.kind=? AND c.text=? LIMIT 1",
                                   (kind, wording)).fetchone()
                eid = self._log_event(conn, "reject", "user", reason=reason or "the user rejected this inference",
                                      data={"claim_ids": ids, "kind": kind, "text": wording})
                conn.executemany("INSERT INTO memory_event_refs(event_id, claim_id) VALUES (?,?)",
                                 [(eid, cid) for cid in ids])
                conn.execute("INSERT INTO suppressions(kind, text, norm_key, vector, claim_id, event_id, created_at) "
                             "VALUES (?,?,?,?,NULL,?,?)", (kind, wording, tk.norm_key(wording),
                                                          vec[0] if vec else None, eid, now()))
                return eid

        return await self._call(_run)

    async def forget_claim(self, claim_id: int, *, reason: str = "") -> int:
        """Retire a claim (and every live version of its lineage) and suppress its text."""

        def _run() -> int:
            with self._tx() as conn:
                row = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
                if row is None:
                    raise KeyError(f"no claim {claim_id}")
                live = conn.execute("SELECT id FROM claims WHERE lineage_id=? AND status='live'",
                                    (row["lineage_id"],)).fetchall()
                vec = conn.execute("SELECT vector FROM embeddings WHERE ref_kind='claim' AND ref_id=?",
                                   (claim_id,)).fetchone()
                ts = now()
                changes = [self._set_state(conn, r[0], status="forgotten", valid_to=ts) for r in live]
                eid = self._log_event(conn, "forget", "user", claim_id=claim_id, changes=changes,
                                      reason=reason or "the user asked to forget this")
                conn.execute("INSERT INTO suppressions(kind, text, norm_key, vector, claim_id, event_id, created_at) "
                             "VALUES (?,?,?,?,?,?,?)",
                             (row["kind"], row["text"], tk.norm_key(row["text"]), vec[0] if vec else None,
                              claim_id, eid, ts))
                return eid

        return await self._call(_run)

    async def erase_claim(self, claim_id: int) -> int:
        """Permanently remove a claim lineage's *derived* content (unlike `forget_claim`, not undoable).

        Blanks every version's text, context and reasoning (and with them the search index), deletes
        its evidence quotes, links, mentions and embeddings, scrubs the change-log payloads and profile
        snapshots that carried it, and keeps only a hash of the wording as a suppression so
        it is not re-learned. Source observations, screenshots, episode summaries and any earlier
        export commits are separate stores and are not touched. Returns the number of versions erased."""

        def _run() -> int:
            with self._tx() as conn:
                row = conn.execute("SELECT lineage_id, kind FROM claims WHERE id=?", (claim_id,)).fetchone()
                if row is None:
                    raise KeyError(f"no claim {claim_id}")
                rows = conn.execute("SELECT id, kind, text, status FROM claims WHERE lineage_id=?",
                                    (row["lineage_id"],)).fetchall()
                ids = [r["id"] for r in rows]
                ts = now()
                # Withdraw live versions first so dependent patterns, pages and profile points follow.
                changes = [self._set_state(conn, r["id"], status="forgotten", valid_to=ts)
                           for r in rows if r["status"] == "live"]
                self._log_event(conn, "erase", "user", changes=changes, reason="erased")
                hashes = {r["id"]: "h:" + hashlib.sha256(tk.norm_key(r["text"]).encode()).hexdigest() for r in rows}
                # No vector either: embeddings can be partly inverted to text. The cost is that only
                # the exact wording stays suppressed; a paraphrase from the kept observations can return.
                for r in rows:
                    conn.execute("DELETE FROM suppressions WHERE claim_id=?", (r["id"],))
                    conn.execute("INSERT INTO suppressions(kind, text, norm_key, vector, claim_id, event_id, created_at)"
                                 " VALUES (?,?,?,NULL,?,NULL,?)",
                                 (r["kind"], "", hashes[r["id"]], r["id"], ts))
                # Events (and their redo payloads) that carried the content lose it; they can no longer be reverted.
                events = [e[0] for e in conn.execute(
                    f"SELECT DISTINCT event_id FROM memory_event_refs WHERE claim_id IN ({_ph(ids)})", ids)]
                if events:
                    conn.execute(f"UPDATE memory_events SET changes='[]', created='[]', reason='erased', "
                                 f"data='{{\"erased\": true}}' WHERE id IN ({_ph(events)})", events)
                conn.execute(f"DELETE FROM memory_event_refs WHERE claim_id IN ({_ph(ids)})", ids)
                for e in conn.execute("SELECT id, data FROM memory_events WHERE data LIKE '%\"sections\"%'").fetchall():
                    data = json.loads(e["data"])
                    for key in ("previous", "profile_before", "profile_after", "profile_was", "profile_set"):
                        if isinstance(data.get(key), dict):
                            self._drop_profile_points(data[key], set(ids))
                    conn.execute("UPDATE memory_events SET data=? WHERE id=?", (json.dumps(data), e["id"]))
                if (profile := self._current_profile(conn)) is not None:
                    self._drop_profile_points(profile, set(ids))
                    self._set_profile(conn, profile)
                for cid in ids:
                    self._drop_vector(conn, "claim", cid)
                conn.execute(f"DELETE FROM evidence WHERE claim_id IN ({_ph(ids)})", ids)
                conn.execute(f"DELETE FROM mentions WHERE ref_kind='claim' AND ref_id IN ({_ph(ids)})", ids)
                for col in ("src", "dst"):
                    conn.execute(f"DELETE FROM links WHERE {col}_kind='claim' AND {col}_id IN ({_ph(ids)})", ids)
                conn.execute(f"UPDATE claims SET text='[erased]', context='', reasoning='', action='', "
                             f"derived_from='[]' WHERE id IN ({_ph(ids)})", ids)
                return len(ids)

        return await self._call(_run)

    @staticmethod
    def _drop_profile_points(profile: dict, ids: set[int]) -> None:
        for sec in profile.get("sections", []):
            sec["points"] = [p for p in sec["points"] if not ids & set(p.get("claims", []))]
        profile["sections"] = [sec for sec in profile.get("sections", []) if sec["points"]]

    async def correct_claim(self, claim_id: int, text: str, *, reason: str = "",
                            context: str | None = None) -> tuple[int, int]:
        """A new `stated` version with the user's wording and optional replacement context.
        The old one is superseded.
        Returns (new claim id, event id)."""

        def _run() -> tuple[int, int]:
            with self._tx() as conn:
                asked = conn.execute("SELECT lineage_id FROM claims WHERE id=?", (claim_id,)).fetchone()
                if asked is None:
                    raise KeyError(f"no claim {claim_id}")
                # Correct the lineage's current version, whichever version was named.
                row = conn.execute("SELECT * FROM claims WHERE lineage_id=? AND status='live' "
                                   "ORDER BY version DESC LIMIT 1", (asked["lineage_id"],)).fetchone()
                if row is None:
                    raise KeyError(f"claim {claim_id} has no live version to correct")
                target = row["id"]
                version = conn.execute("SELECT MAX(version) FROM claims WHERE lineage_id=?",
                                       (row["lineage_id"],)).fetchone()[0] or 1
                ev = conn.execute("SELECT observation_id, quote FROM evidence WHERE claim_id=? AND relation='supports'",
                                  (target,)).fetchall()
                draft = ClaimDraft(kind=row["kind"], text=text.strip(),
                                   context=row["context"] if context is None else context,
                                   reasoning="corrected by the user", modality="stated",
                                   episode_id=row["episode_id"], occurred_from=row["occurred_from"],
                                   occurred_to=row["occurred_to"], action=row["action"],
                                   evidence=[{"observation_id": r[0], "quote": r[1]} for r in ev])
                nid = self._insert_claim(conn, draft, lineage_id=row["lineage_id"], version=version + 1,
                                         derived_from=[target], allow_suppressed=True)
                self._carry_mentions(conn, [target], nid)
                change = self._set_state(conn, target, status="superseded", valid_to=now(), superseded_by=nid)
                eid = self._log_event(conn, "correct", "user", claim_id=nid, created=[nid], changes=[change],
                                      reason=reason or "the user corrected the wording")
                # What led to the action still led to it: copy its trail to the new version
                # (tagged with this event, so reverting the correction retires the copies).
                conn.execute(
                    "INSERT OR IGNORE INTO links(src_kind, src_id, dst_kind, dst_id, relation, confidence, reason, "
                    "evidence, status, event_id, created_at) SELECT src_kind, src_id, dst_kind, ?, relation, "
                    "confidence, reason, evidence, 'live', ?, ? FROM links WHERE dst_kind='claim' AND dst_id=? "
                    "AND status='live' AND relation IN ('led_to', 'step')", (nid, eid, now(), target))
                return nid, eid

        return await self._call(_run)

    async def suppressed(self, kind: str, text: str, vector=None) -> bool:
        """Would this new claim re-learn something the user told us to forget?"""
        return await self._call(lambda: self._suppressed(self.conn, kind, text, vector))

    def _suppressed(self, conn, kind: str, text: str, vector=None) -> bool:
        """Transaction-time gate shared by every claim writer."""
        key = tk.norm_key(text)
        digest = "h:" + hashlib.sha256(key.encode()).hexdigest()  # erased suppressions keep only this
        rows: list[sqlite3.Row] = conn.execute(
            "SELECT norm_key, text, vector FROM suppressions WHERE kind=?", (kind,)).fetchall()
        if not rows:
            return False
        toks = set(tk.content_tokens(text))
        v = np.asarray(vector, dtype=np.float32) if vector is not None else None
        for r in rows:
            if r["norm_key"] in (key, digest):
                return True
            other = set(tk.content_tokens(r["text"]))
            if toks and other and len(toks & other) / len(toks | other) >= T.SUPPRESS_JACCARD:
                return True
            if v is not None and r["vector"] is not None:
                sv = np.frombuffer(r["vector"], dtype=np.float32)
                if sv.shape == v.shape and float(sv @ v) / (float(np.linalg.norm(sv) * np.linalg.norm(v)) or 1.0) \
                        >= T.SUPPRESS_SIM:
                    return True
        return False
