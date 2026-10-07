"""Beliefs: patterns dreamed from many episodes, how strongly each memory is held,
conflict resolution, staleness and the profile. Every change is an event (history.py).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from ..config import T
from ..models import ClaimDraft
from ..timeutil import day_key, now
from .links import LinksMixin


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


@dataclass
class Strength:
    """How firmly a memory is held: on how many separate episodes and days its evidence
    was seen, and when first and last. `tier` is what Bubble and the export show."""

    episodes: int
    days: int
    first: float
    last: float

    @property
    def tier(self) -> str:
        if self.days >= T.ESTABLISHED_DAYS and self.episodes >= T.ESTABLISHED_EPISODES:
            return "established"
        return "tentative" if self.episodes <= 1 else "recurring"

    def describe(self) -> str:
        span = f"seen in {self.episodes} episode{'s' * (self.episodes != 1)}"
        if self.days > 1:
            span += f" over {self.days} days"
        return span


class BeliefsMixin(LinksMixin):
    # ------------------------------------------------------------------ strength

    async def strengths(self, claim_ids: Sequence[int]) -> dict[int, Strength]:
        if not claim_ids:
            return {}
        rows = await self.q(
            f"SELECT e.claim_id, o.episode_id, o.started_at FROM evidence e JOIN observations o "
            f"ON o.id=e.observation_id WHERE e.relation='supports' AND e.claim_id IN ({_ph(claim_ids)})",
            list(claim_ids))
        acc: dict[int, tuple[set, set, list]] = {}
        for cid, ep, at in rows:
            eps, days, times = acc.setdefault(cid, (set(), set(), []))
            eps.add(ep if ep is not None else f"o{at}")
            days.add(day_key(at))
            times.append(at)
        return {cid: Strength(len(e), len(d), min(t), max(t)) for cid, (e, d, t) in acc.items()}

    # ------------------------------------------------------------------ gathering

    async def claims_for_dream(self, since: float, *, limit: int) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM claims WHERE status='live' AND occurred_to >= ? "
            "ORDER BY occurred_from DESC LIMIT ?", (since, limit))

    async def new_episodes_since(self, ts: float) -> int:
        return await self.scalar("SELECT COUNT(*) FROM episodes WHERE status='closed' AND updated_at > ?", (ts,))

    async def episodes_of_claims(self, claim_ids: Sequence[int]) -> dict[int, set[int]]:
        """Distinct episodes of each claim: its own and those of its supporting evidence."""
        if not claim_ids:
            return {}
        out: dict[int, set[int]] = {}
        for cid, ep in await self.q(
                f"SELECT e.claim_id, o.episode_id FROM evidence e JOIN observations o ON o.id=e.observation_id "
                f"WHERE e.relation='supports' AND e.claim_id IN ({_ph(claim_ids)})", list(claim_ids)):
            if ep is not None:
                out.setdefault(cid, set()).add(ep)
        for cid, ep in await self.q(f"SELECT id, episode_id FROM claims WHERE id IN ({_ph(claim_ids)})",
                                    list(claim_ids)):
            if ep is not None:
                out.setdefault(cid, set()).add(ep)
        return out

    async def support_times(self, claim_ids: Sequence[int]) -> dict[int, list[float]]:
        """Full supporting occurrence times, independent of the quote sample."""
        if not claim_ids:
            return {}
        out: dict[int, list[float]] = {}
        for cid, at in await self.q(
                f"SELECT e.claim_id, o.started_at FROM evidence e JOIN observations o ON o.id=e.observation_id "
                f"WHERE e.relation='supports' AND e.claim_id IN ({_ph(claim_ids)})", list(claim_ids)):
            out.setdefault(cid, []).append(at)
        return out

    # ------------------------------------------------------------------ patterns

    async def add_pattern(self, text: str, category: str, support: Sequence[int], *, reason: str,
                          run_id: str) -> int:
        """A dreamed pattern: a new claim pointing down at the claims it rests on, with
        their evidence, so its confidence comes from the same episode/day count."""

        def _run() -> int:
            with self._tx() as conn:
                rows = conn.execute(f"SELECT * FROM claims WHERE id IN ({_ph(support)}) AND status='live' "
                                    "AND flag!='contested' AND kind!='pattern'",
                                    list(support)).fetchall()
                if len(rows) != len(set(support)):
                    raise ValueError("support changed while dreaming")
                ev = conn.execute(
                    f"SELECT observation_id, quote FROM evidence WHERE relation='supports' "
                    f"AND claim_id IN ({_ph(support)}) ORDER BY created_at DESC", list(support)).fetchall()
                if not self._pattern_support_valid(conn, category, text, support):
                    raise ValueError("insufficient supporting occasions")
                mods = {r["modality"] for r in rows}
                draft = ClaimDraft(
                    kind="pattern", text=text, context=category, reasoning=f"dreamed from {len(rows)} memories",
                    # A pattern is an induction, never testimony: "stated" would exempt it from
                    # staleness and make it win every conflict.
                    modality="engaged" if mods & {"engaged", "stated"} else "passive",
                    occurred_from=min(r["occurred_from"] for r in rows),
                    occurred_to=max(r["occurred_to"] for r in rows),
                    evidence=[{"observation_id": r[0], "quote": r[1]} for r in ev])
                cid = self._insert_claim(conn, draft, lineage_id=str(uuid4()), version=1,
                                         derived_from=sorted(set(support)))
                self._carry_mentions(conn, list(support), cid)
                self._log_event(conn, "dream_add", "dream", claim_id=cid, created=[cid], reason=reason,
                                run_id=run_id, data={"support": sorted(set(support)), "category": category})
                return cid

        return await self._call(_run)

    async def confirm_pattern(self, pattern_id: int, support: Sequence[int], *, reason: str, run_id: str) -> bool:
        """New memories back an existing pattern: add them to its support and evidence."""

        def _run() -> bool:
            with self._tx() as conn:
                p = conn.execute("SELECT * FROM claims WHERE id=? AND status='live'", (pattern_id,)).fetchone()
                if p is None or self._suppressed(conn, p["kind"], p["text"]):
                    return False
                eligible = conn.execute(f"SELECT id FROM claims WHERE id IN ({_ph(support)}) "
                                        "AND status='live' AND flag!='contested' AND kind!='pattern'", list(support)).fetchall()
                if len(eligible) != len(set(support)):
                    return False
                old = set(json.loads(p["derived_from"] or "[]"))
                new = sorted(set(support) - old)
                if not new:
                    return False
                had = self._evidence_rows(conn, pattern_id)
                fields_before = {"derived_from": p["derived_from"], "occurred_to": p["occurred_to"], "flag": p["flag"]}
                for oid, quote in conn.execute(
                        f"SELECT observation_id, quote FROM evidence WHERE relation='supports' "
                        f"AND claim_id IN ({_ph(new)})", new).fetchall():
                    self._link(conn, pattern_id, oid, "supports", quote)
                last = conn.execute(f"SELECT MAX(occurred_to) FROM claims WHERE id IN ({_ph(new)})", new).fetchone()[0]
                conn.execute("UPDATE claims SET derived_from=?, occurred_to=MAX(occurred_to, ?), updated_at=? "
                             "WHERE id=?", (json.dumps(sorted(old | set(new))), last or 0, now(), pattern_id))
                self._cap_evidence(conn, pattern_id)
                self._recompute_confidence(conn, pattern_id)
                if p["flag"] == "stale":
                    conn.execute("UPDATE claims SET flag='' WHERE id=?", (pattern_id,))
                now_row = conn.execute("SELECT derived_from, occurred_to, flag FROM claims WHERE id=?",
                                       (pattern_id,)).fetchone()
                change = self._evidence_change(conn, pattern_id, had, fields_before=fields_before,
                                               fields_after=dict(now_row))
                self._log_event(conn, "dream_confirm", "dream", claim_id=pattern_id, changes=[change],
                                reason=reason, run_id=run_id, data={"support_added": new})
                return True

        return await self._call(_run)

    async def retire_claim(self, claim_id: int, *, op: str, actor: str, reason: str, run_id: str = "",
                           by_id: int | None = None) -> bool:
        def _run() -> bool:
            with self._tx() as conn:
                if by_id is not None and conn.execute("SELECT status FROM claims WHERE id=?",
                                                      (by_id,)).fetchone()[0] != "live":
                    return False  # the winner was itself retired earlier in this run
                fields = {"status": "retired" if by_id is None else "invalidated", "valid_to": now()}
                change = self._set_state(conn, claim_id, only_if_live=True, **fields)
                if change is None:
                    return False
                self._log_event(conn, op, actor, claim_id=claim_id, changes=[change], reason=reason, run_id=run_id,
                                data={"by": by_id} if by_id else None)
                return True

        return await self._call(_run)

    async def flag_claims(self, claim_ids: Sequence[int], flag: str, *, op: str, actor: str, reason: str,
                          run_id: str = "") -> int:
        def _run() -> int:
            with self._tx() as conn:
                changes = [self._set_state(conn, cid, only_if_live=True, flag=flag) for cid in claim_ids]
                changes = [c for c in changes if c]
                if changes:
                    self._log_event(conn, op, actor, claim_id=changes[0]["id"], changes=changes, reason=reason,
                                    run_id=run_id)
                return len(changes)

        return await self._call(_run)

    async def stale_candidates(self, before: float) -> list[sqlite3.Row]:
        """Lasting memories not seen again since `before` (events never go stale: they are history)."""
        return await self.q("SELECT * FROM claims WHERE status='live' AND kind IN ('fact','preference','pattern') "
                            "AND flag='' AND modality!='stated' AND occurred_to < ?", (before,))

    async def unstale(self, after: float) -> int:
        return await self.flag_claims(
            [r[0] for r in await self.q("SELECT id FROM claims WHERE status='live' AND flag='stale' AND occurred_to >= ?",
                                        (after,))],
            "", op="fresh", actor="cleanup", reason="seen again")

    # ------------------------------------------------------------------ profile

    async def save_profile(self, profile: dict, *, run_id: str) -> int:
        def _run() -> int:
            with self._tx() as conn:
                clean = json.loads(json.dumps(profile))
                for sec in clean.get("sections", []):
                    points = []
                    for point in sec.get("points", []):
                        ids = point.get("claims", [])
                        live = conn.execute(f"SELECT id FROM claims WHERE id IN ({_ph(ids)}) AND status='live' "
                                            "AND flag NOT IN ('stale','contested')", ids).fetchall() if ids else []
                        if ids and len(live) == len(set(ids)) and not self._suppressed(conn, "pattern", point["text"]):
                            points.append(point)
                    sec["points"] = points
                clean["sections"] = [s for s in clean.get("sections", []) if s["points"]]
                prev = conn.execute("SELECT value FROM meta WHERE key='profile'").fetchone()
                conn.execute("INSERT INTO meta(key, value) VALUES ('profile', ?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(clean),))
                return self._log_event(conn, "profile", "dream", reason="profile rewritten", run_id=run_id,
                                       data={"previous": json.loads(prev[0]) if prev else None})

        return await self._call(_run)

    async def get_profile(self, *, live_only: bool = True) -> dict:
        """The stored profile. By default a point whose memories were forgotten, corrected
        or retired since it was written is left out, so "forget" reaches the profile at once."""
        raw = await self.get_meta("profile", "")
        profile = json.loads(raw) if raw else {}
        if not live_only or not profile.get("sections"):
            return profile
        cited = sorted({c for sec in profile["sections"] for p in sec["points"] for c in p["claims"]})
        live = {r["id"] for r in await self.get_claims(cited, live_only=True) if r["flag"] not in {"contested", "stale"}}
        sections = [{**sec, "points": [p for p in sec["points"] if set(p["claims"]) <= live]}
                    for sec in profile["sections"]]
        return {**profile, "sections": [sec for sec in sections if sec["points"]]}
