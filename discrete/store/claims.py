"""Claims, evidence and derived confidence (planS2 §5-9)."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from .. import textkit as tk
from ..confidence import compute_confidence, modality_of
from ..config import T
from ..models import ClaimDraft
from ..timeutil import day_key, now
from .base import rowid
from .history import HistoryMixin

# (vector, text_hash) of the claim's embedding text, when the caller already has it.
Vector = tuple[Sequence[float], str]


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


class ClaimsMixin(HistoryMixin):
    def _recompute_confidence(self, conn, claim_id: int) -> float:
        """planS2 §8: per distinct supporting episode, its best evidence modality."""
        row = conn.execute("SELECT modality FROM claims WHERE id=?", (claim_id,)).fetchone()
        rows = conn.execute(
            "SELECT o.episode_id, o.keys, o.clicks, o.clipboard, o.started_at FROM evidence e "
            "JOIN observations o ON o.id=e.observation_id WHERE e.claim_id=? AND e.relation='supports'",
            (claim_id,),
        ).fetchall()
        best: dict[Any, str] = {}
        days: set[str] = set()
        for r in rows:
            mod = modality_of(r["keys"], r["clicks"], r["clipboard"])
            ep = r["episode_id"] if r["episode_id"] is not None else f"obs{id(r)}"
            if best.get(ep) != "engaged":
                best[ep] = mod
            days.add(day_key(r["started_at"]))
        mods = list(best.values())
        if row and row["modality"] == "stated":
            mods.append("stated")
        conf = compute_confidence(mods, max(1, len(days)))
        modality = "stated" if row and row["modality"] == "stated" else ("engaged" if "engaged" in mods else "passive")
        conn.execute("UPDATE claims SET confidence=?, modality=? WHERE id=?", (conf, modality, claim_id))
        return conf

    def _link(self, conn, claim_id: int, observation_id: int, relation: str, quote: str, field: str = "") -> None:
        conn.execute(
            """INSERT INTO evidence(claim_id, observation_id, relation, quote, field, created_at) VALUES (?,?,?,?,?,?)
               ON CONFLICT(claim_id, observation_id) DO UPDATE SET
                 relation=CASE WHEN excluded.relation='supports' THEN 'supports' ELSE evidence.relation END,
                 quote=CASE WHEN excluded.relation='supports' THEN excluded.quote ELSE evidence.quote END,
                 field=CASE WHEN excluded.relation='supports' THEN excluded.field ELSE evidence.field END""",
            (claim_id, observation_id, relation, quote, field, now()),
        )

    def _cap_evidence(self, conn, claim_id: int) -> None:
        """Keep full support membership; only model-facing quote samples are capped."""
        sup = conn.execute(
            "SELECT observation_id, created_at FROM evidence WHERE claim_id=? AND relation='supports' "
            "ORDER BY created_at DESC",
            (claim_id,),
        ).fetchall()
        if len(sup) >= 3:
            conn.execute("DELETE FROM evidence WHERE claim_id=? AND relation='cooccurs'", (claim_id,))

    def _insert_claim(self, conn, draft: ClaimDraft, *, lineage_id: str, version: int,
                      refines_id: int | None = None, invalidates_id: int | None = None,
                      derived_from: list[int] | None = None, vector: Vector | None = None,
                      allow_suppressed: bool = False) -> int:
        text = tk.truncate(draft.text, T.CLAIM_MAX_CHARS)
        if not allow_suppressed and self._suppressed(conn, draft.kind, text, vector[0] if vector else None):
            raise ValueError("claim is suppressed")
        ts = now()
        cur = conn.execute(
            """INSERT INTO claims(lineage_id, version, kind, text, context, reasoning, modality,
                 occurred_from, occurred_to, valid_from, status, episode_id, refines_id, invalidates_id,
                 confidence, derived_from, action, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,'live',?,?,?,0,?,?,?,?)""",
            (lineage_id, version, draft.kind, text, draft.context,
             draft.reasoning, draft.modality, draft.occurred_from or ts, draft.occurred_to or ts,
             draft.occurred_from or ts, draft.episode_id, refines_id, invalidates_id,
             json.dumps(derived_from or []), draft.action, ts, ts),
        )
        cid = rowid(cur)
        for ev in draft.evidence:
            self._link(conn, cid, int(ev["observation_id"]), "supports" if ev.get("quote") else "cooccurs",
                       str(ev.get("quote") or ""), str(ev.get("field") or ""))
        self._cap_evidence(conn, cid)
        self._recompute_confidence(conn, cid)
        if vector is not None:
            self._write_vector(conn, "claim", cid, vector[0], vector[1])  # embedded once, by the caller
        else:
            self._enqueue(conn, "claim", cid)
        return cid

    async def create_claim(self, draft: ClaimDraft, *, refines_id: int | None = None,
                           invalidates_id: int | None = None, vector: Vector | None = None,
                           actor: str = "user", reason: str = "") -> int:
        def _create():
            with self._tx() as conn:
                cid = self._insert_claim(conn, draft, lineage_id=str(uuid4()), version=1,
                                         refines_id=refines_id, invalidates_id=invalidates_id, vector=vector)
                self._log_event(conn, "add", actor, claim_id=cid, created=[cid], reason=reason or draft.reasoning)
                return cid

        return await self._call(_create)

    async def get_claim(self, claim_id: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM claims WHERE id=?", (claim_id,))

    async def get_claims(self, ids: Sequence[int], *, live_only: bool = False) -> list[sqlite3.Row]:
        if not ids:
            return []
        extra = " AND status='live'" if live_only else ""
        return await self.q(f"SELECT * FROM claims WHERE id IN ({_ph(ids)}){extra}", list(ids))

    async def first_quotes(self, claim_ids: Sequence[int]) -> dict[int, str]:
        """One supporting quote per claim, in one query."""
        if not claim_ids:
            return {}
        rows = await self.q(
            f"SELECT claim_id, quote FROM evidence WHERE claim_id IN ({_ph(claim_ids)}) "
            f"AND relation='supports' AND quote!='' ORDER BY created_at", list(claim_ids))
        out: dict[int, str] = {}
        for r in rows:
            out.setdefault(r[0], r[1])
        return out

    async def evidence_summary(self, claim_ids: Sequence[int], *, per_claim: int = 1) -> dict[int, list[dict]]:
        """Supporting screen moments: quoted, engaged observations first, then recent ones."""
        out: dict[int, list[dict]] = {cid: [] for cid in claim_ids}
        if not claim_ids or per_claim <= 0:
            return out
        rows = await self.q(
            f"SELECT e.claim_id, o.id AS observation_id, o.app, o.window_title, o.site, o.url, "
            f"o.started_at, e.quote, c.context FROM evidence e JOIN observations o ON o.id=e.observation_id "
            f"JOIN claims c ON c.id=e.claim_id WHERE e.claim_id IN ({_ph(claim_ids)}) AND e.relation='supports' "
            f"ORDER BY e.quote!='' DESC, (o.keys>0 OR o.clicks>0 OR o.clipboard>0) DESC, "
            f"o.started_at DESC, o.id DESC", list(claim_ids))
        for r in rows:
            items = out[r["claim_id"]]
            if len(items) < per_claim:
                items.append({"observation_id": r["observation_id"], "app": r["app"],
                              "window_title": r["window_title"], "site": tk.site_from_url(r["url"]) or r["site"],
                              "started_at": float(r["started_at"]), "quote": r["quote"], "context": r["context"]})
        return out

    def _reinforce_claim(self, conn, claim_id: int, draft: ClaimDraft) -> dict:
        """REINFORCE: add verified evidence, extend occurred_to; text untouched (planS2 §6.2).
        Returns the change for the event log: evidence added and evicted by the cap, in full."""
        current = conn.execute("SELECT kind, text, status FROM claims WHERE id=?", (claim_id,)).fetchone()
        if current is None or current["status"] != "live" or self._suppressed(conn, current["kind"], current["text"]):
            raise ValueError("reinforcement target is withdrawn or suppressed")
        before = self._evidence_rows(conn, claim_id)
        old_to = conn.execute("SELECT occurred_to FROM claims WHERE id=?", (claim_id,)).fetchone()[0]
        for ev in draft.evidence:
            self._link(conn, claim_id, int(ev["observation_id"]), "supports", str(ev.get("quote") or ""),
                       str(ev.get("field") or ""))
        conn.execute("UPDATE claims SET occurred_to=MAX(occurred_to, ?), updated_at=? WHERE id=?",
                     (draft.occurred_to, now(), claim_id))
        self._cap_evidence(conn, claim_id)
        self._recompute_confidence(conn, claim_id)
        new_to = conn.execute("SELECT occurred_to FROM claims WHERE id=?", (claim_id,)).fetchone()[0]
        return self._evidence_change(conn, claim_id, before, fields_before={"occurred_to": old_to},
                                     fields_after={"occurred_to": new_to})

    def _recompute_after_revert(self, conn, claim_id: int) -> None:
        self._recompute_confidence(conn, claim_id)

    def _invalidate_claim(self, conn, claim_id: int, *, by_id: int, at: float) -> dict | None:
        target = conn.execute("SELECT occurred_to FROM claims WHERE id=?", (claim_id,)).fetchone()
        incoming = conn.execute("SELECT occurred_to FROM claims WHERE id=?", (by_id,)).fetchone()
        if target is None or incoming is None or max(at, incoming[0]) < target[0]:
            return None  # arrival order cannot turn older evidence into a newer truth
        change = self._set_state(conn, claim_id, only_if_live=True, status="invalidated", valid_to=at)
        if change is None:
            return None
        conn.execute("UPDATE claims SET invalidates_id=? WHERE id=?", (claim_id, by_id))
        return change

    def _pattern_support_valid(self, conn, category: str, text: str, support: Sequence[int]) -> bool:
        if not support:
            return False
        rows = conn.execute(
            f"SELECT o.episode_id, o.started_at FROM evidence e JOIN observations o ON o.id=e.observation_id "
            f"WHERE e.relation='supports' AND e.claim_id IN ({_ph(support)})", list(support)).fetchall()
        eps = {r[0] for r in rows if r[0] is not None}
        times = sorted({r[1] for r in rows})
        days = {day_key(t) for t in times}
        occasions = (1 + sum(b - a >= T.DREAM_OCCASION_GAP_MIN * 60
                             for a, b in zip(times, times[1:], strict=False))) if times else 0
        frequency = re.search(r"\b(?:always|daily|every ?day|each day|every (?:morning|evening|night|week)|"
                              r"regularly|routinely|usually|habitually|constantly|never)\b", text, re.I)
        return (len(eps) >= T.DREAM_MIN_EPISODES and occasions >= T.DREAM_MIN_OCCASIONS
                and (category not in {"habit", "preference"} or len(days) >= T.DREAM_MIN_DAYS)
                and (not frequency or len(days) >= T.DREAM_FREQUENCY_DAYS))

    def _withdraw_dependents(self, conn, withdrawn: set[int]) -> list[dict]:
        """Remove copied support immediately, keeping only sufficiently backed inductions.
        Record the evidence and dependency changes in the triggering event for undo."""
        changes = []
        while withdrawn:
            next_withdrawn = set()
            for p in conn.execute("SELECT * FROM claims WHERE status='live' AND kind='pattern'").fetchall():
                support = json.loads(p["derived_from"] or "[]")
                if not withdrawn.intersection(support):
                    continue
                live = [r[0] for r in conn.execute(
                    f"SELECT id FROM claims WHERE id IN ({_ph(support)}) AND status='live' AND flag!='contested'",
                    support)]
                before = self._evidence_rows(conn, p["id"])
                conn.execute("DELETE FROM evidence WHERE claim_id=?", (p["id"],))
                if live:
                    for ev in conn.execute(f"SELECT * FROM evidence WHERE relation='supports' "
                                           f"AND claim_id IN ({_ph(live)})", live).fetchall():
                        self._link(conn, p["id"], ev["observation_id"], "supports", ev["quote"], ev["field"])
                conn.execute("UPDATE claims SET derived_from=?, updated_at=? WHERE id=?",
                             (json.dumps(sorted(live)), now(), p["id"]))
                self._recompute_confidence(conn, p["id"])
                change = self._evidence_change(conn, p["id"], before,
                    fields_before={"derived_from": p["derived_from"]}, fields_after={"derived_from": json.dumps(sorted(live))})
                if not self._pattern_support_valid(conn, p["context"], p["text"], live):
                    state = self._set_state(conn, p["id"], status="retired", valid_to=now())
                    if state:
                        change.update(before=state["before"], after=state["after"])
                    next_withdrawn.add(p["id"])
                changes.append(change)
            withdrawn = next_withdrawn
        return changes

    async def live_claims(self, *, kind: str | None = None, limit: int | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM claims WHERE status='live'", []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY updated_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return await self.q(sql, args)

    async def live_claims_with_quotes(self) -> list[tuple[int, str, list[str]]]:
        """(id, text, supporting quotes) for every live claim, in one query."""
        rows = await self.q(
            "SELECT c.id, c.text, e.quote FROM claims c LEFT JOIN evidence e ON e.claim_id=c.id AND e.quote!='' "
            "WHERE c.status='live' ORDER BY c.id")
        out: dict[int, tuple[int, str, list[str]]] = {}
        for cid, text, quote in rows:
            item = out.setdefault(cid, (cid, text, []))
            if quote:
                item[2].append(quote)
        return list(out.values())

    async def evidence_for_claim(self, claim_id: int) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM evidence WHERE claim_id=? ORDER BY relation='supports' DESC, created_at DESC LIMIT ?",
            (claim_id, T.EVIDENCE_MAX_SUPPORTS)
        )

    async def consolidate_cluster(self, cluster: list[sqlite3.Row], outputs: list[dict], *,
                                  run_id: str = "") -> list[int]:
        """planS2 §7.1: N inputs → a bounded set. Each output inherits evidence only
        from the inputs it derives from; inputs no output uses stay live."""
        cluster_ids = {r["id"] for r in cluster}
        by_id = {r["id"]: r for r in cluster}

        def _run():
            new_ids: list[int] = []
            with self._tx() as conn:
                current = conn.execute(f"SELECT id, status, text FROM claims WHERE id IN ({_ph(list(cluster_ids))})",
                                       list(cluster_ids)).fetchall()
                if len(current) != len(cluster_ids) or any(r["status"] != "live" or
                        r["text"] != by_id[r["id"]]["text"] for r in current):
                    return []  # another draft/refinement already replaced the inputs
                replaced: dict[int, int] = {}
                profile_before = self._current_profile(conn)
                cited = {c for sec in (profile_before or {}).get("sections", []) for point in sec["points"]
                         for c in point["claims"]}
                for row in conn.execute("SELECT derived_from FROM claims WHERE status='live' AND kind='pattern'"):
                    cited.update(json.loads(row[0] or "[]"))
                for out in outputs[: len(cluster) + 2]:
                    inputs = [int(x) for x in (out.get("input_ids") or []) if int(x) in cluster_ids]
                    text = str(out.get("text") or "").strip()
                    if not inputs or not text:
                        continue
                    primary = by_id[inputs[0]]
                    ev = conn.execute(
                        f"SELECT observation_id, quote, relation FROM evidence WHERE claim_id IN ({_ph(inputs)})",
                        inputs,
                    ).fetchall()
                    lineage_version = conn.execute(
                        "SELECT MAX(version) FROM claims WHERE lineage_id=?", (primary["lineage_id"],)
                    ).fetchone()[0] or 1
                    draft = ClaimDraft(
                        kind=primary["kind"], text=text, context=primary["context"],
                        reasoning="consolidated", modality=("stated" if any(by_id[i]["modality"] == "stated"
                            and tk.norm_key(by_id[i]["text"]) == tk.norm_key(text) for i in inputs)
                            else "engaged" if any(by_id[i]["modality"] in {"engaged", "stated"} for i in inputs)
                            else "passive"), episode_id=primary["episode_id"],
                        occurred_from=min(by_id[i]["occurred_from"] for i in inputs),
                        occurred_to=max(by_id[i]["occurred_to"] for i in inputs),
                        evidence=[{"observation_id": r[0], "quote": r[1] if r[2] == "supports" else ""} for r in ev],
                    )
                    nid = self._insert_claim(conn, draft, lineage_id=primary["lineage_id"],
                                             version=lineage_version + 1, derived_from=inputs)
                    self._carry_mentions(conn, inputs, nid)
                    new_ids.append(nid)
                    for i in inputs:
                        equivalent = tk.norm_key(by_id[i]["text"]) == tk.norm_key(text)
                        # input_ids prove support for the output, not entailment of each premise.
                        # Keep cited or user-stated premises unless equivalence is established.
                        if equivalent or (i not in cited and by_id[i]["modality"] != "stated"):
                            replaced.setdefault(i, nid)
                ts = now()
                dependencies = self._remap_dependencies(conn, {old: nid for old, nid in replaced.items()
                    if tk.norm_key(by_id[old]["text"]) == tk.norm_key(
                        conn.execute("SELECT text FROM claims WHERE id=?", (nid,)).fetchone()[0])})
                changes = [self._set_state(conn, old, status="superseded", valid_to=ts, superseded_by=nid)
                           for old, nid in replaced.items()] + dependencies
                if new_ids:
                    self._log_event(conn, "merge", "consolidate", claim_id=new_ids[0], created=new_ids,
                                    changes=changes, run_id=run_id,
                                    reason=f"merged {len(replaced)} overlapping claims into {len(new_ids)}",
                                    data={"profile_before": profile_before, "profile_after": self._current_profile(conn)}
                                    if profile_before != self._current_profile(conn) else None)
            return new_ids

        return await self._call(_run)

    def _remap_dependencies(self, conn, replaced: dict[int, int]) -> list[dict]:
        changes = []
        if not replaced:
            return changes
        for p in conn.execute("SELECT id, derived_from FROM claims WHERE status='live' AND kind='pattern'").fetchall():
            ids = json.loads(p["derived_from"] or "[]")
            if not set(ids) & replaced.keys():
                continue
            updated = json.dumps(sorted({replaced.get(i, i) for i in ids}))
            conn.execute("UPDATE claims SET derived_from=?, updated_at=? WHERE id=?", (updated, now(), p["id"]))
            changes.append({"id": p["id"], "fields_before": {"derived_from": p["derived_from"]},
                            "fields_after": {"derived_from": updated}})
        profile = self._current_profile(conn)
        if profile:
            for sec in profile.get("sections", []):
                for point in sec["points"]:
                    point["claims"] = sorted({replaced.get(i, i) for i in point["claims"]})
            self._set_profile(conn, profile)
        return changes

    def _carry_mentions(self, conn, old_ids: list[int], new_id: int) -> None:
        for (eid,) in conn.execute(
            f"SELECT DISTINCT entity_id FROM mentions WHERE ref_kind='claim' AND ref_id IN ({_ph(old_ids)})",
            old_ids,
        ).fetchall():
            conn.execute(
                "INSERT OR IGNORE INTO mentions(entity_id, ref_kind, ref_id, created_at) VALUES (?, 'claim', ?, ?)",
                (eid, new_id, now()),
            )

    async def supersede_unsupported(self, ttl_days: float) -> int:
        """planS2 §7.5; `stated` claims rest on user testimony and are exempt (Rev 3)."""
        cutoff = now() - ttl_days * 86400

        def _run():
            with self._tx() as conn:
                rows = conn.execute(
                    """SELECT c.id FROM claims c WHERE c.status='live' AND c.modality!='stated' AND c.created_at < ?
                       AND NOT EXISTS (SELECT 1 FROM evidence e WHERE e.claim_id=c.id AND e.relation='supports')""",
                    (cutoff,),
                ).fetchall()
                ts = now()
                for (cid,) in rows:
                    change = self._set_state(conn, cid, status="superseded", valid_to=ts, flag="unsupported")
                    self._log_event(conn, "expire", "cleanup", claim_id=cid, changes=[change],
                                    reason=f"no supporting evidence after {ttl_days:g} days")
                return len(rows)

        return await self._call(_run)

    async def lineage_anomalies(self) -> int:
        return await self.scalar(
            "SELECT COUNT(*) FROM (SELECT lineage_id FROM claims GROUP BY lineage_id HAVING MAX(version) > ?)",
            (T.LINEAGE_MAX_VERSIONS,),
        )

    async def confidence_histogram(self, *, decimals: int = 3) -> list[tuple[float, int]]:
        rows = await self.q("SELECT ROUND(confidence, ?) c, COUNT(*) n FROM claims WHERE status='live' "
                            "GROUP BY c ORDER BY n DESC", (decimals,))
        return [(r[0], r[1]) for r in rows]
