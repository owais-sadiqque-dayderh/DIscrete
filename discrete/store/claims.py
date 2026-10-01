"""Claims, evidence and derived confidence (planS2 §5-9)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from .. import textkit as tk
from ..confidence import compute_confidence, modality_of
from ..config import T
from ..models import ClaimDraft
from ..timeutil import day_key, now
from .base import StoreBase, rowid

# (vector, text_hash) of the claim's embedding text, when the caller already has it.
Vector = tuple[Sequence[float], str]


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


class ClaimsMixin(StoreBase):
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

    def _link(self, conn, claim_id: int, observation_id: int, relation: str, quote: str) -> None:
        conn.execute(
            """INSERT INTO evidence(claim_id, observation_id, relation, quote, created_at) VALUES (?,?,?,?,?)
               ON CONFLICT(claim_id, observation_id) DO UPDATE SET
                 relation=CASE WHEN excluded.relation='supports' THEN 'supports' ELSE evidence.relation END,
                 quote=CASE WHEN excluded.relation='supports' THEN excluded.quote ELSE evidence.quote END""",
            (claim_id, observation_id, relation, quote, now()),
        )

    def _cap_evidence(self, conn, claim_id: int) -> None:
        """planS2 §9: ≤ EVIDENCE_MAX_SUPPORTS supports; cooccurs dropped once ≥3 supports."""
        sup = conn.execute(
            "SELECT observation_id, created_at FROM evidence WHERE claim_id=? AND relation='supports' "
            "ORDER BY created_at DESC",
            (claim_id,),
        ).fetchall()
        if len(sup) >= 3:
            conn.execute("DELETE FROM evidence WHERE claim_id=? AND relation='cooccurs'", (claim_id,))
        if len(sup) > T.EVIDENCE_MAX_SUPPORTS:
            for r in sup[T.EVIDENCE_MAX_SUPPORTS:]:
                conn.execute("DELETE FROM evidence WHERE claim_id=? AND observation_id=?", (claim_id, r[0]))

    def _insert_claim(self, conn, draft: ClaimDraft, *, lineage_id: str, version: int,
                      refines_id: int | None = None, invalidates_id: int | None = None,
                      derived_from: list[int] | None = None, vector: Vector | None = None) -> int:
        ts = now()
        cur = conn.execute(
            """INSERT INTO claims(lineage_id, version, kind, text, context, reasoning, modality,
                 occurred_from, occurred_to, valid_from, status, episode_id, refines_id, invalidates_id,
                 confidence, derived_from, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,'live',?,?,?,0,?,?,?)""",
            (lineage_id, version, draft.kind, tk.truncate(draft.text, T.CLAIM_MAX_CHARS), draft.context,
             draft.reasoning, draft.modality, draft.occurred_from or ts, draft.occurred_to or ts,
             draft.occurred_from or ts, draft.episode_id, refines_id, invalidates_id,
             json.dumps(derived_from or []), ts, ts),
        )
        cid = rowid(cur)
        for ev in draft.evidence:
            self._link(conn, cid, int(ev["observation_id"]), "supports" if ev.get("quote") else "cooccurs",
                       str(ev.get("quote") or ""))
        self._cap_evidence(conn, cid)
        self._recompute_confidence(conn, cid)
        if vector is not None:
            self._write_vector(conn, "claim", cid, vector[0], vector[1])  # embedded once, by the caller
        else:
            self._enqueue(conn, "claim", cid)
        return cid

    async def create_claim(self, draft: ClaimDraft, *, refines_id: int | None = None,
                           invalidates_id: int | None = None, vector: Vector | None = None) -> int:
        def _create():
            with self._tx() as conn:
                return self._insert_claim(conn, draft, lineage_id=str(uuid4()), version=1,
                                          refines_id=refines_id, invalidates_id=invalidates_id, vector=vector)

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

    async def reinforce_claim(self, claim_id: int, draft: ClaimDraft) -> float:
        """REINFORCE: add verified evidence, extend occurred_to; text untouched (planS2 §6.2)."""

        def _r():
            with self._tx() as conn:
                for ev in draft.evidence:
                    self._link(conn, claim_id, int(ev["observation_id"]), "supports", str(ev.get("quote") or ""))
                conn.execute(
                    "UPDATE claims SET occurred_to=MAX(occurred_to, ?), updated_at=? WHERE id=?",
                    (draft.occurred_to, now(), claim_id),
                )
                self._cap_evidence(conn, claim_id)
                return self._recompute_confidence(conn, claim_id)

        return await self._call(_r)

    async def invalidate_claim(self, claim_id: int, *, by_id: int, at: float) -> None:
        def _inv():
            with self._tx() as conn:
                conn.execute(
                    "UPDATE claims SET status='invalidated', valid_to=?, updated_at=? WHERE id=? AND status='live'",
                    (at, now(), claim_id),
                )
                conn.execute("UPDATE claims SET invalidates_id=? WHERE id=?", (claim_id, by_id))
                self._drop_vector(conn, "claim", claim_id)

        await self._call(_inv)

    async def flag_claims(self, ids: Sequence[int], flag: str) -> None:
        if ids:
            await self.execute(f"UPDATE claims SET flag=? WHERE id IN ({_ph(ids)})", [flag, *ids])

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
            "SELECT * FROM evidence WHERE claim_id=? ORDER BY relation='supports' DESC, created_at", (claim_id,)
        )

    async def consolidate_cluster(self, cluster: list[sqlite3.Row], outputs: list[dict]) -> list[int]:
        """planS2 §7.1: N inputs → a bounded set. Each output inherits evidence only
        from the inputs it derives from; inputs no output uses stay live."""
        cluster_ids = {r["id"] for r in cluster}
        by_id = {r["id"]: r for r in cluster}

        def _run():
            new_ids: list[int] = []
            with self._tx() as conn:
                replaced: dict[int, int] = {}
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
                        reasoning="consolidated", modality=primary["modality"], episode_id=primary["episode_id"],
                        occurred_from=min(by_id[i]["occurred_from"] for i in inputs),
                        occurred_to=max(by_id[i]["occurred_to"] for i in inputs),
                        evidence=[{"observation_id": r[0], "quote": r[1] if r[2] == "supports" else ""} for r in ev],
                    )
                    nid = self._insert_claim(conn, draft, lineage_id=primary["lineage_id"],
                                             version=lineage_version + 1, derived_from=inputs)
                    self._carry_mentions(conn, inputs, nid)
                    new_ids.append(nid)
                    for i in inputs:
                        replaced.setdefault(i, nid)
                ts = now()
                for old, nid in replaced.items():
                    conn.execute(
                        "UPDATE claims SET status='superseded', valid_to=?, superseded_by=?, updated_at=? WHERE id=?",
                        (ts, nid, ts, old),
                    )
                    self._drop_vector(conn, "claim", old)
            return new_ids

        return await self._call(_run)

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
                    conn.execute(
                        "UPDATE claims SET status='superseded', valid_to=?, flag='unsupported', updated_at=? WHERE id=?",
                        (ts, ts, cid),
                    )
                    self._drop_vector(conn, "claim", cid)
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
