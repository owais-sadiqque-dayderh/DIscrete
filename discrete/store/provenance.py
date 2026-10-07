"""Visit-level evidence reads and transactional deterministic graph writes."""

from __future__ import annotations

from dataclasses import asdict

from ..models import AttentionMoment, ProvenanceEdge
from ..timeutil import now
from .base import StoreBase

_FIELDS = ("screen_text", "caption", "window_title", "page_title", "url", "navigated_from", "app")
_SELECT = """SELECT o.*, s.id AS sighting_id, s.started_at AS at, s.ended_at AS sighting_end,
 s.keys AS sighting_keys, s.clicks AS sighting_clicks, s.clipboard AS sighting_clipboard
 FROM sightings s JOIN observations o ON o.id=s.obs_id"""


def _moment(r) -> AttentionMoment:
    # The legacy sightings schema has no session per visit. A return sighting can
    # belong to a different session from its deduplicated observation: unknown is
    # preferable to assigning it the first visit's session and blocking real edges.
    session = r["session_id"] if abs(r["at"] - r["started_at"]) < 0.001 else ""
    return AttentionMoment(r["sighting_id"], r["id"], r["at"], r["sighting_end"],
                           {k: r[k] or "" for k in _FIELDS}, r["app"], session, r["episode_id"],
                           r["sighting_keys"], r["sighting_clicks"], r["sighting_clipboard"], r["updated_at"])


class ProvenanceMixin(StoreBase):
    async def has_provenance(self) -> bool:
        return bool(await self.q1("SELECT 1 FROM sqlite_master WHERE type='table' AND name='attention_provenance'"))

    async def attention_moments(self, start: float, end: float) -> list[AttentionMoment]:
        rows = await self.q(_SELECT + " WHERE o.origin='user' AND s.started_at>=? AND s.started_at<=? "
                            "ORDER BY s.started_at, s.id", (start, end))
        return [_moment(r) for r in rows]

    async def effect_moments(self, terms: list[str], *, observation_ids=(), interval=None, limit=300,
                              before: float | None = None):
        """Indexed search followed by exact visit/time filtering in the resolver."""
        from .. import textkit as tk

        ids = set(observation_ids)
        if terms:
            fts = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            cutoff = " AND o.started_at<=?" if before is not None else ""
            args = (fts, before, limit) if before is not None else (fts, limit)
            rows = await self.q("SELECT f.rowid FROM observations_fts f JOIN observations o ON o.id=f.rowid "
                                "WHERE observations_fts MATCH ? AND o.origin='user'" + cutoff +
                                " ORDER BY bm25(observations_fts),o.started_at DESC,o.id DESC LIMIT ?", args)
            ids.update(r[0] for r in rows)
            # Structured page fields are not in the legacy FTS index.
            for t in terms[:6]:
                if len(t) >= 3:
                    needle = "%" + tk.norm_key(t).replace("%", r"\%").replace("_", r"\_") + "%"
                    cutoff = " AND started_at<=?" if before is not None else ""
                    args = (needle, needle, before, limit) if before is not None else (needle, needle, limit)
                    rows = await self.q("SELECT id FROM observations WHERE origin='user' AND "
                                        "(page_title LIKE ? ESCAPE '\\' OR url LIKE ? ESCAPE '\\')" + cutoff +
                                        " ORDER BY started_at DESC,id DESC LIMIT ?", args)
                    ids.update(r[0] for r in rows)
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        sql = _SELECT + f" WHERE o.origin='user' AND o.id IN ({ph})"
        args = list(ids)
        if before is not None:
            sql += " AND s.started_at<=?"
            args.append(before)
        if interval:
            sql += " AND s.started_at>=? AND s.started_at<?"
            args += list(interval)
        rows = await self.q(sql + " ORDER BY s.started_at DESC, s.id DESC LIMIT ?", (*args, limit))
        return [_moment(r) for r in rows]

    async def pending_provenance(self, rules_key: str, *, limit: int, window_s: float):
        # Window freshness avoids a correlated rescan for every checkpoint. Include
        # pending sources: a later enrichment must invalidate an empty destination.
        rows = await self.q("WITH visits AS (" + _SELECT +
                            " WHERE o.origin='user'), freshness AS (SELECT visits.*, "
                            "MAX(updated_at) OVER (ORDER BY at RANGE BETWEEN ? PRECEDING AND CURRENT ROW) AS latest "
                            "FROM visits) SELECT f.* FROM freshness f "
                            "LEFT JOIN provenance_inputs p ON p.sighting_id=f.sighting_id "
                            "WHERE f.enrich_state!='pending' AND (p.sighting_id IS NULL OR "
                            "p.updated_at<f.latest OR p.rules_key!=?) ORDER BY f.at,f.sighting_id LIMIT ?",
                            (window_s, rules_key, limit))
        return [_moment(r) for r in rows]

    async def provenance_edges(self, start: float, end: float, *, rules_key: str | None = None) -> list[ProvenanceEdge]:
        if not await self.has_provenance():
            return []  # old read-only stores can build ephemeral edges
        sql = "SELECT p.* FROM attention_provenance p JOIN sightings s ON s.id=p.target "
        args = [start, end]
        if rules_key is not None:
            sql += "JOIN provenance_inputs i ON i.sighting_id=p.target "
        sql += "WHERE s.started_at>=? AND s.started_at<=?"
        if rules_key is not None:
            sql += " AND i.rules_key=?"
            args.append(rules_key)
        rows = await self.q(sql, args)
        return [ProvenanceEdge(**(dict(r) | {"term": (r["term"] or "") if "term" in r.keys() else ""}))
                for r in rows]

    async def provenance_checkpoints(self, start: float, end: float, rules_key: str) -> dict[int, float]:
        if not await self.has_provenance():
            return {}
        rows = await self.q("SELECT p.sighting_id,p.updated_at FROM provenance_inputs p "
                            "JOIN sightings s ON s.id=p.sighting_id WHERE s.started_at>=? AND s.started_at<=? "
                            "AND p.rules_key=?", (start, end, rules_key))
        return {r["sighting_id"]: r["updated_at"] for r in rows}

    async def save_provenance(self, moment: AttentionMoment, edges: list[ProvenanceEdge], rules_key: str,
                              *, checkpoint_at: float | None = None) -> int:
        return await self.save_provenance_batch([(moment, edges, checkpoint_at if checkpoint_at is not None else now())],
                                                rules_key)

    async def save_provenance_batch(self, batches, rules_key: str) -> int:
        """One snapshot read and transaction per batch, preserving endpoint checks."""
        from ..process.provenance import valid_edge

        def write():
            with self._tx() as conn:
                ids = {m.sighting_id for m, _, _ in batches}
                ids.update(e.source for _, edges, _ in batches for e in edges)
                by_id = {}
                # SQLite builds may have a small parameter limit.
                ordered_ids = sorted(ids)
                for offset in range(0, len(ordered_ids), 500):
                    chunk = ordered_ids[offset:offset + 500]
                    rows = conn.execute(_SELECT + " WHERE o.origin='user' AND s.id IN (" +
                                        ",".join("?" for _ in chunk) + ")", chunk).fetchall()
                    by_id.update((r["sighting_id"], _moment(r)) for r in rows)
                made = 0
                for moment, edges, checkpoint in batches:
                    current = by_id.get(moment.sighting_id)
                    if current is None or current.updated_at != moment.updated_at:
                        continue
                    if any(edge.target != moment.sighting_id for edge in edges):
                        raise ValueError("edge target must be the checkpointed visit")
                    if any(not valid_edge(edge, by_id) for edge in edges):
                        continue  # source changed while inference was in flight
                    conn.execute("DELETE FROM attention_provenance WHERE target=?", (moment.sighting_id,))
                    for edge in edges:
                        d = asdict(edge)
                        conn.execute(f"INSERT INTO attention_provenance({','.join(d)}) VALUES ({','.join('?' * len(d))})",
                                     tuple(d.values()))
                    conn.execute("INSERT OR REPLACE INTO provenance_inputs VALUES (?,?,?)",
                                 (moment.sighting_id, checkpoint, rules_key))
                    made += len(edges)
                return made
        return await self._call(write)

    async def reset_provenance(self) -> None:
        def write():
            with self._tx() as conn:
                conn.execute("DELETE FROM attention_provenance")
                conn.execute("DELETE FROM provenance_inputs")
        await self._call(write)
