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

    async def effect_moments(self, terms: list[str], *, observation_ids=(), interval=None, limit=300):
        """Indexed search followed by exact visit/time filtering in the resolver."""
        from .. import textkit as tk

        ids = set(observation_ids)
        if terms:
            fts = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            rows = await self.q("SELECT rowid FROM observations_fts WHERE observations_fts MATCH ? "
                                "ORDER BY bm25(observations_fts) LIMIT ?", (fts, limit))
            ids.update(r[0] for r in rows)
            # Structured page fields are not in the legacy FTS index.
            for t in terms[:6]:
                if len(t) >= 3:
                    needle = "%" + tk.norm_key(t).replace("%", r"\%").replace("_", r"\_") + "%"
                    rows = await self.q("SELECT id FROM observations WHERE origin='user' AND "
                                        "(page_title LIKE ? ESCAPE '\\' OR url LIKE ? ESCAPE '\\') LIMIT ?",
                                        (needle, needle, limit))
                    ids.update(r[0] for r in rows)
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        sql = _SELECT + f" WHERE o.origin='user' AND o.id IN ({ph})"
        args = list(ids)
        if interval:
            sql += " AND s.started_at>=? AND s.started_at<?"
            args += list(interval)
        rows = await self.q(sql + " ORDER BY s.started_at DESC, s.id DESC LIMIT ?", (*args, limit))
        return [_moment(r) for r in rows]

    async def pending_provenance(self, rules_key: str, *, limit: int, window_s: float):
        rows = await self.q(_SELECT + " LEFT JOIN provenance_inputs p ON p.sighting_id=s.id "
                            "WHERE o.origin='user' AND o.enrich_state!='pending' AND "
                            "(p.sighting_id IS NULL OR p.updated_at<o.updated_at OR p.rules_key!=? OR EXISTS "
                            "(SELECT 1 FROM sightings n JOIN observations c ON c.id=n.obs_id "
                            "WHERE n.started_at>=s.started_at-? AND n.started_at<=s.started_at "
                            "AND c.updated_at>p.updated_at)) "
                            "ORDER BY s.started_at, s.id LIMIT ?", (rules_key, window_s, limit))
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
        return [ProvenanceEdge(**dict(r)) for r in rows]

    async def save_provenance(self, moment: AttentionMoment, edges: list[ProvenanceEdge], rules_key: str,
                              *, checkpoint_at: float | None = None) -> int:
        def write():
            with self._tx() as conn:
                # Don't checkpoint a snapshot changed while inference was in flight.
                current = conn.execute("SELECT updated_at, origin FROM observations WHERE id=?",
                                       (moment.observation_id,)).fetchone()
                if current is None or current["updated_at"] != moment.updated_at or current["origin"] != "user":
                    return 0
                for edge in edges:
                    if edge.target != moment.sighting_id:
                        raise ValueError("edge target must be the checkpointed visit")
                    row = conn.execute(_SELECT + " WHERE s.id=? AND o.origin='user'", (edge.source,)).fetchone()
                    if row is None:
                        return 0
                    source = _moment(row)
                    if not (edge.source_quote and edge.source_quote in source.fields.get(edge.source_field, "")
                            and edge.target_quote and edge.target_quote in moment.fields.get(edge.target_field, "")
                            and (source.at, source.sighting_id) < (moment.at, moment.sighting_id)
                            and abs(edge.gap_s - (moment.at - source.at)) < 0.001):
                        return 0  # a changed source will be reconsidered next time
                conn.execute("DELETE FROM attention_provenance WHERE target=?", (moment.sighting_id,))
                for edge in edges:
                    d = asdict(edge)
                    conn.execute(f"INSERT INTO attention_provenance({','.join(d)}) VALUES ({','.join('?' * len(d))})",
                                 tuple(d.values()))
                conn.execute("INSERT OR REPLACE INTO provenance_inputs VALUES (?,?,?)",
                             (moment.sighting_id, checkpoint_at if checkpoint_at is not None else now(), rules_key))
                return len(edges)
        return await self._call(write)

    async def reset_provenance(self) -> None:
        def write():
            with self._tx() as conn:
                conn.execute("DELETE FROM attention_provenance")
                conn.execute("DELETE FROM provenance_inputs")
        await self._call(write)
