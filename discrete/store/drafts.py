"""Commit a whole episode draft atomically, after all remote work finishes."""

from __future__ import annotations

import json
from uuid import uuid4

from .. import textkit as tk
from ..models import ClaimWrite
from ..timeutil import now
from .claims import ClaimsMixin


class DraftConflict(RuntimeError):
    """A merge target changed while a draft was being prepared; retry preparation."""


class DraftsMixin(ClaimsMixin):
    async def commit_episode_draft(self, episode_id: int, *, title: str, summary: str,
                                   apps: list[str], sites: list[str], engaged_fraction: float,
                                   writes: list[ClaimWrite], candidates: list[str], rejected: dict) -> list[str]:
        def _commit():
            with self._tx() as conn:
                ep = conn.execute("SELECT status FROM episodes WHERE id=?", (episode_id,)).fetchone()
                if ep is None or ep[0] != "drafting":
                    raise DraftConflict("episode is no longer owned by this draft")
                # Check all remote decisions against the committed state before changing it.
                for w in writes:
                    for target in (w.target_id, w.refines_id, w.contested_id):
                        if target is not None:
                            row = conn.execute("SELECT status FROM claims WHERE id=?", (target,)).fetchone()
                            if row is None or row[0] != "live":
                                raise DraftConflict("merge target is no longer live")
                ops = []
                for w in writes:
                    cd = w.draft
                    if w.operation == "reinforced":
                        assert w.target_id is not None
                        cid = w.target_id
                        self._reinforce_claim(conn, cid, cd)
                    else:
                        vector = (w.vector, w.text_hash) if w.vector is not None else None
                        cid = self._insert_claim(conn, cd, lineage_id=str(uuid4()), version=1,
                                                 refines_id=w.refines_id,
                                                 invalidates_id=w.target_id if w.operation == "invalidated" else None,
                                                 vector=vector)
                        if w.operation == "invalidated":
                            assert w.target_id is not None
                            self._invalidate_claim(conn, w.target_id, by_id=cid, at=cd.occurred_from)
                        if w.contested_id is not None:
                            conn.execute("UPDATE claims SET flag='contested' WHERE id IN (?,?)", (cid, w.contested_id))
                    for eid in w.mentions:
                        conn.execute("INSERT OR IGNORE INTO mentions(entity_id, ref_kind, ref_id, created_at) "
                                     "VALUES (?, 'claim', ?, ?)", (eid, cid, now()))
                        conn.execute("UPDATE entities SET updated_at=? WHERE id=?", (now(), eid))
                    ops.append(w.operation)
                for name in candidates:
                    key = tk.entity_key(name)
                    row = conn.execute("SELECT episodes FROM entity_candidates WHERE name_norm=?", (key,)).fetchone()
                    if row:
                        eps = sorted(set(tk.parse_json_list(row[0])) | {episode_id})
                        conn.execute("UPDATE entity_candidates SET episodes=?, last_seen=? WHERE name_norm=?",
                                     (json.dumps(eps), now(), key))
                    else:
                        conn.execute("INSERT INTO entity_candidates(name_norm, name, episodes, first_seen, last_seen) "
                                     "VALUES (?,?,?,?,?)",
                                     (key, name, json.dumps([episode_id]), now(), now()))
                for reason, n in rejected.items():
                    self._metric(conn, f"claims_rejected_{reason}", n)
                conn.execute("UPDATE episodes SET title=?, summary=?, apps=?, sites=?, engaged_fraction=?, "
                             "status='closed', updated_at=? WHERE id=?",
                             (title, summary, json.dumps(apps), json.dumps(sites), engaged_fraction, now(), episode_id))
                self._enqueue(conn, "episode", episode_id)
                return ops

        return await self._call(_commit)
