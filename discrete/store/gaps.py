"""Knowledge gaps and recorded retrieval misses (planS3 §8)."""

from __future__ import annotations

import sqlite3

from ..timeutil import now
from .base import StoreBase, rowid


class GapsMixin(StoreBase):
    async def record_gap(self, question: str) -> int:
        def _rec():
            return rowid(self.conn.execute(
                "INSERT INTO gaps(question, status, created_at) VALUES (?, 'open', ?)", (question, now())))

        return await self._call(_rec)

    async def record_retrieval_miss(self, query: str, missed: str, stage: str) -> None:
        await self.execute(
            "INSERT INTO retrieval_misses(query, missed, stage, created_at) VALUES (?,?,?,?)",
            (query, missed, stage, now()),
        )

    async def answer_gap(self, gap_id: int, answer: str) -> sqlite3.Row | None:
        def _ans():
            with self._tx() as conn:
                if not conn.execute("SELECT 1 FROM gaps WHERE id=? AND status='open'", (gap_id,)).fetchone():
                    return None
                conn.execute("UPDATE gaps SET status='answered', answer=?, answered_at=? WHERE id=?",
                             (answer.strip(), now(), gap_id))
                return conn.execute("SELECT * FROM gaps WHERE id=?", (gap_id,)).fetchone()

        return await self._call(_ans)

    async def link_gap_claim(self, gap_id: int, claim_id: int) -> None:
        await self.execute("UPDATE gaps SET claim_id=? WHERE id=?", (claim_id, gap_id))

    async def open_gaps(self) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM gaps WHERE status='open' ORDER BY id")
