"""Step-0 measurement log (docs/C7.md): model calls, stage samples, and the
capture-activity reads `discrete bench` builds its report from."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from ..perf import Call, Sample
from ..timeutil import now
from .base import StoreBase
from .health import HEALTH_EVENT_PREFIX


def _window(since: float | None, until: float | None, sources: Sequence[str] | None) -> tuple[str, list]:
    where, args = ["at >= ?", "at < ?"], [since or 0.0, until or float("inf")]
    if sources:
        where.append(f"src IN ({','.join('?' * len(sources))})")
        args += list(sources)
    return " AND ".join(where), args


class PerfMixin(StoreBase):
    async def record_perf(self, calls: Sequence[Call], samples: Sequence[Sample]) -> None:
        def _run():
            with self._tx() as conn:
                conn.executemany(
                    "INSERT INTO perf_calls(at, src, component, model, status, ms, first_ms, prompt_tokens, "
                    "completion_tokens, cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(c.at, c.source, c.component, c.model, c.status, c.ms, c.first_ms, c.prompt_tokens,
                      c.completion_tokens, c.cost) for c in calls])
                conn.executemany(
                    "INSERT INTO perf_samples(at, src, stage, ms, n, ref_id) VALUES (?,?,?,?,?,?)",
                    [(s.at, s.source, s.stage, s.ms, s.n, s.ref_id) for s in samples])
                for sample in samples:
                    if sample.ms is None and sample.stage.startswith(HEALTH_EVENT_PREFIX) and sample.n:
                        self._metric(conn, sample.stage.removeprefix(HEALTH_EVENT_PREFIX), sample.n)

        await self._call(_run)

    async def perf_calls(self, since: float | None = None, until: float | None = None,
                         sources: Sequence[str] | None = None) -> list[sqlite3.Row]:
        where, args = _window(since, until, sources)
        return await self.q(f"SELECT * FROM perf_calls WHERE {where} ORDER BY at", args)

    async def perf_samples(self, since: float | None = None, until: float | None = None,
                           sources: Sequence[str] | None = None) -> list[sqlite3.Row]:
        where, args = _window(since, until, sources)
        return await self.q(f"SELECT * FROM perf_samples WHERE {where} ORDER BY at", args)

    async def perf_sources(self) -> list[str]:
        rows = await self.q("SELECT src FROM perf_calls UNION SELECT src FROM perf_samples")
        return sorted(r[0] for r in rows)

    async def prune_perf(self, *, ttl_days: float) -> int:
        if ttl_days <= 0:
            return 0
        cutoff = now() - ttl_days * 86400

        def _run():
            with self._tx() as conn:
                a = conn.execute("DELETE FROM perf_calls WHERE at < ?", (cutoff,)).rowcount
                b = conn.execute("DELETE FROM perf_samples WHERE at < ?", (cutoff,)).rowcount
            return a + b

        return await self._call(_run)

    # ------------------------------------------------------ capture activity

    async def activity_spans(self, since: float | None = None, until: float | None = None) -> list[tuple[float, float]]:
        """(started_at, ended_at) of every capture row in the window, by start time."""
        rows = await self.q("SELECT started_at, ended_at FROM observations WHERE started_at >= ? AND started_at < ? "
                            "ORDER BY started_at", (since or 0.0, until or float("inf")))
        return [(r[0], max(r[0], r[1])) for r in rows]

    async def capture_counts(self, since: float | None = None, until: float | None = None) -> dict[str, int]:
        """What happened to the captures in the window: rows by kind and outcome."""
        row = await self.q1(
            """SELECT COUNT(*) rows,
                      COALESCE(SUM(dup_count - 1), 0) collapsed,
                      SUM(source='screen') screen, SUM(source='heartbeat') heartbeat, SUM(source='degraded') degraded,
                      SUM(origin='discarded') discarded, SUM(origin NOT IN ('user','discarded')) self_output,
                      SUM(enrich_state='done') captioned, SUM(enrich_state='failed') caption_failed,
                      SUM(enrich_state='pending') caption_pending,
                      SUM(text_source='ax') text_ax, SUM(text_source LIKE '%ocr%') text_ocr,
                      SUM(frame_path != '') frames_saved,
                      COALESCE(SUM(LENGTH(screen_text)), 0) screen_text_chars
               FROM observations WHERE started_at >= ? AND started_at < ?""",
            (since or 0.0, until or float("inf")))
        return {k: int(row[k] or 0) for k in row.keys()} if row else {}
