"""Store-wide counts, health figures and model usage."""

from __future__ import annotations

from typing import Any

from .base import StoreBase
from .index import _LIVE

USAGE_PREFIX = "llm."
_USAGE_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cost")


class HealthMixin(StoreBase):
    async def stats(self) -> dict[str, Any]:
        def _stats():
            c = self.rconn
            one = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
            rng = c.execute("SELECT MIN(started_at), MAX(started_at) FROM observations").fetchone()
            coverage = {}
            for kind, (table, live) in _LIVE.items():
                n = one(f"SELECT COUNT(*) FROM {table} WHERE {live}")
                have = one(
                    f"SELECT COUNT(*) FROM {table} t WHERE {live} AND EXISTS "
                    f"(SELECT 1 FROM embeddings e WHERE e.ref_kind='{kind}' AND e.ref_id=t.id)"
                )
                coverage[kind] = 1.0 if n == 0 else round(have / n, 3)
            return {
                "observations": one("SELECT COUNT(*) FROM observations"),
                "observations_agent": one("SELECT COUNT(*) FROM observations WHERE origin!='user'"),
                "duplicates_collapsed": one("SELECT COALESCE(SUM(dup_count-1),0) FROM observations"),
                "episodes": one("SELECT COUNT(*) FROM episodes"),
                "episodes_by_status": dict(c.execute("SELECT status, COUNT(*) FROM episodes GROUP BY status").fetchall()),
                "claims_live": one("SELECT COUNT(*) FROM claims WHERE status='live'"),
                "claims_by_kind": dict(c.execute("SELECT kind, COUNT(*) FROM claims WHERE status='live' GROUP BY kind").fetchall()),
                "entities": one("SELECT COUNT(*) FROM entities WHERE merged_into IS NULL"),
                "entity_candidates": one("SELECT COUNT(*) FROM entity_candidates"),
                "open_gaps": one("SELECT COUNT(*) FROM gaps WHERE status='open'"),
                "embeddings": one("SELECT COUNT(*) FROM embeddings"),
                "outbox_pending": one("SELECT COUNT(*) FROM index_outbox"),
                "enrich_pending": one("SELECT COUNT(*) FROM observations WHERE enrich_state='pending'"),
                "index_coverage": coverage,
                "first_observation": rng[0],
                "last_observation": rng[1],
                "metrics": dict(c.execute("SELECT name, value FROM metrics").fetchall()),
            }

        return await self._read(_stats)

    async def top_apps(self, limit: int = 6) -> list:
        return await self.q("SELECT app, COUNT(*) n FROM observations WHERE origin='user' AND app!='' "
                            "GROUP BY app ORDER BY n DESC LIMIT ?", (limit,))

    async def snapshot_id(self) -> str:
        row = await self.q1(
            "SELECT (SELECT COUNT(*) FROM observations), (SELECT COALESCE(MAX(id),0) FROM observations), "
            "(SELECT COUNT(*) FROM claims WHERE status='live')"
        )
        n, max_id, claims = tuple(row or (0, 0, 0))
        return f"obs{n}-max{max_id}-claims{claims}"

    # ------------------------------------------------------------------ usage

    async def record_usage(self, usage: dict[str, dict[str, float]]) -> None:
        """Add per-component model usage ({component: {calls, prompt_tokens,
        completion_tokens, cost}}) to the metrics table as `llm.<component>.<field>`."""
        items = [(f"{USAGE_PREFIX}{comp}.{f}", float(v)) for comp, fields in usage.items()
                 for f, v in fields.items() if v]
        if not items:
            return

        def _run():
            with self._tx() as conn:
                for name, delta in items:
                    self._metric(conn, name, delta)

        await self._call(_run)

    async def usage_by_component(self) -> dict[str, dict[str, float]]:
        """Token and call counts are whole numbers; `cost` is USD."""
        rows = await self.q("SELECT name, value FROM metrics WHERE name LIKE ?", (USAGE_PREFIX + "%",))
        out: dict[str, dict[str, float]] = {}
        for name, value in rows:
            comp, _, fld = name[len(USAGE_PREFIX):].rpartition(".")
            if comp and fld in _USAGE_FIELDS:
                out.setdefault(comp, dict.fromkeys(_USAGE_FIELDS, 0))[fld] = value if fld == "cost" else int(value)
        return out
