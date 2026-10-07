"""Store-wide counts, health figures and model usage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .base import StoreBase
from .index import _LIVE

USAGE_PREFIX = "llm."
_USAGE_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cost")
HEALTH_COUNTERS = (
    "draft_window_dropped", "draft_chunks", "draft_observations", "draft_text_truncated",
    "draft_skipped_unconfigured", "text_truncated.4000", "capture_screen_text_cap_hits", "capture_ax_depth_limit",
    "capture_screen_text_truncated", "capture_ax_text_truncated", "capture_ax_deadline", "capture_ax_node_limit",
    "capture_ax_empty.unavailable", "capture_ax_empty.no_pid", "capture_ax_empty.untrusted",
    "capture_ax_empty.no_window", "capture_ax_empty.viewport_mismatch", "capture_ax_empty.deadline",
    "capture_ax_empty.node_limit", "capture_ax_empty.no_text", "capture_ax_empty.error",
    "capture_ax_empty.stale", "capture_redaction_frame_drops",
    "draft_failed", "draft_abandoned", "draft_cancelled", "trail_failed", "trail_abandoned", "trail_cancelled",
)
# Event samples with these names also enter the durable metrics table on flush.
HEALTH_EVENT_PREFIX = "health."


@dataclass
class DraftCoverage:
    input_hash: str
    chunks: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> DraftCoverage:
        raw = json.loads(text)
        if (not isinstance(raw, dict) or not isinstance(raw.get("input_hash"), str)
                or not isinstance(raw.get("chunks"), dict)
                or any(not isinstance(chunk, list) or any(type(oid) is not int for oid in chunk)
                       for chunk in raw["chunks"].values())):
            raise ValueError("invalid delivered-chunk coverage")
        return cls(raw["input_hash"], raw["chunks"])


def health_counters(metrics: dict[str, float]) -> dict[str, int]:
    names = list(HEALTH_COUNTERS) + sorted(k for k in metrics if k.startswith("text_truncated.")
                                          and k not in HEALTH_COUNTERS)
    return {name: int(metrics.get(name, 0)) for name in names}


class HealthMixin(StoreBase):
    async def stats(self) -> dict[str, Any]:
        def _stats():
            c = self.rconn
            one = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
            rng = c.execute("SELECT MIN(started_at), MAX(started_at) FROM observations").fetchone()
            metrics = dict(c.execute("SELECT name, value FROM metrics").fetchall())
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
                "tasks": one("SELECT COUNT(*) FROM tasks WHERE status='active'"),
                "task_episodes": one("SELECT COUNT(*) FROM episode_tasks et JOIN episodes e ON e.id=et.episode_id "
                                     "JOIN tasks t ON t.id=et.task_id WHERE e.status='closed' AND t.status='active'"),
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
                "metrics": metrics,
                "silent_loss": health_counters(metrics),
                "drafts_failed_current": one("SELECT COUNT(*) FROM episodes WHERE status='failed'"),
                # Risk is visible even for legacy rows that lack prompt-cut telemetry.
                "caption_text_at_risk": one("SELECT COUNT(*) FROM observations WHERE origin='user' "
                                            "AND LENGTH(screen_text)>4000"),
            }

        return await self._read(_stats)

    async def record_draft_coverage(self, episode_id: int, input_hash: str, chunk_index: int,
                                    observation_ids: list[int]) -> None:
        """Retain delivered chunk ids beyond closure (which deletes retry checkpoints).

        A new input hash resets coverage; retries on the same input do not count twice.
        Use meta so this works without a schema migration in another agent's files.
        """
        def _run():
            with self._tx() as conn:
                key = f"draft_coverage:{episode_id}"
                row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                record = DraftCoverage.from_json(row[0]) if row else DraftCoverage(input_hash)
                if record.input_hash != input_hash:
                    record = DraftCoverage(input_hash)
                index = str(chunk_index)
                if index not in record.chunks:
                    self._metric(conn, "draft_chunks")
                    self._metric(conn, "draft_observations", len(observation_ids))
                record.chunks[index] = observation_ids
                conn.execute("INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) "
                             "DO UPDATE SET value=excluded.value", (key, json.dumps(asdict(record))))
        await self._call(_run)

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
