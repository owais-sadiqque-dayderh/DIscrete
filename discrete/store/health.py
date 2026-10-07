"""Store-wide counts, health figures and model usage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .base import StoreBase
from .index import _LIVE

USAGE_PREFIX = "llm."
_USAGE_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cost", "unknown_cost_calls")
HEALTH_COUNTERS = (
    "draft_window_dropped", "draft_chunks", "draft_observations", "draft_text_truncated",
    "draft_skipped_unconfigured", "text_truncated.4000", "capture_screen_text_cap_hits", "capture_ax_depth_limit",
    "capture_screen_text_truncated", "capture_ax_text_truncated", "capture_ax_deadline", "capture_ax_node_limit",
    "capture_ax_empty.unavailable", "capture_ax_empty.no_pid", "capture_ax_empty.untrusted",
    "capture_ax_empty.no_window", "capture_ax_empty.viewport_mismatch", "capture_ax_empty.deadline",
    "capture_ax_empty.node_limit", "capture_ax_empty.no_text", "capture_ax_empty.error",
    "capture_ax_empty.stale", "capture_redaction_frame_drops",
    "draft_failed", "draft_abandoned", "draft_cancelled", "trail_failed", "trail_abandoned", "trail_cancelled",
    "llm_provider_retry", "llm_provider_giveup",
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

    async def snapshot_manifest(self) -> dict:
        def read():
            self.rconn.execute("BEGIN")
            try:
                result = content_manifest(self.rconn)
                self.rconn.execute("COMMIT")
                return result
            except BaseException:
                self.rconn.execute("ROLLBACK")
                raise
        return {**await self._read(read), "config": config_manifest()}

    async def snapshot_id(self) -> str:
        manifest = await self.snapshot_manifest()
        return "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

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


def config_manifest() -> dict:
    """Resolved nonsecret settings; arbitrary provider bodies and URLs are hashed."""
    from ..config import T, all_endpoints

    models = {}
    for ep in all_endpoints():
        models[ep.name] = {"model": ep.model, "extra_tokens": ep.extra_tokens,
                           "reasoning_effort": ep.reasoning_effort, "temperature": ep.temperature,
                           "no_temperature": ep.no_temperature, "json_mode": ep.json_mode,
                           "timeout_s": ep.timeout_s,
                           "extra_body_sha256": hashlib.sha256(ep.extra_body.encode()).hexdigest(),
                           "base_url_sha256": hashlib.sha256((ep.base_url or "").encode()).hexdigest()}
    return {"tunables": {name: value for name, value in vars(T).items() if name.isupper()}, "models": models}


def content_manifest(conn) -> dict:
    """Hash retrieval inputs from a consistent snapshot, including vectors and policy.

    Measurements and query diagnostics are excluded so measuring a run does not
    change its own identity. No source text or credential appears in the manifest.
    """
    digest = hashlib.sha256()
    excluded = {"metrics", "perf_calls", "perf_samples", "gaps", "retrieval_misses"}
    tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    for name, schema in tables:
        if name in excluded or name.startswith("sqlite_") or "_fts" in name:
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")]
        order = ','.join('"' + c.replace('"', '""') + '"' for c in columns)
        digest.update(json.dumps([name, schema]).encode())
        for row in conn.execute(f"SELECT * FROM {quoted} ORDER BY {order}"):
            if name == "meta" and (str(row[0]).startswith(("telemetry:", "last_", "draft_coverage:"))):
                continue
            values = [value.hex() if isinstance(value, bytes) else value for value in row]
            payload = json.dumps(values, ensure_ascii=False).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return {"format": 1, "schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
            "content_sha256": digest.hexdigest()}
