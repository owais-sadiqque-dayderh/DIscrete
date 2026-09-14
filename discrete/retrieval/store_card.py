"""Cached store description for query planners (planS3 §9)."""

from __future__ import annotations

import time

from ..config import T
from ..store import Store
from ..timeutil import iso

_cache: tuple[float, str, str] | None = None


async def store_card(store: Store) -> str:
    global _cache
    key = str(store.conn)
    if _cache and _cache[2] == key and time.time() - _cache[0] < T.STORE_CARD_TTL_S:
        return _cache[1]
    s = await store.stats()
    top = await store.q(
        "SELECT e.name, COUNT(DISTINCT m.ref_id) n FROM entities e JOIN mentions m ON m.entity_id=e.id "
        "WHERE e.merged_into IS NULL GROUP BY e.id ORDER BY n DESC LIMIT 8"
    )
    apps = await store.q(
        "SELECT app, COUNT(*) n FROM observations WHERE origin='user' AND app!='' GROUP BY app ORDER BY n DESC LIMIT 6"
    )
    confs = await store.q("SELECT ROUND(confidence, 2) c, COUNT(*) n FROM claims WHERE status='live' GROUP BY c ORDER BY n DESC LIMIT 1")
    top_share = (confs[0]["n"] / s["claims_live"]) if confs and s["claims_live"] else 0.0
    lines = [
        f"Coverage: {iso(s['first_observation'])} → {iso(s['last_observation'])}",
        f"Rows: {s['observations']} observations ({s['observations_agent']} self), {s['episodes']} episodes "
        f"{s['episodes_by_status']}, {s['claims_live']} live claims {s['claims_by_kind']}, "
        f"{s['entities']} entities (+{s['entity_candidates']} candidates), {s['open_gaps']} open gaps",
        "Top apps: " + ", ".join(f"{r['app']} ({r['n']})" for r in apps),
        "Top entities: " + ", ".join(f"{r['name']} ({r['n']})" for r in top),
        f"Health: index coverage {s['index_coverage']}, outbox {s['outbox_pending']}, "
        f"confidence top-value share {top_share:.0%}, lineage anomalies {await store.lineage_anomalies()}",
        "Counters: " + ", ".join(f"{k}={int(v)}" for k, v in sorted(s["metrics"].items())),
    ]
    text = "\n".join(lines)
    _cache = (time.time(), text, key)
    return text
