"""Cached store description for query planners and `discrete stats` (planS3 §9)."""

from __future__ import annotations

import time

from ..config import T
from ..store import Store
from ..store.health import USAGE_PREFIX
from ..timeutil import iso

_cache: tuple[float, str, int] | None = None


async def store_card(store: Store) -> str:
    global _cache
    key = id(store)
    if _cache and _cache[2] == key and time.time() - _cache[0] < T.STORE_CARD_TTL_S:
        return _cache[1]
    s = await store.stats()
    top = await store.top_entities(8)
    apps = await store.top_apps(6)
    confs = await store.confidence_histogram(decimals=2)
    top_share = (confs[0][1] / s["claims_live"]) if confs and s["claims_live"] else 0.0
    counters = {k: v for k, v in s["metrics"].items() if not k.startswith(USAGE_PREFIX)}
    lines = [
        f"Coverage: {iso(s['first_observation'])} → {iso(s['last_observation'])}",
        f"Rows: {s['observations']} observations ({s['observations_agent']} self/discarded), {s['episodes']} episodes "
        f"{s['episodes_by_status']}, {s['tasks']} tasks, {s['claims_live']} live claims {s['claims_by_kind']}, "
        f"{s['entities']} entities (+{s['entity_candidates']} candidates), {s['open_gaps']} open gaps",
        "Top apps: " + ", ".join(f"{r['app']} ({r['n']})" for r in apps),
        "Top entities: " + ", ".join(f"{r['name']} ({r['n']})" for r in top),
        f"Health: index coverage {s['index_coverage']}, outbox {s['outbox_pending']}, "
        f"task coverage {s['task_episodes']}/{s['episodes_by_status'].get('closed', 0)} closed episodes, "
        f"captions pending {s['enrich_pending']}, confidence top-value share {top_share:.0%}, "
        f"lineage anomalies {await store.lineage_anomalies()}, failed drafts {s['drafts_failed_current']}, "
        f"caption text >4000 chars (at risk) {s['caption_text_at_risk']}; silent-loss counters "
        + ", ".join(f"{k}={v}" for k, v in s["silent_loss"].items()),
        "Counters: " + ", ".join(f"{k}={int(v)}" for k, v in sorted(counters.items())),
    ]
    text = "\n".join(lines)
    _cache = (time.time(), text, key)
    return text


async def usage_table(store: Store) -> str:
    """Model calls, tokens and provider-reported cost per component, all time."""
    usage = await store.usage_by_component()
    if not usage:
        return "Model use: nothing recorded yet"
    rows = [("component", "calls", "prompt tok", "completion tok", "cost $")]
    for comp, u in sorted(usage.items(), key=lambda kv: -(kv[1]["prompt_tokens"] + kv[1]["completion_tokens"])):
        rows.append((comp, f"{u['calls']:,.0f}", f"{u['prompt_tokens']:,.0f}", f"{u['completion_tokens']:,.0f}",
                     f"{u['cost']:.4f}"))
    total = {f: sum(u[f] for u in usage.values()) for f in ("calls", "prompt_tokens", "completion_tokens", "cost")}
    rows.append(("total", f"{total['calls']:,.0f}", f"{total['prompt_tokens']:,.0f}",
                 f"{total['completion_tokens']:,.0f}", f"{total['cost']:.4f}"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["Model use (all time; cost as reported by the provider):"]
    for r in rows:
        out.append("  " + r[0].ljust(widths[0]) + "  " + "  ".join(r[i].rjust(widths[i]) for i in range(1, len(r))))
    return "\n".join(out)
