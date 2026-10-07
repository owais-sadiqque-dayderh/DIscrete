"""`discrete bench`: one report of cost, speed, volume, quality and storage over a
time window, savable as a JSON snapshot and comparable against another one
(docs/C7.md). Every optimization step is judged by `bench --compare`.

Active time is the time covered by captures, with pauses up to IDLE_THRESHOLD_S
bridged (a longer pause is time away). Heartbeats capture at least every
HEARTBEAT_MAX_GAP_S while the screen is in use, so active reading counts too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROOT, T, all_endpoints, data_dir, db_path, frames_dir
from ..store import Store
from ..timeutil import from_local, iso, now
from .slos import EXCLUDED_SOURCES, evaluate_slos

# Below this much activity, per-hour rates are extrapolated from too little.
SHORT_WINDOW_H = 0.25

# (stage, label) in pipeline order; stages not listed are shown after these.
STAGES: list[tuple[str, str]] = [
    ("capture.local", "capture: local work before the queue"),
    ("capture.text", "capture: screen text read (AX/OCR)"),
    ("ingest.stored", "capture → row stored"),
    ("enrich.wait", "caption: waiting for a vision slot"),
    ("enrich.vision", "caption: vision call (with retries)"),
    ("enrich.total", "caption: stored → final caption"),
    ("segment.embed", "embed one capture"),
    ("segment.lag", "stored → placed in an episode"),
    ("embed.outbox", "outbox drain (one batch run)"),
    ("draft.verify", "draft: verify claims against evidence"),
    ("draft.episode", "draft one episode"),
    ("consolidate.run", "one consolidation run"),
    ("retrieve.plan", "search: query planning"),
    ("retrieve.search", "search: BM25 + dense + gates"),
    ("retrieve.rerank", "search: rerank"),
    ("retrieve.mmr", "search: diversity (MMR)"),
    ("retrieve.enumerate", "search: list-all (enumerate)"),
    ("retrieve.support", "search: verify records answer the question"),
    ("retrieve.total", "search: total"),
    ("bubble.field_read", "Bubble: read the text field"),
    ("bubble.retrieve", "Bubble: retrieval"),
    ("bubble.first_chunk", "Bubble: writer start → first words"),
    ("bubble.write", "Bubble: writing"),
    ("bubble.total", "Bubble: hotkey → done"),
]

# Flat numbers `--compare` lines up, and which direction is better.
HEADLINE: list[tuple[str, str | None]] = [
    ("cost.usd_per_active_hour", "lower"),
    ("cost.usd_total", None),
    ("activity.active_hours", None),
    ("captures.rows_per_active_hour", None),
    ("captures.vision_calls_per_active_hour", "lower"),
    ("caption.prompt_tokens_per_call", "lower"),
    ("caption.usd_per_call", "lower"),
    ("caption.ms_p50", "lower"),
    ("latency.enrich.total.p50", "lower"),
    ("latency.enrich.total.p90", "lower"),
    ("latency.segment.lag.p50", "lower"),
    ("latency.draft.episode.p50", "lower"),
    ("latency.retrieve.total.p50", "lower"),
    ("latency.bubble.retrieve.p50", "lower"),
    ("latency.bubble.first_chunk.p50", "lower"),
    ("latency.bubble.total.p50", "lower"),
    ("latency.bubble.total.p90", "lower"),
    ("quality.hit_at_k", "higher"),
    ("quality.mrr", "higher"),
    ("quality.refusal_accuracy", "higher"),
    ("storage.mb_per_active_hour", "lower"),
]


@dataclass
class Window:
    since: float | None
    until: float | None
    label: str


_REL = re.compile(r"^(\d+(?:\.\d+)?)\s*([mhd])$")


async def parse_window(store: Store, since: str | None, until: str | None) -> Window:
    """`since`: "all" (default), "last-run", a span like 90m / 6h / 2d, or a local
    date/time ("2026-09-30", "2026-09-30 14:00"). `until`: a date/time, or now."""
    def point(s: str) -> float:
        m = _REL.match(s.strip())
        if m:
            return now() - float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]
        return from_local(datetime.fromisoformat(s.strip()))

    lo = hi = None
    label = "all time"
    if since and since != "all":
        if since == "last-run":
            row = await store.q1("SELECT MAX(at) FROM perf_samples WHERE stage='runtime.start'")
            if not row or row[0] is None:
                raise ValueError("no `discrete run` recorded yet")
            lo, label = float(row[0]), f"last run (since {iso(row[0])})"
        else:
            lo = point(since)
            label = f"since {iso(lo)}"
    if until:
        hi = point(until)
        label += f" until {iso(hi)}"
    return Window(lo, hi, label)


def _pct(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "p50": round(float(np.percentile(a, 50)), 1),
            "p90": round(float(np.percentile(a, 90)), 1), "max": round(float(a.max()), 1)}


def active_seconds(spans: list[tuple[float, float]], bridge_s: float) -> float:
    total, cur = 0.0, None
    for start, end in spans:
        if cur and start - cur[1] <= bridge_s:
            cur[1] = max(cur[1], end)
            continue
        if cur:
            total += cur[1] - cur[0]
        cur = [start, end]
    if cur:
        total += cur[1] - cur[0]
    return total


def _dir_mb(path: Path) -> tuple[float, int]:
    size, n = 0, 0
    for f in path.glob("*"):
        if f.is_file():
            size += f.stat().st_size
            n += 1
    return size / 1e6, n


def _git() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"rev": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _config() -> dict[str, Any]:
    tunables = {k: v for k, v in vars(T).items() if k.isupper()}
    models: dict[str, Any] = {}
    for ep in all_endpoints():
        models[ep.name] = ep.model or ""
        for k, v in ep.settings().items():  # e.g. caption.reasoning_effort: a config difference like any other
            models[f"{ep.name}.{k}"] = v
    return {"models": models, "tunables": tunables}


async def collect(store: Store, window: Window, *, sources: list[str] | None = None,
                  eval_metrics: dict | None = None, name: str = "") -> dict[str, Any]:
    """The whole report as plain data (what `--save` writes)."""
    lo, hi = window.since, window.until
    present = await store.perf_sources()
    srcs = sources if sources is not None else [s for s in present if s not in EXCLUDED_SOURCES]
    calls = await store.perf_calls(lo, hi, srcs)
    samples = await store.perf_samples(lo, hi, srcs)
    spans = await store.activity_spans(lo, hi)
    active_h = active_seconds(spans, T.IDLE_THRESHOLD_S) / 3600
    per_h = (lambda x: round(x / active_h, 4) if active_h > 0 else None)

    # ---- model calls
    models: dict[str, dict[str, Any]] = {}
    by_comp: dict[str, list] = {}
    for c in calls:
        by_comp.setdefault(c["component"], []).append(c)
    total_cost = sum(c["cost"] or 0.0 for c in calls)
    for comp, rows in sorted(by_comp.items()):
        n = len(rows)
        cost = sum(r["cost"] or 0.0 for r in rows)
        ok = [r for r in rows if r["status"] == "ok"]
        firsts = [r["first_ms"] for r in ok if r["first_ms"] is not None]
        models[comp] = {
            "model": rows[-1]["model"],
            "calls": n,
            "errors": sum(r["status"] == "error" for r in rows),
            "cancelled": sum(r["status"] == "cancelled" for r in rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "prompt_tokens_per_call": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
            "completion_tokens_per_call": round(sum(r["completion_tokens"] for r in rows) / n, 1),
            "cost_usd": round(cost, 6),
            "usd_per_call": round(cost / n, 6),
            "cost_share": round(cost / total_cost, 3) if total_cost else None,
            "cost_missing_calls": sum(r["cost"] is None for r in rows),
            "latency_ms": _pct([r["ms"] for r in ok]) if ok else None,
            "first_token_ms": _pct(firsts) if firsts else None,
        }
    by_source: dict[str, float] = {}
    for c in calls:
        by_source[c["src"]] = round(by_source.get(c["src"], 0.0) + (c["cost"] or 0.0), 6)

    # ---- stages and counters
    timed: dict[str, list[float]] = {}
    counters: dict[str, int] = {}
    runtime_up_ms = 0.0
    for s in samples:
        if s["stage"] == "runtime.up":
            runtime_up_ms += s["ms"] or 0.0
        elif s["ms"] is None:
            counters[s["stage"]] = counters.get(s["stage"], 0) + (s["n"] or 0)
        else:
            timed.setdefault(s["stage"], []).append(s["ms"])
    latency = {stage: _pct(v) for stage, v in timed.items()}

    # ---- captures
    rows = await store.capture_counts(lo, hi)
    vision_calls = models.get("caption", {}).get("calls", 0)
    captures = {
        **rows,
        **{k: v for k, v in counters.items() if k.startswith(("capture.", "enrich."))},
        "rows_per_active_hour": per_h(rows.get("rows", 0)),
        "vision_calls_per_active_hour": per_h(vision_calls),
    }

    # ---- storage (current, not windowed)
    db = db_path()
    db_mb = sum(p.stat().st_size for p in (db, Path(f"{db}-wal")) if p.exists()) / 1e6
    frames_mb, n_frames = _dir_mb(frames_dir())
    all_h = active_seconds(await store.activity_spans(), T.IDLE_THRESHOLD_S) / 3600
    storage = {"db_mb": round(db_mb, 2), "frames_mb": round(frames_mb, 2), "frames": n_frames,
               "data_dir": str(data_dir()),
               "mb_per_active_hour": round((db_mb + frames_mb) / all_h, 2) if all_h > 0 else None}

    unknown_calls = sum(c["cost"] is None for c in calls)
    dropped_calls = counters.get("telemetry.dropped_calls", 0)
    dropped_samples = counters.get("telemetry.dropped_samples", 0)
    return {
        "meta": {"name": name, "created_at": iso(now()), "window": {"label": window.label, "since": iso(lo),
                 "until": iso(hi) or iso(now())}, "sources": srcs, "sources_present": present, "git": _git(),
                 **_config()},
        "activity": {"active_hours": round(active_h, 4), "captures_in_window": len(spans),
                     "daemon_hours": round(runtime_up_ms / 3.6e6, 3)},
        "cost": {"usd_total": round(total_cost, 6), "usd_per_active_hour": per_h(total_cost),
                 "unknown_calls": unknown_calls, "priced_calls": len(calls) - unknown_calls,
                 "dropped_calls": dropped_calls, "dropped_samples": dropped_samples,
                 "is_lower_bound": bool(unknown_calls or dropped_calls),
                 "usd_per_capture": round(total_cost / len(spans), 6) if spans else None, "by_source": by_source},
        "models": models,
        "latency_ms": latency,
        "slos": [asdict(result) for result in evaluate_slos(samples, pipeline_only=False)],
        "captures": captures,
        "quality": eval_metrics,
        "storage": storage,
    }


# ------------------------------------------------------------------ compare

def headline(snap: dict[str, Any]) -> dict[str, float]:
    cap = snap["models"].get("caption") or {}
    flat: dict[str, Any] = {
        "cost.usd_per_active_hour": snap["cost"]["usd_per_active_hour"],
        "cost.usd_total": snap["cost"]["usd_total"],
        "activity.active_hours": snap["activity"]["active_hours"],
        "captures.rows_per_active_hour": snap["captures"].get("rows_per_active_hour"),
        "captures.vision_calls_per_active_hour": snap["captures"].get("vision_calls_per_active_hour"),
        "caption.prompt_tokens_per_call": cap.get("prompt_tokens_per_call"),
        "caption.usd_per_call": cap.get("usd_per_call"),
        "caption.ms_p50": (cap.get("latency_ms") or {}).get("p50"),
        "storage.mb_per_active_hour": snap["storage"].get("mb_per_active_hour"),
    }
    for stage, d in snap["latency_ms"].items():
        flat[f"latency.{stage}.p50"] = d["p50"]
        flat[f"latency.{stage}.p90"] = d["p90"]
    for k, v in (snap.get("quality") or {}).items():
        if isinstance(v, int | float):
            flat[f"quality.{k}"] = v
    return {k: float(v) for k, v in flat.items() if isinstance(v, int | float)}


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    """A → B, headline numbers first, then any config that differs."""
    ha, hb = headline(a), headline(b)
    direction = dict(HEADLINE)
    keys = [k for k, _ in HEADLINE if k in ha or k in hb]
    keys += sorted(k for k in set(ha) | set(hb) if k not in direction)
    name = lambda s: s["meta"].get("name") or s["meta"]["window"]["label"]  # noqa: E731
    out = [f"A = {name(a)}  ({a['meta']['git'].get('rev', '?')}{'*' if a['meta']['git'].get('dirty') else ''})",
           f"B = {name(b)}  ({b['meta']['git'].get('rev', '?')}{'*' if b['meta']['git'].get('dirty') else ''})", ""]
    rows = [("metric", "A", "B", "change", "")]
    for k in keys:
        va, vb = ha.get(k), hb.get(k)
        change, mark = "", ""
        if va is not None and vb is not None:
            change = f"{(vb - va) / abs(va) * 100:+.0f}%" if va else ("0%" if vb == va else "new")
            better = direction.get(k)
            if better and vb != va:
                mark = "better" if (vb < va) == (better == "lower") else "WORSE"
        rows.append((k, _num(va), _num(vb), change, mark))
    out += _table(rows)
    diffs = _config_diff(a["meta"], b["meta"])
    out += ["", "Config differences:" if diffs else "Config: identical"] + [f"  {d}" for d in diffs]
    return "\n".join(out)


def _config_diff(a: dict, b: dict) -> list[str]:
    out = []
    for group in ("models", "tunables"):
        ga, gb = a.get(group, {}), b.get(group, {})
        for k in sorted(set(ga) | set(gb)):
            if ga.get(k) != gb.get(k):
                out.append(f"{k}: {ga.get(k)!r} → {gb.get(k)!r}")
    return out


# ------------------------------------------------------------------- render

def _num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.1f}"
    return f"{v:,.0f}"


def _table(rows: list[tuple]) -> list[str]:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return ["  " + str(r[0]).ljust(widths[0]) + "  " + "  ".join(str(r[i]).rjust(widths[i]) for i in range(1, len(r)))
            for r in rows]


def render(snap: dict[str, Any]) -> str:
    m, act, cost = snap["meta"], snap["activity"], snap["cost"]
    git = m["git"]
    out = [f"Bench: {m['window']['label']}   sources: {', '.join(m['sources']) or '(none)'}   "
           f"git {git.get('rev') or '?'}{' (uncommitted changes)' if git.get('dirty') else ''}",
           f"Active time {act['active_hours']:.2f} h over {act['captures_in_window']} captures"
           + (f"; daemon up {act['daemon_hours']:.2f} h" if act["daemon_hours"] else "")]
    if act["active_hours"] < SHORT_WINDOW_H:
        out.append(f"  note: under {SHORT_WINDOW_H * 60:.0f} min of activity; per-hour numbers are not meaningful yet")
    out.append("")

    out.append(f"COST  total ${cost['usd_total']:.4f}   per active hour {_usd(cost['usd_per_active_hour'])}   "
               f"per capture {_usd(cost['usd_per_capture'])}")
    if cost.get("is_lower_bound"):
        out.append(f"  lower bound: {cost.get('unknown_calls', 0)} calls have unknown prices; "
                   f"{cost.get('dropped_calls', 0)} call measurements dropped")
    if cost.get("dropped_samples"):
        out.append(f"  {cost['dropped_samples']} stage measurements dropped")
    other = {s: v for s, v in cost["by_source"].items() if s != "live"}
    if other:
        out.append("  by source: " + ", ".join(f"{s} ${v:.5f}" for s, v in cost["by_source"].items()))
    if snap["models"]:
        rows = [("component", "model", "calls", "err", "tok in/call", "tok out/call", "$/call", "cost $", "share",
                 "p50 ms", "p90 ms")]
        for comp, d in sorted(snap["models"].items(), key=lambda kv: -kv[1]["cost_usd"]):
            lat = d["latency_ms"] or {}
            rows.append((comp, d["model"].split("/")[-1], d["calls"], d["errors"] + d["cancelled"],
                         _num(d["prompt_tokens_per_call"]), _num(d["completion_tokens_per_call"]),
                         f"{d['usd_per_call']:.5f}", f"{d['cost_usd']:.4f}",
                         f"{d['cost_share'] * 100:.0f}%" if d["cost_share"] is not None else "—",
                         _num(lat.get("p50")), _num(lat.get("p90"))))
        out += _table(rows)
        missing = sum(d["cost_missing_calls"] for d in snap["models"].values())
        if missing:
            out.append(f"  note: the provider reported no cost for {missing} calls; totals undercount")
    else:
        out.append("  no model calls in this window")

    out += ["", "LATENCY (ms)"]
    lat = snap["latency_ms"]
    if lat:
        labels = dict(STAGES)
        order = [s for s, _ in STAGES if s in lat] + sorted(s for s in lat if s not in labels)
        rows = [("stage", "", "n", "p50", "p90", "max")]
        for s in order:
            d = lat[s]
            rows.append((s, labels.get(s, ""), d["n"], _num(d["p50"]), _num(d["p90"]), _num(d["max"])))
        out += _table(rows)
    else:
        out.append("  no timings in this window")

    c = snap["captures"]
    out += ["", "SLOs (p90; unmeasured stages are n/a)"]
    for result in snap.get("slos", []):
        out.append(f"  {result['stage']}: {result['status'].upper()}  "
                   f"p90 {_num(result['p90_ms'])} ms < {result['limit_ms']:g} ms (n={result['n']})")
        if result.get("reason"):
            out.append(f"    {result['reason']}")
    out += ["", "CAPTURES"]
    if c.get("rows"):
        out.append(f"  rows {c['rows']} ({_num(c['rows_per_active_hour'])}/active h): screen {c['screen']}, "
                   f"heartbeat {c['heartbeat']}, degraded {c['degraded']}; later duplicates folded in {c['collapsed']}")
        out.append(f"  captions: done {c['captioned']}, failed {c['caption_failed']}, pending {c['caption_pending']}, "
                   f"discarded as useless {c['discarded']}; vision calls "
                   f"{_num(c['vision_calls_per_active_hour'])}/active h")
        out.append(f"  screen text: AX {c['text_ax']}, OCR {c['text_ocr']}; "
                   f"avg {c['screen_text_chars'] // max(1, c['rows']):,} chars/row; frames saved {c['frames_saved']}")
        counts = {k: v for k, v in c.items() if k.startswith(("capture.", "enrich.")) and v}
        if counts:
            out.append("  observer/enricher counters: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    else:
        out.append("  no captures in this window")

    q = snap.get("quality")
    out += ["", "QUALITY (golden cases)"]
    if q:
        out.append(f"  n={q.get('n')}  hit@k {_num(q.get('hit_at_k'))}  MRR {_num(q.get('mrr'))}  "
                   f"refusal accuracy {_num(q.get('refusal_accuracy'))}  "
                   f"grounded {_num(q.get('grounded_rate_answerable'))}  p50 {_num(q.get('latency_ms_p50'))} ms")
        if q.get("by_category"):
            out.append("  by category: " + ", ".join(f"{k} {v:.2f}" for k, v in q["by_category"].items()))
    else:
        out.append("  not run (add --eval)")

    s = snap["storage"]
    out += ["", f"STORAGE  db {s['db_mb']:.1f} MB, frames {s['frames_mb']:.1f} MB ({s['frames']} files), "
                f"{_num(s['mb_per_active_hour'])} MB per active hour (all time)"]
    return "\n".join(out)


def _usd(v: float | None) -> str:
    return "—" if v is None else f"${v:.4f}"


# -------------------------------------------------------------- snapshots

def bench_dir() -> Path:
    return Path(os.environ.get("DISCRETE_BENCH_DIR") or ROOT / "bench")


def snapshot_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".json" or p.parent != Path("."):
        return p
    return bench_dir() / f"{name}.json"


def save(snap: dict[str, Any], name: str) -> Path:
    path = snapshot_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap["meta"]["name"] = path.stem
    path.write_text(json.dumps(snap, indent=2, default=str) + "\n")
    return path


def load(name: str) -> dict[str, Any]:
    return json.loads(snapshot_path(name).read_text())
