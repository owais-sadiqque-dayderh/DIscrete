"""CLI entrypoint.

Modules that load macOS frameworks (capture, bubble, the runtime) are imported
inside the commands that need them, so `discrete stats` or `query` never touch
Quartz/AppKit and start fast. Everything else is imported here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from . import textkit as tk
from .config import ConfigError, T, all_endpoints, data_dir, db_path
from .encryption import EncryptionError
from .evaluation import bench
from .evaluation.check_cli import register_check
from .evaluation.harness import compare, default_cases, run_suite, save_baseline
from .llm import flush_usage
from .perf import PERF
from .process.consolidate import run_consolidation
from .process.embed_worker import drain_outbox
from .process.pipeline import ProcessPipeline
from .process.trails import build_trails
from .retrieval.gaps import answer_and_learn, maybe_record_gap
from .retrieval.grounding import evaluate
from .retrieval.hybrid import retrieve
from .retrieval.store_card import store_card, usage_table
from .store import Store


def _print(*a) -> None:
    print(*a, flush=True)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@asynccontextmanager
async def _session():
    """The store for one command; model usage is recorded before it closes."""
    store = await Store.open()
    try:
        yield store
    finally:
        await flush_usage(store)
        await store.close()


async def cmd_run(args) -> int:
    from .app import run

    await run(bubble=not args.no_bubble)
    return 0


async def cmd_ask(args) -> int:
    """Headless Bubble: the exact hotkey pipeline, answer streamed to stdout."""
    from .bubble.engine import BubbleEngine, render_records
    from .bubble.overlay_client import Overlay

    overlay = Overlay(enabled=args.overlay)
    overlay.start()
    t0 = time.perf_counter()

    def progress(stage, detail) -> None:
        _err(f"[{time.perf_counter() - t0:5.2f}s] {stage} {detail}")
        overlay.status("Searching your memory" if stage == "searching" else f"Ranking {detail} moments")

    async with _session() as store:
        try:
            overlay.show(None)
            overlay.status("Understanding")
            engine = BubbleEngine(store)
            p = await engine.prepare(args.text, app=args.app, progress=progress)
            _err(f"[{time.perf_counter() - t0:5.2f}s] mode={p.mode} intent={p.result.plan.intent} "
                 f"hits={len(p.result.hits)} exhaustive={p.result.exhaustive} query={p.query[:60]!r}")
            if args.records:
                _err(render_records(p.result))
            if p.mode == "empty":
                print("(nothing on file about that)")
                overlay.message("Nothing on file about that", tone="muted")
            else:
                overlay.status(f"Writing from {len(p.result.hits)} sources")
                t_write = time.perf_counter()

                def emit(chunk: str) -> bool:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    return True

                await engine.write(p, emit)
                print()
                PERF.sample("bubble.total", (time.perf_counter() - t0) * 1000)
                overlay.done(f"Done · {time.perf_counter() - t0:.1f}s")
                _err(f"[{time.perf_counter() - t0:5.2f}s] done (retrieve {p.timings['retrieve_s']:.2f}s, first "
                     f"chunk +{p.timings.get('first_chunk_s', 0):.2f}s, write {time.perf_counter() - t_write:.2f}s)")
        finally:
            if overlay.enabled:
                await asyncio.sleep(1.4)
            overlay.close()
    return 0


async def cmd_overlay_demo(args) -> int:
    """Show the retrieval pill cycling through its stages (no model calls)."""
    from .bubble.overlay_client import Overlay

    ov = Overlay()
    ov.start()
    await asyncio.sleep(1.0)  # process boot; `run` keeps it warm so real triggers skip this
    ov.show(None)
    for s in ("Understanding", "Searching your memory", "Ranking 42 moments", "Writing from 6 sources"):
        ov.status(s)
        await asyncio.sleep(args.step)
    ov.done("Done · 3.1s")
    await asyncio.sleep(1.6)
    ov.message("Nothing on file about that", tone="muted")
    await asyncio.sleep(3.2)
    ov.close()
    return 0


async def cmd_doctor(args) -> int:
    from .capture import accessibility as ax
    from .capture.frames import FrameGrabber, is_blank
    from .capture.ocr import ocr_image

    for ep in all_endpoints():
        _print(ep.describe())
    _print(f"db: {db_path()}")
    ok = True
    ctx = ax.window_context()
    _print(f"front window: app={ctx.app!r} title={ctx.window_title!r} bounds={ctx.bounds}")
    if not ctx.app:
        ok = False
        _print("ISSUE: no front window detected")
    g = FrameGrabber()
    img = g.grab(ctx.bounds, max_width=1600, window_id=ctx.window_id)
    g.close()
    if img is None or is_blank(img):
        ok = False
        _print("ISSUE: captured frame is blank — grant Screen Recording to this terminal/app")
    else:
        _print(f"screen capture: OK ({img.width}x{img.height}); OCR sample: {tk.truncate(ocr_image(img), 80)!r}")
    ax_ok = ax.accessibility_granted()
    _print(f"accessibility: {'OK (AX text walk active)' if ax_ok else 'NOT granted — OCR-only screen text'}")
    txt = ""
    if ax_ok and ctx.pid:
        txt = ax.read_screen_text(ctx.pid, max_nodes=2000, timeout_ms=250, max_chars=4000)
        _print(f"AX text sample: {txt[:80]!r}")
    if args.models:
        from .evaluation.model_check import check_models, render

        _print("\nmodels (one real request each, same prompts, limits and .env settings as the pipeline):")
        results = await check_models(img, app=ctx.app, title=ctx.window_title, screen_text=txt)
        _print(render(results))
        ok = ok and all(r.ok for r in results)
    _print("OK" if ok else "issues found")
    return 0 if ok else 1


async def cmd_stats(_args) -> int:
    async with _session() as store:
        _print(await store_card(store))
        _print(await usage_table(store))
    return 0


async def cmd_query(args) -> int:
    async with _session() as store:
        result = await retrieve(store, args.text, field_text=args.field or "")
        d = evaluate(result)
        p = result.plan
        _print(f"intent={p.intent} mode={d.mode} strong={d.strong} weak={d.weak} exhaustive={result.exhaustive} "
               f"interval={p.interval and [tk.truncate(str(x), 20) for x in p.interval]}")
        if result.digest:
            _print(f"digest: {result.digest}\n---")
        for h in result.hits:
            flag = "*" if h.strong else " "
            extra = f" rr={h.rerank_score:.2f}" if h.rerank_score is not None else ""
            if h.ref_kind == "claim":
                extra += f" {h.claim_kind}/{h.modality} conf={h.confidence:.2f}"
            _print(f"{flag}[{h.ref_kind}:{h.ref_id}]{extra} {h.title}")
            _print("   " + tk.truncate(h.body, 300).replace("\n", "\n   "))
        if args.gap and d.mode in ("thin", "empty"):
            outcome = await maybe_record_gap(store, args.text, result=result)
            _print(f"gap outcome: {outcome.kind} {outcome.missed or outcome.gap_id or ''}")
    return 0


async def cmd_why(args) -> int:
    """Debug raw provenance without opening any model endpoint."""
    from .process.provenance import build_provenance
    from .retrieval.why import render_chain, resolve_why

    store = await Store.open()
    try:
        if args.backfill:
            _print(f"built {await build_provenance(store, backfill=True)} attention edges")
        _print(render_chain(await resolve_why(store, args.text), details=args.chain))
    finally:
        await store.close()
    return 0


async def cmd_process(args) -> int:
    async with _session() as store:
        recovered = []
        if args.recover:
            await store.reset_stuck_drafting()
            await store.retry_failed_drafts()
            await store.retry_failed_outbox()
            await store.fail_stale_enrichment()
            recovered = await store.recover_open_episodes(older_than_s=T.EPISODE_IDLE_CLOSE_S)
        pipe = ProcessPipeline(store)
        drafted = []
        while batch := await pipe.process_ready(limit=5):
            drafted += batch
        out = {
            "recovered": recovered,
            "drafted": [{"episode": r.episode_id, "added": r.added, "reinforced": r.reinforced,
                         "invalidated": r.invalidated, "rejected": r.rejected} for r in drafted],
            "embedded": await drain_outbox(store),
            "trail_links": await build_trails(store, limit=500),
        }
        if args.consolidate:
            out["consolidation"] = await run_consolidation(store)
        _print(json.dumps(out, indent=2, default=str))
    return 0


async def cmd_backfill_pages(args) -> int:
    from .capture.enrich import backfill_page_fields  # PIL/frames only; no macOS frameworks

    async with _session() as store:
        _print(json.dumps(await backfill_page_fields(store, limit=args.limit), indent=2))
    return 0


async def cmd_consolidate(_args) -> int:
    async with _session() as store:
        _print(json.dumps(await run_consolidation(store), indent=2, default=str))
    return 0


async def cmd_eval(args) -> int:
    path = Path(args.cases) if args.cases else default_cases()
    async with _session() as store:
        result = await run_suite(store, path, answers=args.answers)
        if not result.cases:
            _print(f"no cases in {path}")
            return 1
        if args.verbose:
            for c in result.cases:
                _print(f"{'✓' if c.correct else '✗'} [{c.case.category}] rank={c.rank} mode={c.mode} "
                       f"{c.latency_ms:.0f}ms  {c.case.question}")
        _print(json.dumps({"metrics": result.metrics(), "health": result.health, "snapshot": result.snapshot,
                           "stale": result.stale}, indent=2))
        if result.stale:
            _print("WARNING: case file snapshot differs from the store; scores are not comparable")
        if args.save_baseline:
            save_baseline(result, path)
            _print("baseline saved")
            return 0
        cmp = compare(result, path)
        if cmp["baseline"] is not None:
            _print(json.dumps({k: v for k, v in cmp.items() if k != "baseline"}, indent=2))
            if cmp["regressions"] and not result.stale:
                return 2
    return 1 if not result.stale and any(not case.correct for case in result.cases) else 0


async def cmd_bench(args) -> int:
    if args.compare and len(args.compare) == 2:
        _print(bench.compare(bench.load(args.compare[0]), bench.load(args.compare[1])))
        return 0
    async with _session() as store:
        try:
            window = await bench.parse_window(store, args.since, args.until)
        except ValueError as exc:
            _err(f"bad window: {exc}")
            return 2
        quality = None
        if args.eval:
            PERF.source = "eval"  # the eval's own model calls are not the pipeline's cost
            suite = await run_suite(store, Path(args.cases) if args.cases else default_cases())
            quality = suite.metrics() if suite.cases else None
            await flush_usage(store)
            PERF.source = "bench"
        sources = [s.strip() for s in args.source.split(",") if s.strip()] if args.source else None
        snap = await bench.collect(store, window, sources=sources, eval_metrics=quality, name=args.save or "")
    if args.json:
        _print(json.dumps(snap, indent=2, default=str))
    else:
        _print(bench.render(snap))
    if args.save:
        _print(f"\nsaved {bench.save(snap, args.save)}")
    if args.compare:
        _print("\n" + bench.compare(bench.load(args.compare[0]), snap))
    return 0


async def cmd_gaps(_args) -> int:
    async with _session() as store:
        for g in await store.open_gaps():
            _print(f"[{g['id']}] {g['question']}")
    return 0


async def cmd_answer(args) -> int:
    async with _session() as store:
        _print(json.dumps(await answer_and_learn(store, args.gap_id, args.text)))
    return 0


# ------------------------------------------------------------ memory history


def _since(text: str) -> float:
    """'90m', '6h', '2d' → a timestamp; '' or 'all' → 0."""
    text = (text or "").strip().lower()
    if not text or text == "all":
        return 0.0
    unit = {"m": 60, "h": 3600, "d": 86400}.get(text[-1])
    if unit is None or not text[:-1].replace(".", "", 1).isdigit():
        raise SystemExit(f"--since: use 90m, 6h or 2d, not {text!r}")
    return time.time() - float(text[:-1]) * unit


def _event_line(e) -> str:
    created = json.loads(e["created"])
    changes = json.loads(e["changes"])
    def _one(c: dict) -> str:
        if c.get("before") and "status" in c["before"]:
            return f"#{c['id']} {c['before']['status']}→{c['after'].get('status', '')}"
        added = len(c.get("evidence_added_rows") or c.get("evidence_added") or [])
        removed = len(c.get("evidence_removed_rows") or [])
        return f"#{c['id']} " + " ".join(p for p in (f"+{added} evidence" if added else "",
                                                      f"-{removed} evidence" if removed else "") if p) or f"#{c['id']}"

    touched = ", ".join([f"+#{c}" for c in created] + [_one(c) for c in changes])
    undone = f"  (reverted by #{e['undone_by']})" if e["undone_by"] else ""
    when = time.strftime("%b %d %H:%M", time.localtime(e["at"]))
    return f"#{e['id']:<5} {when}  {e['actor']:<11} {e['op']:<10} {touched}  — {e['reason']}{undone}"


async def cmd_history(args) -> int:
    async with _session() as store:
        versions = await store.lineage(args.claim_id)
        if not versions:
            _err(f"no claim {args.claim_id}")
            return 1
        _print("Versions:")
        for c in versions:
            mark = "→" if c["id"] == args.claim_id else " "
            _print(f" {mark} #{c['id']} v{c['version']} [{c['kind']} · {c['modality']} · {c['status']}"
                   f"{' · ' + c['flag'] if c['flag'] else ''}] {c['text']}")
        _print("\nChanges:")
        for e in await store.events_for_claim(args.claim_id):
            _print("  " + _event_line(e))
    return 0


async def cmd_changes(args) -> int:
    async with _session() as store:
        rows = await store.recent_events(since=_since(args.since), actor=args.actor, op=args.op, limit=args.n)
        for e in reversed(rows):
            _print(_event_line(e))
        if not rows:
            _print("(no changes)")
    return 0


async def cmd_revert(args) -> int:
    from .store import RevertConflict

    async with _session() as store:
        try:
            rid = await store.revert_event(args.event_id, force=args.force)
        except (KeyError, RevertConflict) as exc:
            _err(str(exc).strip("'\""))
            return 1
        _print(_event_line(await store.get_event(rid)))
    return 0


async def cmd_forget(args) -> int:
    async with _session() as store:
        try:
            eid = await store.forget_claim(args.claim_id, reason=args.reason)
        except KeyError as exc:
            _err(str(exc).strip("'\""))
            return 1
        _print(_event_line(await store.get_event(eid)))
        _print(f"undo with: discrete revert {eid}")
        await _refresh_export(store)
    return 0


async def _refresh_export(store) -> None:
    """A forgotten memory leaves the Markdown mirror at once, not at the next dream."""
    from .config import memory_export_dir
    from .export import export_markdown

    if getattr(store.conn, "_discrete_encrypted", False):
        return  # Plaintext export requires a fresh, explicit user action.
    out = memory_export_dir()
    if out is not None and (out / ".discrete-export").exists():
        await export_markdown(store, out, git=T.MEMORY_EXPORT_GIT)


async def cmd_correct(args) -> int:
    async with _session() as store:
        try:
            nid, eid = await store.correct_claim(args.claim_id, args.text, reason=args.reason)
        except KeyError as exc:
            _err(str(exc).strip("'\""))
            return 1
        _print(_event_line(await store.get_event(eid)))
    return 0


async def cmd_dream(args) -> int:
    from .process.dream import due, run_dream

    async with _session() as store:
        if not args.force and not await due(store):
            _print("not due (DISCRETE_DREAM_INTERVAL_MIN / DREAM_MIN_NEW_EPISODES); use --force")
            return 0
        _print(json.dumps(await run_dream(store), indent=2, default=str))
    return 0


async def cmd_profile(args) -> int:
    from .process.profile import render_profile, write_profile

    async with _session() as store:
        if args.rewrite:
            _print(json.dumps(await write_profile(store), default=str))
        profile = await store.get_profile()
        if args.json:
            _print(json.dumps(profile, indent=1, ensure_ascii=False))
        else:
            _print(render_profile(profile, with_ids=args.ids) or "(no profile yet: discrete dream --force)")
    return 0


async def cmd_graph(args) -> int:
    async with _session() as store:
        if not args.name:
            for r in await store.graph_overview(args.n):
                _print(f"{r['name']:<40} {r['kind']:<12} {r['memories']:>3} memories  {r['links']:>3} links")
            return 0
        ids = await store.entity_ids_named([args.name])
        if not ids:
            _err(f"no entity named {args.name!r} (discrete graph lists them)")
            return 1
        e = await store.get_entity(ids[0])
        _print(f"{e['name']} ({e['kind']})\n{e['summary'] or '(no page yet)'}")
        _print("\nRelated:")
        for n in await store.neighbors(e["id"], limit=args.n):
            _print(f"  {n['name']:<38} together in {int(n['weight'])} episode(s)")
        _print("\nMemories:")
        for c in await store.claims_for_entity(e["id"], limit=args.n):
            _print(f"  #{c['id']} [{c['kind']}] {c['text']}")
    return 0


def _export_dir(arg: str) -> Path:
    from .config import memory_export_dir

    return Path(arg).expanduser() if arg else (memory_export_dir() or data_dir() / "memory")


async def cmd_export(args) -> int:
    from .export import export_markdown

    out = _export_dir(args.dir)
    async with _session() as store:
        report = await export_markdown(store, out, git=args.git or T.MEMORY_EXPORT_GIT, allow_plaintext=args.allow_plaintext)
    _print(json.dumps(report, indent=2, default=str))
    _print(f"open {out}/MEMORY.md (or the folder as an Obsidian vault)")
    return 0


async def cmd_encryption(args) -> int:
    from dataclasses import asdict

    from .config import frames_dir
    from .encryption_migration import migrate_encryption, recover_migration

    path = db_path()
    if args.recover:
        report = await asyncio.to_thread(recover_migration, path)
    else:
        root = Path(args.frames_dir) if args.frames_dir else (
            path.parent / "frames" if args.db else frames_dir())
        operation = "rotate" if args.cmd == "rotate-key" else args.cmd
        report = await asyncio.to_thread(migrate_encryption, path, root, operation=operation, keep_backup=args.keep_backup)
    _print(json.dumps(asdict(report), indent=2))
    return 0


async def cmd_attention(args) -> int:
    async with _session() as store:
        start = _since(args.since) or 0.0
        for a in await store.attention(start, time.time() + 1, by=args.by, limit=args.n):
            _print(a.describe())
    return 0


async def cmd_tasks(args) -> int:
    async with _session() as store:
        await store.backfill_tasks()
        rows = await store.list_tasks(limit=args.n)
        if args.json:
            _print(json.dumps([dict(r) for r in rows], indent=2))
        else:
            from .timeutil import to_local

            for r in rows:
                _print(f"[task:{r['id']}] {r['title']} ({r['status']}, {r['n_episodes']} episodes) "
                       f"{to_local(r['first_seen']):%Y-%m-%d %H:%M}–{to_local(r['last_seen']):%Y-%m-%d %H:%M}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="discrete")
    p.add_argument("--db", help="store path (default: $DISCRETE_DATA_DIR/discrete.db)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    register_check(sub)

    for name, help_text in (("encrypt", "encrypt the store and frames offline using macOS Keychain"),
                            ("decrypt", "decrypt the store and frames to plaintext for debugging"),
                            ("rotate-key", "replace the data key and re-encrypt the store and frames")):
        pcipher = sub.add_parser(name, help=help_text)
        pcipher.add_argument("--frames-dir", default="", help="frame directory (with --db: defaults to <db parent>/frames)")
        pcipher.add_argument("--keep-backup", action="store_true", help="retain a timestamped copy of the ORIGINAL files")
        pcipher.add_argument("--recover", action="store_true", help="recover any interrupted encrypt/decrypt/key rotation")
        pcipher.set_defaults(func=cmd_encryption)

    pt = sub.add_parser("tasks", help="list research/work threads across drafted episodes")
    pt.add_argument("-n", type=int, default=20, help="maximum tasks to list")
    pt.add_argument("--json", action="store_true", help="print structured task rows")
    pt.set_defaults(func=cmd_tasks)

    prun = sub.add_parser("run", help="capture and process continuously, with Bubble on the hotkey")
    prun.add_argument("--no-bubble", action="store_true", help="capture only; do not arm the hotkey")
    prun.set_defaults(func=cmd_run)

    pk = sub.add_parser("ask", help="run the Bubble pipeline headlessly on some field text")
    pk.add_argument("text")
    pk.add_argument("--app", default="Terminal")
    pk.add_argument("--records", action="store_true", help="print the evidence pack sent to the writer")
    pk.add_argument("--overlay", action="store_true", help="also show the status pill")
    pk.set_defaults(func=cmd_ask)

    pd = sub.add_parser("overlay-demo", help="preview the retrieval animation")
    pd.add_argument("--step", type=float, default=1.3)
    pd.set_defaults(func=cmd_overlay_demo)

    pdoc = sub.add_parser("doctor", help="check permissions, capture and endpoints")
    pdoc.add_argument("--models", action="store_true",
                      help="also send one real request to each model (checks a new model in .env; costs < $0.01)")
    pdoc.set_defaults(func=cmd_doctor)
    sub.add_parser("stats", help="print the store card and model usage").set_defaults(func=cmd_stats)

    pp = sub.add_parser("process", help="draft ready episodes and drain the embedding outbox")
    pp.add_argument("--recover", action="store_true", help="also close open episodes left by a dead run")
    pp.add_argument("--consolidate", action="store_true")
    pp.set_defaults(func=cmd_process)

    sub.add_parser("consolidate", help="run offline consolidation now").set_defaults(func=cmd_consolidate)

    pb = sub.add_parser("backfill-pages", help="one-off: add page titles/listing/source to rows captured before C6")
    pb.add_argument("--limit", type=int, default=500)
    pb.set_defaults(func=cmd_backfill_pages)

    pq = sub.add_parser("query", help="show what retrieval returns for a question")
    pq.add_argument("text")
    pq.add_argument("--field", default="")
    pq.add_argument("--gap", action="store_true", help="verify and record a gap if unanswerable")
    pq.set_defaults(func=cmd_query)

    pw = sub.add_parser("why", help="model-free attention provenance diagnosis")
    pw.add_argument("text")
    pw.add_argument("--chain", action="store_true", help="include literal endpoint quotes")
    pw.add_argument("--backfill", action="store_true", help="rebuild stored attention edges without model calls")
    pw.set_defaults(func=cmd_why)

    pe = sub.add_parser("eval", help="score the golden retrieval cases")
    pe.add_argument("--cases", help="case file (default golden/cases.yaml)")
    pe.add_argument("--save-baseline", action="store_true")
    pe.add_argument("--verbose", action="store_true")
    pe.add_argument("--answers", action="store_true", help="also generate and independently check final answers")
    pe.set_defaults(func=cmd_eval)

    pbe = sub.add_parser("bench", help="cost, speed, volume, quality and storage over a time window")
    pbe.add_argument("--since", default="all", help="all | last-run | 90m / 6h / 2d | a local date/time")
    pbe.add_argument("--until", help="a local date/time (default: now)")
    pbe.add_argument("--source", help="comma list of run sources (default: all but eval/query/bench)")
    pbe.add_argument("--eval", action="store_true", help="also score the golden cases (quality)")
    pbe.add_argument("--cases", help="case file for --eval (default golden/cases.yaml)")
    pbe.add_argument("--save", metavar="NAME", help="write the snapshot to bench/NAME.json")
    pbe.add_argument("--compare", nargs="+", metavar="NAME",
                     help="one name: compare it with this report; two: compare two saved snapshots")
    pbe.add_argument("--json", action="store_true", help="print the snapshot as JSON")
    pbe.set_defaults(func=cmd_bench)

    sub.add_parser("gaps", help="list open knowledge gaps").set_defaults(func=cmd_gaps)

    pa = sub.add_parser("answer", help="answer a knowledge gap (stored as a stated fact)")
    pa.add_argument("gap_id", type=int)
    pa.add_argument("text")
    pa.set_defaults(func=cmd_answer)

    ph = sub.add_parser("history", help="every version of a memory and every change to it, with reasons")
    ph.add_argument("claim_id", type=int)
    ph.set_defaults(func=cmd_history)

    pc = sub.add_parser("changes", help="the memory change log: what changed, who changed it, why")
    pc.add_argument("--since", default="all", help="90m / 6h / 2d (default: all)")
    pc.add_argument("--actor", default="", help="draft | consolidate | dream | cleanup | user")
    pc.add_argument("--op", default="", help="add | reinforce | invalidate | merge | dream_add | forget | …")
    pc.add_argument("-n", type=int, default=50)
    pc.set_defaults(func=cmd_changes)

    pr = sub.add_parser("revert", help="undo one change from the change log (a revert can itself be reverted)")
    pr.add_argument("event_id", type=int)
    pr.add_argument("--force", action="store_true", help="even if later changes touched the same memories")
    pr.set_defaults(func=cmd_revert)

    pf = sub.add_parser("forget", help="forget a memory and never re-learn it")
    pf.add_argument("claim_id", type=int)
    pf.add_argument("--reason", default="")
    pf.set_defaults(func=cmd_forget)

    pco = sub.add_parser("correct", help="replace a memory's wording with yours (kept as a stated fact)")
    pco.add_argument("claim_id", type=int)
    pco.add_argument("text")
    pco.add_argument("--reason", default="")
    pco.set_defaults(func=cmd_correct)

    pdr = sub.add_parser("dream", help="find patterns across episodes, resolve conflicts, refresh the profile")
    pdr.add_argument("--force", action="store_true", help="run now even if not due")
    pdr.set_defaults(func=cmd_dream)

    ppr = sub.add_parser("profile", help="what DIscrete has learned about you, with sources")
    ppr.add_argument("--ids", action="store_true", help="show the memories behind each point")
    ppr.add_argument("--json", action="store_true")
    ppr.add_argument("--rewrite", action="store_true", help="rewrite it now from current memories")
    ppr.set_defaults(func=cmd_profile)

    pg = sub.add_parser("graph", help="entities and how they connect; with a name, one entity's neighbourhood")
    pg.add_argument("name", nargs="?", default="")
    pg.add_argument("-n", type=int, default=25)
    pg.set_defaults(func=cmd_graph)

    px = sub.add_parser("export", help="write memory as Markdown (git history, opens as an Obsidian vault)")
    px.add_argument("--dir", default="", help="default: DISCRETE_MEMORY_EXPORT_DIR or <data dir>/memory")
    px.add_argument("--git", action="store_true",
                    help="commit the export to git (keeps every version, forgotten ones too)")
    px.add_argument("--allow-plaintext", action="store_true", help="explicitly export plaintext from an encrypted store")
    px.set_defaults(func=cmd_export)

    pat = sub.add_parser("attention", help="where your time went: per site, app or page")
    pat.add_argument("--since", default="7d", help="90m / 6h / 2d / all (default 7d)")
    pat.add_argument("--by", choices=["site", "app", "page"], default="site")
    pat.add_argument("-n", type=int, default=20)
    pat.set_defaults(func=cmd_attention)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    PERF.source = "live" if args.cmd == "run" else args.cmd  # tags every measurement this process makes
    if args.db:
        os.environ["DISCRETE_DB"] = str(Path(args.db).expanduser())
    if args.cmd != "run":
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                            format="%(levelname)s %(name)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        code = asyncio.run(args.func(args))
    except KeyboardInterrupt:
        code = 130
    except ConfigError as exc:
        _err(f".env: {exc}")
        code = 2
    except EncryptionError as exc:
        _err(f"encryption: {exc}")
        code = 2
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
