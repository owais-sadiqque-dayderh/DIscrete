"""CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from . import textkit as tk
from .config import all_endpoints, db_path


def _print(*a) -> None:
    print(*a, flush=True)


async def _open():
    from .store import Store

    return await Store.open()


async def cmd_run(args) -> int:
    from .app import run

    await run(bubble=not args.no_bubble)
    return 0


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def cmd_ask(args) -> int:
    """Headless Bubble: the exact hotkey pipeline, answer streamed to stdout."""
    import time

    from .bubble.engine import BubbleEngine, render_records
    from .bubble.overlay_client import Overlay

    overlay = Overlay(enabled=args.overlay)
    overlay.start()
    store = await _open()
    t0 = time.perf_counter()

    def progress(stage, detail) -> None:
        _err(f"[{time.perf_counter() - t0:5.2f}s] {stage} {detail}")
        overlay.status("Searching your memory" if stage == "searching" else f"Ranking {detail} moments")

    try:
        overlay.show(None)
        overlay.status("Understanding")
        engine = BubbleEngine(store)
        p = await engine.prepare(args.text, app=args.app, progress=progress)
        _err(f"[{time.perf_counter() - t0:5.2f}s] mode={p.mode} intent={p.result.plan.intent} hits={len(p.result.hits)} "
             f"exhaustive={p.result.exhaustive} query={p.query[:60]!r}")
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
            overlay.done(f"Done · {time.perf_counter() - t0:.1f}s")
            _err(f"[{time.perf_counter() - t0:5.2f}s] done (retrieve {p.timings['retrieve_s']:.2f}s, first chunk "
                 f"+{p.timings.get('first_chunk_s', 0):.2f}s, write {time.perf_counter() - t_write:.2f}s)")
    finally:
        if overlay.enabled:
            await asyncio.sleep(1.4)
        overlay.close()
        await store.close()
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


async def cmd_doctor(_args) -> int:
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
    img = g.grab(ctx.bounds, max_width=1600)
    g.close()
    if is_blank(img):
        ok = False
        _print("ISSUE: captured frame is blank — grant Screen Recording to this terminal/app")
    else:
        _print(f"screen capture: OK ({img.width}x{img.height}); OCR sample: {tk.truncate(ocr_image(img), 80)!r}")
    ax_ok = ax.accessibility_granted()
    _print(f"accessibility: {'OK (AX text walk active)' if ax_ok else 'NOT granted — OCR-only screen text'}")
    if ax_ok and ctx.pid:
        txt = ax.read_screen_text(ctx.pid, max_nodes=2000, timeout_ms=250, max_chars=200)
        _print(f"AX text sample: {txt[:80]!r}")
    _print("OK" if ok else "issues found")
    return 0 if ok else 1


async def cmd_stats(_args) -> int:
    from .retrieval.store_card import store_card

    store = await _open()
    try:
        _print(await store_card(store))
    finally:
        await store.close()
    return 0


async def cmd_query(args) -> int:
    from .retrieval.gaps import maybe_record_gap
    from .retrieval.grounding import evaluate
    from .retrieval.search import retrieve

    store = await _open()
    try:
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
    finally:
        await store.close()
    return 0


async def cmd_process(args) -> int:
    from .config import T
    from .process.consolidate import run_consolidation
    from .process.embed_worker import drain_outbox
    from .process.pipeline import ProcessPipeline

    store = await _open()
    try:
        recovered = []
        if args.recover:
            await store.reset_stuck_drafting()
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
        }
        if args.consolidate:
            out["consolidation"] = await run_consolidation(store)
        _print(json.dumps(out, indent=2, default=str))
    finally:
        await store.close()
    return 0


async def cmd_consolidate(_args) -> int:
    from .process.consolidate import run_consolidation

    store = await _open()
    try:
        _print(json.dumps(await run_consolidation(store), indent=2, default=str))
    finally:
        await store.close()
    return 0


async def cmd_eval(args) -> int:
    from .evaluation.harness import compare, default_cases, run_suite, save_baseline

    path = Path(args.cases) if args.cases else default_cases()
    store = await _open()
    try:
        result = await run_suite(store, path)
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
    finally:
        await store.close()
    return 0


async def cmd_gaps(_args) -> int:
    store = await _open()
    try:
        for g in await store.open_gaps():
            _print(f"[{g['id']}] {g['question']}")
    finally:
        await store.close()
    return 0


async def cmd_answer(args) -> int:
    from .retrieval.gaps import answer_and_learn

    store = await _open()
    try:
        _print(json.dumps(await answer_and_learn(store, args.gap_id, args.text)))
    finally:
        await store.close()
    return 0


async def cmd_replay_workaa(args) -> int:
    """Replay the WorkAA prototype's observations through Stage 2 (no capture, no VLM).

    WorkAA's `app` was model-written (insights.md §7.9): a domain-looking value
    becomes `site`, and `app` becomes "Browser". `dense_text` stands in for
    screen_text, and interaction is approximated from event_kind.
    """
    import sqlite3
    from datetime import datetime, timezone

    from .process.consolidate import run_consolidation
    from .process.embed_worker import drain_outbox
    from .process.episodes import EpisodeTracker
    from .process.pipeline import ProcessPipeline
    from .store import ObservationDraft, Store

    target = Path(os.environ["DISCRETE_DB"])
    if target.exists():
        if not args.fresh:
            _print(f"{target} exists; pass --fresh to rebuild it")
            return 1
        for suffix in ("", "-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{Path(args.source).expanduser()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM observations WHERE origin='user' ORDER BY id").fetchall()[: args.limit or None]
    store = await Store.open(str(target))
    tracker = EpisodeTracker(store)
    new = 0
    try:
        for r in rows:
            ts = datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc).timestamp()
            raw_app = r["app"] or ""
            url = r["url"] or ""
            domainish = bool(tk._DOMAINISH.match(raw_app.lower()))
            site = tk.site_from_url(url if url.startswith("http") else f"https://{url}") if url else ""
            if domainish and not site:
                site = tk.site_from_url(f"https://{raw_app}")
            app = "Browser" if (domainish or site) and raw_app not in ("Adobe Illustrator 2026",) else raw_app
            draft = ObservationDraft(
                source="replay", event_kind=r["event_kind"], session_id=r["session_id"],
                started_at=ts, ended_at=ts + 5, app=app, window_title=r["window_title"] or "", url=url, site=site,
                screen_text=r["dense_text"] or "", text_source="replay", caption=r["caption"] or "",
                keys=1 if r["event_kind"] == "keyboard" else 0, clicks=1 if r["event_kind"] == "mouse" else 0,
            )
            oid, is_new = await store.add_observation(draft)
            if is_new and oid:
                new += 1
                await tracker.on_observation(oid, await store.get_observation(oid), session_id=draft.session_id)
        await tracker.close_current()
        _print(f"ingested {new}/{len(rows)} observations (rest collapsed as duplicates)")
        pipe = ProcessPipeline(store)
        while batch := await pipe.process_ready(limit=5):
            for res in batch:
                _print(f"episode {res.episode_id}: +{res.added} ~{res.reinforced} !{res.invalidated} rejected={res.rejected}")
        _print(f"embedded {await drain_outbox(store)}")
        if not args.no_consolidate:
            _print(json.dumps(await run_consolidation(store), default=str))
    finally:
        await store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="discrete")
    p.add_argument("--db", help="store path (default: $DISCRETE_DATA_DIR/discrete.db)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    prun = sub.add_parser("run", help="capture and process continuously, with Bubble on the hotkey")
    prun.add_argument("--no-bubble", action="store_true", help="capture only; do not arm the hotkey")
    pk = sub.add_parser("ask", help="run the Bubble pipeline headlessly on some field text")
    pk.add_argument("text")
    pk.add_argument("--app", default="Terminal")
    pk.add_argument("--records", action="store_true", help="print the evidence pack sent to the writer")
    pk.add_argument("--overlay", action="store_true", help="also show the status pill")
    pd = sub.add_parser("overlay-demo", help="preview the retrieval animation")
    pd.add_argument("--step", type=float, default=1.3)
    sub.add_parser("doctor", help="check permissions, capture and endpoints")
    sub.add_parser("stats", help="print the store card")
    pp = sub.add_parser("process", help="draft ready episodes and drain the embedding outbox")
    pp.add_argument("--recover", action="store_true", help="also close open episodes left by a dead run")
    pp.add_argument("--consolidate", action="store_true")
    sub.add_parser("consolidate")
    pq = sub.add_parser("query")
    pq.add_argument("text")
    pq.add_argument("--field", default="")
    pq.add_argument("--gap", action="store_true", help="verify and record a gap if unanswerable")
    pe = sub.add_parser("eval")
    pe.add_argument("--cases", help="case file (default golden/cases.yaml)")
    pe.add_argument("--save-baseline", action="store_true")
    pe.add_argument("--verbose", action="store_true")
    sub.add_parser("gaps", help="list open knowledge gaps")
    pa = sub.add_parser("answer")
    pa.add_argument("gap_id", type=int)
    pa.add_argument("text")
    pr = sub.add_parser("replay-workaa", help="rebuild a store from WorkAA observations")
    pr.add_argument("--source", default="~/.cache/workaa/workaa.db")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--fresh", action="store_true")
    pr.add_argument("--no-consolidate", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "replay-workaa" and not args.db:
        from .config import data_dir

        args.db = str(data_dir() / "replay-workaa.db")
    if args.db:
        os.environ["DISCRETE_DB"] = str(Path(args.db).expanduser())
    if args.cmd != "run":
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                            format="%(levelname)s %(name)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
    handler = globals()["cmd_" + args.cmd.replace("-", "_")]
    try:
        code = asyncio.run(handler(args))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
