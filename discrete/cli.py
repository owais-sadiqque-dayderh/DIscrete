"""CLI entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .app import run
from .config import all_endpoints, db_path, golden_dir
from .evaluation.harness import compare, load_cases, run_suite, save_baseline
from .process.consolidate import run_consolidation
from .process.embed_worker import drain_outbox
from .process.pipeline import ProcessPipeline
from .retrieval.gaps import answer_and_learn, maybe_record_gap
from .retrieval.grounding import evaluate
from .retrieval.search import retrieve
from .store import Store


def _print(*a, **k):
    print(*a, **k, flush=True)


async def cmd_run(_args) -> int:
    await run()
    return 0


async def cmd_doctor(_args) -> int:
    from .capture import accessibility as ax

    issues = ax.check_permissions()
    for ep in all_endpoints():
        _print(ep.describe())
    _print(f"db: {db_path()}")
    if issues:
        for i in issues:
            _print(f"ISSUE: {i}")
        return 1
    _print("OK")
    return 0


async def cmd_stats(_args) -> int:
    store = await Store.open()
    try:
        _print(json.dumps(await store.stats(), indent=2, default=str))
    finally:
        await store.close()
    return 0


async def cmd_query(args) -> int:
    store = await Store.open()
    try:
        result = await retrieve(store, args.text, field_text=args.field or "")
        decision = evaluate(result, intent=result.plan.intent)
        _print(f"intent={result.plan.intent} mode={decision.mode} exhaustive={result.exhaustive}")
        for h in result.hits:
            flag = "*" if h.strong else " "
            _print(f"{flag}[{h.ref_kind}:{h.ref_id}] {h.title}")
            from . import textkit as tk

            _print(tk.truncate(h.body, 300))
            _print("---")
        if args.gap and decision.mode in ("thin", "empty"):
            outcome = await maybe_record_gap(store, args.text, field_text=args.field or "")
            _print(f"gap outcome: {outcome.kind}")
    finally:
        await store.close()
    return 0


async def cmd_eval(args) -> int:
    store = await Store.open()
    try:
        cases = load_cases()
        if not cases:
            _print("No cases in golden/cases.yaml")
            return 1
        result = await run_suite(store, cases)
        _print(f"hit@8: {result.hit_at_k:.3f} refusal: {result.refusal_accuracy:.3f} n={len(cases)}")
        _print(f"health: {result.health}")
        if args.save_baseline:
            save_baseline(result)
            _print("baseline saved")
        baseline_path = golden_dir() / "baseline.json"
        if baseline_path.exists() and not args.save_baseline:
            base_data = json.loads(baseline_path.read_text())
            from .evaluation.harness import SuiteResult

            baseline = SuiteResult()
            baseline.cases = []
            comp = compare(result, baseline)
            comp["baseline_hit"] = base_data.get("hit_at_k")
            _print(json.dumps(comp, indent=2))
    finally:
        await store.close()
    return 0


async def cmd_consolidate(_args) -> int:
    store = await Store.open()
    try:
        rep = await run_consolidation(store)
        _print(json.dumps(rep, default=str))
    finally:
        await store.close()
    return 0


async def cmd_process(_args) -> int:
    store = await Store.open()
    try:
        pipe = ProcessPipeline(store)
        n = 0
        for ep in await store.episodes_ready_for_draft(limit=10):
            await pipe.draft_episode(ep["id"])
            n += 1
        embedded = await drain_outbox(store)
        _print(json.dumps({"drafted": n, "embedded": embedded}))
    finally:
        await store.close()
    return 0


async def cmd_answer(args) -> int:
    store = await Store.open()
    try:
        out = await answer_and_learn(store, args.gap_id, args.text)
        _print(json.dumps(out))
    finally:
        await store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="discrete")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("doctor")
    sub.add_parser("stats")
    sub.add_parser("process")
    sub.add_parser("consolidate")
    pr = sub.add_parser("query")
    pr.add_argument("text")
    pr.add_argument("--field", default="")
    pr.add_argument("--gap", action="store_true", help="Record gap if unanswerable")
    ev = sub.add_parser("eval")
    ev.add_argument("--save-baseline", action="store_true")
    ans = sub.add_parser("answer")
    ans.add_argument("gap_id", type=int)
    ans.add_argument("text")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handlers = {
        "run": cmd_run,
        "doctor": cmd_doctor,
        "stats": cmd_stats,
        "query": cmd_query,
        "eval": cmd_eval,
        "process": cmd_process,
        "consolidate": cmd_consolidate,
        "answer": cmd_answer,
    }
    try:
        code = asyncio.run(handlers[args.cmd](args))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
