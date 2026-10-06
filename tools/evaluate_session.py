"""Evaluate final answers with the same harness against any revision's pipeline.

Run with PYTHONPATH pointing at the baseline or working checkout. The harness
itself stays fixed, so metric fixes do not invalidate before/after comparisons.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

from discrete.llm import flush_usage
from discrete.perf import PERF
from discrete.store import Store

HARNESS = Path(__file__).resolve().parents[1] / "discrete/evaluation/harness.py"


async def evaluate(args):
    PERF.source = "eval"
    name = "discrete.evaluation.session_harness"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    sys.modules[name] = harness
    spec.loader.exec_module(harness)
    store = await Store.open(args.db)
    try:
        cases, _ = harness.load_cases(Path(args.cases))
        sem = asyncio.Semaphore(args.concurrency)

        async def one(case):
            async with sem:
                result = await harness.run_case(store, case, answers=True)
                print(json.dumps({"question": case.question, "correct": result.correct, "mode": result.mode,
                                  "answer": result.answer}), flush=True)
                return result

        results = await asyncio.gather(*(one(case) for case in cases))
        suite = harness.SuiteResult(cases=results, health=await harness.health_checks(store))
        payload = {"metrics": suite.metrics(), "health": suite.health,
                   "cases": [{**asdict(result), "correct": result.correct} for result in results]}
        await asyncio.to_thread(Path(args.output).write_text, json.dumps(payload, indent=2))
        print(json.dumps(payload["metrics"]), flush=True)
    finally:
        await flush_usage(store)
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    asyncio.run(evaluate(parser.parse_args()))
