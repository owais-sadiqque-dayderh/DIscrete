"""Compare configured and replacement models on recorded windows and support checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from discrete.capture.enrich import Enricher, EnrichJob
from discrete.config import caption_endpoint, insight_endpoint
from discrete.encryption import open_frame
from discrete.llm import complete_json
from discrete.process.prompts import VERIFY_CLAIMS_PROMPT
from discrete.store import Store


async def compare(args):
    raw = await asyncio.to_thread(Path(args.manifest).read_text)
    items = {x["label"]: x for x in map(json.loads, raw.splitlines())}
    checks = [
        ("third-party-preference", "preference", "The user prefers concise technical notes.", False),
        ("personal-preferences", "preference", "The user prefers local SQLite files for database prototypes.", True),
        ("personal-preferences", "event", "The user exported a report today.", False),
        ("personal-preferences", "event", "The user booked travel today.", False),
        ("postgres-gin", "event", "The user viewed PostgreSQL GIN index documentation.", True),
        ("postgres-gin", "fact", "The user migrated their database to PostgreSQL.", False),
        ("postgres-gin", "preference", "The user prefers PostgreSQL over SQLite.", False),
        ("budget-arithmetic", "event", "The user calculated 250 times 12, obtaining 3000.", True),
        ("budget-arithmetic", "event", "The user calculated 250 times 12, obtaining 3500.", False),
        ("budget-arithmetic", "event", "The user paid 3000 for a subscription.", False),
        ("scanned-review-ocr", "preference", "The user prefers SQLite for offline notebooks.", False),
        ("scanned-review-ocr", "event", "The user exported notebook-review.pdf.", False),
        ("third-party-preference", "fact", "The user regularly uses PostgreSQL for research.", False),
        ("personal-preferences", "preference", "The user prefers technical notes with explicit sources.", True),
    ]
    candidates = [{"index": i, "kind": kind, "text": text, "observations": [items[label]["draft"]]}
                  for i, (label, kind, text, _) in enumerate(checks)]
    sem = asyncio.Semaphore(3)

    async def one(model):
        async with sem:
            result = {"model": model}
            try:
                ep = replace(insight_endpoint(), model=model)
                start = time.perf_counter()
                data = await complete_json(ep, [{"role": "user", "content": VERIFY_CLAIMS_PROMPT.format(
                    candidates=json.dumps(candidates))}], max_tokens=3000)
                result["verification_ms"] = round(1000 * (time.perf_counter() - start))
                decisions = {c["index"]: c["supported"] for c in data.get("decisions", [])}
                result["verification_correct"] = sum(decisions.get(i) is expected for i, (*_, expected) in enumerate(checks))
                result["verification_n"] = len(checks)
                result["decisions"] = data
                store = await Store.open(":memory:")
                try:
                    enricher = Enricher(store)
                    enricher.ep = replace(caption_endpoint(), model=model)
                    result["captions"] = []
                    for label in ("budget-arithmetic", "scanned-review-ocr"):
                        item = items[label]
                        with open_frame(item["image"]) as im:
                            frames = [im.convert("RGB")]
                        d = item["draft"]
                        job = EnrichJob(frames, d["app"], d["window_title"], d["url"], d["screen_text"],
                                        {k: d[k] for k in ("keys", "clicks", "scrolls", "clipboard")})
                        start = time.perf_counter()
                        caption = await enricher.vlm(job)
                        result["captions"].append({"label": label, "ms": round(1000 * (time.perf_counter() - start)),
                                                   "result": caption})
                finally:
                    await store.close()
            except Exception as exc:
                result["error"] = str(exc)
            return result

    results = await asyncio.gather(*(one(model) for model in args.models.split(",")))
    await asyncio.to_thread(Path(args.output).write_text, json.dumps(results, indent=2))
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k not in ("decisions", "captions")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="meta/muse-spark-1.3-contributor,openai/gpt-6-luna,google/gemini-3.8-flash")
    asyncio.run(compare(parser.parse_args()))
