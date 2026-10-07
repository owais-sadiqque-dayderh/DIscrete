"""Replay captured real task windows through models, segmentation and memory writes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from PIL import Image

from discrete.capture.enrich import Enricher, EnrichJob
from discrete.llm import flush_usage
from discrete.models import ObservationDraft
from discrete.perf import PERF
from discrete.process.consolidate import run_consolidation
from discrete.process.embed_worker import embed_row, observation_text
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.store import Store


async def replay(args):
    PERF.source = "replay"
    store = await Store.open(args.db)
    enricher = Enricher(store)
    tracker = EpisodeTracker(store)
    pipe = ProcessPipeline(store)
    try:
        group = None
        lines = await asyncio.to_thread(Path(args.manifest).read_text)
        for line in lines.splitlines():
            item = json.loads(line)
            if group is not None and group != item["group"]:
                await tracker.close_current()
            group = item["group"]
            draft = ObservationDraft(**item["draft"])
            oid, new = await store.add_observation(draft)
            if not new or oid is None:
                continue
            if item.get("image"):
                with Image.open(item["image"]) as im:
                    job = EnrichJob([im.convert("RGB")], draft.app, draft.window_title, draft.url, draft.screen_text,
                                   {k: getattr(draft, k) for k in ("keys", "clicks", "scrolls", "clipboard")})
                task = enricher.submit(oid, job)
                if task is not None:
                    await task
            row = await store.get_observation(oid)
            if row is None:
                print(json.dumps({"label": item["label"], "discarded": True}), flush=True)
                continue
            if row["origin"] == "user":
                vec = await embed_row(store, "observation", oid, observation_text(row))
                await tracker.on_observation(oid, row, session_id=item["group"], vec=vec)
            print(json.dumps({"label": item["label"], "id": oid, "caption": row["caption"]}), flush=True)
        await tracker.close_current()
        for result in await pipe.process_ready(limit=10_000):
            print(json.dumps(vars(result)), flush=True)
        print(json.dumps({"consolidation": await run_consolidation(store)}), flush=True)
        print(json.dumps({"stats": await store.stats()}), flush=True)
    finally:
        await enricher.cancel_all()
        await flush_usage(store)
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--db", required=True)
    asyncio.run(replay(parser.parse_args()))
