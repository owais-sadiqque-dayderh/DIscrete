"""Replay captured real task windows through models, segmentation and memory writes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from PIL import Image

from discrete.capture.enrich import Enricher, EnrichJob
from discrete.capture.redact import SecretMemory, redact_capture
from discrete.capture.suppress import classify_origin
from discrete.config import T
from discrete.llm import flush_usage
from discrete.models import ObservationDraft
from discrete.perf import PERF
from discrete.process.consolidate import run_consolidation
from discrete.process.embed_worker import embed_row, observation_text
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.store import Store

try:  # older checkouts (the baseline) have no causal trails
    from discrete.process.trails import build_trails
except ImportError:
    build_trails = None


async def replay(args):
    PERF.source = "replay"
    store = await Store.open(args.db)
    enricher = Enricher(store)
    tracker = EpisodeTracker(store)
    pipe = ProcessPipeline(store)
    try:
        # 1. Store every capture in order, as live capture would (origin, dedup).
        lines = await asyncio.to_thread(Path(args.manifest).read_text)
        stored: list[tuple[int, dict, ObservationDraft]] = []
        secrets = SecretMemory()
        for line in lines.splitlines():
            item = json.loads(line)
            draft = ObservationDraft(**item["draft"])
            # The same redaction live capture applies, so a manifest of old captures
            # can never send a key to a model.
            red = redact_capture(draft.window_title, draft.url, draft.screen_text, draft.text_source, memory=secrets)
            draft.window_title, draft.url, draft.screen_text = red.title, red.url, red.text
            if red.drop_frame:
                item.pop("image", None)
            if draft.origin == "user":
                # Live capture classifies origin before storing; a replay must too.
                draft.origin = classify_origin(draft.app, draft.window_title, draft.screen_text)
            oid, new = await store.add_observation(draft)
            if new and oid is not None:
                stored.append((oid, item, draft))

        # 2. Captions, DISCRETE_ENRICH_CONCURRENCY at a time (live capture also captions
        # asynchronously; each caption only reads captures before it).
        slots = asyncio.Semaphore(max(1, T.ENRICH_CONCURRENCY))

        async def caption(oid: int, item: dict, draft: ObservationDraft) -> None:
            if not item.get("image"):
                return
            async with slots:  # decode a frame only when a caption slot is free (bounded memory)
                with Image.open(item["image"]) as im:
                    job = EnrichJob([im.convert("RGB")], draft.app, draft.window_title, draft.url,
                                    draft.screen_text,
                                    {k: getattr(draft, k) for k in ("keys", "clicks", "scrolls", "clipboard")})
                await enricher._run(oid, job)

        await asyncio.gather(*(caption(*x) for x in stored))

        # 3. Segment in capture order, keeping the manifest's episode boundaries.
        group = None
        for oid, item, _ in stored:
            if group is not None and group != item["group"]:
                await tracker.close_current()
            group = item["group"]
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
        if build_trails is not None:
            print(json.dumps({"trail_links": await build_trails(store, limit=10_000)}), flush=True)
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
