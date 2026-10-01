"""Vision enrichment, off the capture path (planS1 §6.2).

The observer stores each capture immediately with a basic caption ("App:
Window") and hands its frames here. A bounded pool of workers makes the vision
call and fills in the caption, entities and page fields afterwards, so a slow
or failing model delays captions, never captures. A capture that dedups
against an existing row costs no model call at all.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import textkit as tk
from ..aio import spawn
from ..config import T, caption_endpoint
from ..llm import EXPECTED_ERRORS, TokenLimit, complete_json, image_part
from ..models import ObservationDraft
from ..perf import PERF, since_ms
from ..store import Store
from .frames import downscale
from .prompts import CAPTURE_PROMPT

log = logging.getLogger("discrete.capture.enrich")


# The caption's answer budget; a reasoning model's thinking needs CAPTION_LM_EXTRA_TOKENS on top.
VLM_MAX_TOKENS = 500


@dataclass
class EnrichJob:
    frames: list[Image.Image]
    app: str
    window_title: str
    url: str
    screen_text: str
    counts: dict[str, int]
    prior_caption: str = ""  # backfill only: the row's existing caption, as extra context


@dataclass
class Capture:
    """What the observer queues: the row to store now, and its deferred vision call."""

    draft: ObservationDraft
    job: EnrichJob | None = None
    queued_at: float = field(default_factory=time.perf_counter)


@dataclass
class EnrichStats:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        ("captured", "dropped_useless", "degraded", "vlm_attempts_failed"), 0))


class Enricher:
    def __init__(self, store: Store, *, concurrency: int | None = None):
        self.store = store
        self.ep = caption_endpoint()
        self._sem = asyncio.Semaphore(concurrency or T.ENRICH_CONCURRENCY)
        self._tasks: set[asyncio.Task] = set()
        self.stats = EnrichStats().counts

    @property
    def configured(self) -> bool:
        return self.ep.configured

    def submit(self, oid: int, job: EnrichJob) -> asyncio.Task:
        """Start enriching `oid`; the returned task finishes when its row is final."""
        return spawn(self._run(oid, job), name=f"enrich-{oid}", owner=self._tasks)

    async def cancel_all(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, oid: int, job: EnrichJob) -> None:
        t0 = time.perf_counter()
        try:
            async with self._sem:
                PERF.sample("enrich.wait", since_ms(t0), ref_id=oid)  # queued behind other vision calls
                t_call = time.perf_counter()
                data = await self.vlm(job)
                PERF.sample("enrich.vision", since_ms(t_call), n=len(job.frames), ref_id=oid)
        except asyncio.CancelledError:
            await asyncio.shield(self.store.enrichment_failed(oid))
            raise
        try:
            await self._apply(oid, job, data)
        finally:
            PERF.sample("enrich.total", since_ms(t0), ref_id=oid)  # stored → final caption

    async def _apply(self, oid: int, job: EnrichJob, data: dict | None) -> None:
        if data is None:
            self.stats["degraded"] += 1
            await self.store.enrichment_failed(oid)
            return
        caption = str(data.get("caption") or "").strip()
        if not data.get("useful", True) or (len(job.screen_text) < 40 and len(caption) < 5):
            self.stats["dropped_useless"] += 1
            await self.store.discard_observation(oid)
            return
        self.stats["captured"] += 1
        await self.store.enrich_observation(
            oid, caption=caption,
            entities=[str(e).strip() for e in (data.get("entities") or []) if str(e).strip()],
            page_title=tk.truncate(str(data.get("page_title") or "").strip(), 200),
            is_listing=data.get("is_listing") is True,
            navigated_from=tk.truncate(str(data.get("navigated_from") or "").strip(), 200),
        )

    async def vlm(self, job: EnrichJob) -> dict | None:
        """The vision call with retries; no sleep after the final attempt."""
        if not self.ep.configured:
            return None
        messages = vlm_messages(job)
        for attempt in range(T.CAPTURE_MAX_RETRIES):
            if attempt:
                await asyncio.sleep(T.CAPTURE_RETRY_BASE_S * (2 ** (attempt - 1)))
            try:
                data = await complete_json(self.ep, messages, max_tokens=VLM_MAX_TOKENS)
                if isinstance(data, dict):
                    return data
            except TokenLimit as exc:  # the same request would run out again: no retry
                log.warning("VLM: %s", exc)
                self.stats["vlm_attempts_failed"] += 1
                return None
            except EXPECTED_ERRORS as exc:
                log.warning("VLM attempt %s failed: %s", attempt + 1, exc)
            self.stats["vlm_attempts_failed"] += 1
        return None


def vlm_messages(job: EnrichJob) -> list[dict]:
    """The vision request for one capture (also used by `discrete doctor --models`)."""
    imgs = [image_part(jpeg(downscale(f, T.VLM_MAX_WIDTH)), detail=T.VLM_IMAGE_DETAIL)
            for f in job.frames if f is not None]
    prompt = CAPTURE_PROMPT.format(
        app=job.app, window_title=job.window_title, url=job.url,
        screen_text=tk.truncate(job.screen_text, 4000), **job.counts,
    )
    if job.prior_caption:
        prompt += ("\nAn earlier description of this moment (data, may be imprecise):\n<<<DATA>>>\n"
                   f"{job.prior_caption}\n<<<END>>>\n")
    return [{"role": "user", "content": [{"type": "text", "text": prompt}, *imgs]}]


async def backfill_page_fields(store: Store, *, limit: int = 500) -> dict[str, int]:
    """One-off upgrade for rows captured before page fields existed: one vision call
    per row (its saved frame when still on disk, else text only) fills page_title,
    is_listing and navigated_from. The caption is kept, so evidence quotes still match."""
    enricher = Enricher(store)
    out = {"rows": 0, "filled": 0, "failed": 0, "with_frame": 0}
    if not enricher.configured:
        return out
    rows = await store.observations_missing_page_fields(limit)
    out["rows"] = len(rows)

    def load(path: str) -> list[Image.Image]:
        if not path or not Path(path).exists():
            return []
        with Image.open(path) as im:
            return [im.convert("RGB")]

    async def one(r) -> None:
        frames = await asyncio.to_thread(load, r["frame_path"])
        out["with_frame"] += bool(frames)
        job = EnrichJob(frames=frames, app=r["app"], window_title=r["window_title"], url=r["url"],
                        screen_text=r["screen_text"], prior_caption=r["caption"],
                        counts={k: r[k] for k in ("keys", "clicks", "scrolls", "clipboard")})
        async with enricher._sem:
            data = await enricher.vlm(job)
        if data is None:
            out["failed"] += 1
            return
        await store.enrich_observation(
            r["id"], caption=r["caption"], entities=tk.parse_json_list(r["entities"]),
            page_title=tk.truncate(str(data.get("page_title") or "").strip(), 200),
            is_listing=data.get("is_listing") is True,
            navigated_from=tk.truncate(str(data.get("navigated_from") or "").strip(), 200),
        )
        out["filled"] += 1

    await asyncio.gather(*(one(r) for r in rows))
    return out


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=T.FRAME_JPEG_QUALITY)
    return buf.getvalue()
