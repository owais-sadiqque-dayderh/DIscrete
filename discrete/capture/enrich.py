"""Vision enrichment, off the capture path (planS1 §6.2).

The observer stores each capture immediately with a basic caption ("App:
Window") and hands its frames here. A bounded pool of workers makes the vision
call and fills in the caption, entities and page fields afterwards, so a slow
or failing model delays captions, never captures. A capture that dedups
against an existing row costs no model call at all.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .. import textkit as tk
from ..aio import spawn
from ..config import T, caption_endpoint
from ..encryption import open_frame
from ..llm import EXPECTED_ERRORS, TokenLimit, complete_json, image_part
from ..models import ObservationDraft
from ..perf import PERF, since_ms
from ..store import Store
from .frames import downscale
from .prompts import CAPTURE_PROMPT
from .redact import redact

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
    # The screen text minus this app/site's navigation chrome, for the prompt (set by the
    # Enricher; the stored row keeps the full text).
    content_text: str | None = None


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
    """Vision captions with bounded memory: at most ENRICH_MAX_PENDING rows hold a task
    (and their images) at once. Rows beyond that stay `pending` in the store with a
    saved frame, and `fill_pending` starts them as slots free up, oldest first."""

    def __init__(self, store: Store, *, concurrency: int | None = None, on_done=None):
        self.store = store
        self.ep = caption_endpoint()
        self._sem = asyncio.Semaphore(concurrency or T.ENRICH_CONCURRENCY)
        self._tasks: set[asyncio.Task] = set()
        self._active: set[int] = set()
        self._on_done = on_done  # called when a row's caption is final (the runtime's wake-up)
        self._closed = False
        self.stats = EnrichStats().counts

    @property
    def configured(self) -> bool:
        return self.ep.configured

    def submit(self, oid: int, job: EnrichJob) -> asyncio.Task | None:
        """Start enriching `oid`; the returned task finishes when its row is final.
        None when the pool is full (or shutting down): the row stays pending."""
        if self._closed or oid in self._active or len(self._active) >= T.ENRICH_MAX_PENDING:
            return None
        self._active.add(oid)
        task = spawn(self._run(oid, job), name=f"enrich-{oid}", owner=self._tasks)

        def done(_: asyncio.Task) -> None:
            self._active.discard(oid)
            if self._on_done is not None:
                self._on_done()

        task.add_done_callback(done)
        return task

    async def fill_pending(self) -> int:
        """Start pending rows from their saved frame, oldest first, while slots are free."""
        slots = T.ENRICH_MAX_PENDING - len(self._active)
        if slots <= 0 or self._closed:
            return 0
        n = 0
        for r in await self.store.pending_enrichment(limit=T.ENRICH_MAX_PENDING + len(self._active)):
            if n >= slots:
                break
            if r["id"] in self._active:
                continue

            def load(path=r["frame_path"]) -> list[Image.Image]:
                if path and Path(path).exists():
                    with open_frame(path) as im:
                        return [im.convert("RGB")]
                return []

            frames = await asyncio.to_thread(load)
            row = await self.store.get_observation(r["id"])
            if row is None or row["enrich_state"] != "pending" or r["id"] in self._active:
                continue  # finished while the frame loaded
            job = EnrichJob(frames, r["app"], r["window_title"], r["url"], r["screen_text"],
                            {k: r[k] for k in ("keys", "clicks", "scrolls", "clipboard")})
            if self.submit(r["id"], job) is not None:
                n += 1
        return n

    async def cancel_all(self) -> None:
        """Shutdown: stop captions in flight and accept no new ones. Their rows stay
        pending (the saved frame is on disk) and are captioned on the next run."""
        self._closed = True
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, oid: int, job: EnrichJob) -> None:
        t0 = time.perf_counter()
        try:
            async with self._sem:
                PERF.sample("enrich.wait", since_ms(t0), ref_id=oid)  # queued behind other vision calls
                t_call = time.perf_counter()
                job = await self._declutter(oid, job)
                data = await self.vlm(job)
                PERF.sample("enrich.vision", since_ms(t_call), n=len(job.frames), ref_id=oid)
            await self._apply(oid, job, data)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never leave a row pending after an unexpected error: segmentation waits on it.
            log.exception("enrichment of %s failed", oid)
            self.stats["degraded"] += 1
            await self.store.enrichment_failed(oid)
        finally:
            PERF.sample("enrich.total", since_ms(t0), ref_id=oid)  # stored → final caption

    async def _declutter(self, oid: int, job: EnrichJob) -> EnrichJob:
        """Drop this app/site's recurring chrome from what the caption model reads."""
        if T.CHROME_SHARE <= 0 or not job.screen_text:
            return job
        row = await self.store.get_observation(oid)
        if row is None:
            return job
        texts = await self.store.recent_texts(row["app"], row["site"], before=row["started_at"],
                                              limit=T.CHROME_WINDOW)
        chrome = tk.chrome_lines(texts, share=T.CHROME_SHARE, min_texts=T.CHROME_MIN_CAPTURES)
        return dataclasses.replace(job, content_text=tk.without_lines(job.screen_text, chrome)) if chrome else job

    async def _apply(self, oid: int, job: EnrichJob, data: dict | None) -> None:
        if data is None:
            self.stats["degraded"] += 1
            await self.store.enrichment_failed(oid)
            return
        caption = str(data.get("caption") or "").strip()
        if T.REDACT_SECRETS:
            # The model saw the frame: a key it read off the screen must not be stored.
            caption = redact(caption).text
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
        screen_text=tk.truncate(job.content_text if job.content_text is not None else job.screen_text, 4000),
        **job.counts,
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
        with open_frame(path) as im:
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
