"""Runtime orchestration: capture → store → enrich → embed + segment → drafting → consolidation.

    observer ──queue──▶ ingest ──▶ store row (basic caption) ──▶ enricher (vision, pooled)
                                         │                               │
                                         └──── segment queue (FIFO) ◀────┘ awaits its row's enrichment
                                                     │
                                                     ▼
                                   embed once (vector stored) → episode tracker

Ingest never waits on a model call. Segmentation consumes rows in capture order
and waits for each row's caption, so episodes see final captions and vectors.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from .aio import spawn
from .capture.enrich import Capture, Enricher
from .capture.events import EventTriggers
from .capture.retention import run_retention
from .capture.screen import ScreenObserver
from .capture.session import SessionTracker
from .config import T
from .llm import flush_usage
from .perf import PERF, since_ms
from .process.consolidate import run_consolidation
from .process.embed_worker import drain_outbox, embed_row, observation_text
from .process.episodes import EpisodeTracker
from .process.pipeline import ProcessPipeline
from .store import Store
from .timeutil import now

log = logging.getLogger("discrete.app")


class Runtime:
    def __init__(self, store: Store, *, bubble: bool = True):
        self.store = store
        self.bubble = None
        if bubble:
            from .bubble.service import BubbleService  # AppKit/Quartz: only when Bubble is on

            self.bubble = BubbleService(store)
        self.queue: asyncio.Queue[Capture] = asyncio.Queue()
        # (row id, session, its enrichment task, perf_counter when stored)
        self._segq: asyncio.Queue[tuple[int, str, asyncio.Task | None, float | None]] = asyncio.Queue()
        self.session = SessionTracker()
        self.observer = ScreenObserver(queue=self.queue, session=self.session)
        self.enricher = Enricher(store)
        self.episodes = EpisodeTracker(store)
        self.pipeline = ProcessPipeline(store)
        self.events = EventTriggers(on_trigger=self.observer.trigger, on_idle=self._on_idle)
        self._tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()
        self._ep_lock = asyncio.Lock()
        self._counted: dict[str, int] = {}
        self._started = 0.0

    async def start(self) -> None:
        await self.store.reset_stuck_drafting()
        stale = await self.store.fail_stale_enrichment()
        if stale:
            log.info("%d captures from a previous run keep their basic caption", stale)
        recovered = await self.store.recover_open_episodes()
        if recovered:
            log.info("recovered %d open episodes from a previous run", len(recovered))
        self._started = time.perf_counter()
        PERF.count("runtime.start")  # `discrete bench --since last-run` starts here
        await self.observer.start()
        await self.events.start()
        if self.bubble:
            await self.bubble.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="discrete-ingest"),
            asyncio.create_task(self._segment_loop(), name="discrete-segment"),
            asyncio.create_task(self._draft_loop(), name="discrete-draft"),
            asyncio.create_task(self._maint_loop(), name="discrete-maint"),
            asyncio.create_task(self._heartbeat_loop(), name="discrete-heartbeat"),
        ]
        log.info("discrete runtime started")

    async def stop(self) -> None:
        if self.bubble:
            await self.bubble.stop()
        await self.events.stop()
        await self.observer.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Captions still in flight are abandoned (their rows keep the basic caption);
        # the rows are still segmented so no capture is left without an episode. No
        # model calls on the way out: their vectors come from the outbox next run.
        await self.enricher.cancel_all()
        while not self._segq.empty():
            try:
                await self._segment(*self._segq.get_nowait(), embed=False)
            except Exception:
                log.exception("final segmentation failed")
        for t in list(self._bg):
            t.cancel()
        await asyncio.gather(*self._bg, return_exceptions=True)
        async with self._ep_lock:
            await self.episodes.close_current()
        self._flush_counters()
        if self._started:
            PERF.sample("runtime.up", since_ms(self._started))
        await flush_usage(self.store)
        log.info("capture stats: %s; captions: %s", self.observer.stats, self.enricher.stats)
        log.info("open work is drafted on the next `discrete run` or `discrete process`")
        await self.store.close()

    def _on_idle(self, away: bool) -> None:
        self.observer.away = away
        self.session.new_session()
        if away:
            spawn(self._close_episode(), name="discrete-idle-close", owner=self._bg)

    async def _close_episode(self) -> None:
        async with self._ep_lock:
            await self.episodes.close_current()

    async def _ingest_loop(self) -> None:
        while True:
            cap = await self.queue.get()
            try:
                oid, is_new = await self.store.add_observation(cap.draft)
                if not (is_new and oid) or cap.draft.origin != "user":
                    continue  # a duplicate costs no vision call
                stored = time.perf_counter()
                PERF.sample("ingest.stored", since_ms(cap.queued_at), ref_id=oid)  # queued → row written
                task = self.enricher.submit(oid, cap.job) if cap.job is not None else None
                await self._segq.put((oid, cap.draft.session_id, task, stored))
            except Exception:
                log.exception("ingest failed")
                await self.store.bump_metric("ingest_errors")

    async def _segment_loop(self) -> None:
        while True:
            item = await self._segq.get()
            try:
                await self._segment(*item)
            except Exception:
                log.exception("segmentation failed")
                await self.store.bump_metric("segment_errors")

    async def _segment(self, oid: int, session_id: str, task: asyncio.Task | None, stored: float | None = None, *,
                       embed: bool = True) -> None:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)  # the enricher records its own outcome
        row = await self.store.get_observation(oid)
        if row is None or row["origin"] != "user":
            return  # discarded as useless by the vision call
        vec = None
        if embed:
            with PERF.timer("segment.embed", ref_id=oid):
                vec = await embed_row(self.store, "observation", oid, observation_text(row))
        async with self._ep_lock:
            await self.episodes.on_observation(oid, row, session_id=session_id, vec=vec)
        if stored is not None:
            PERF.sample("segment.lag", since_ms(stored), ref_id=oid)  # stored → in an episode

    def _flush_counters(self) -> None:
        """Observer and enricher counters as perf events (deltas since the last flush),
        so `discrete bench` can count them over any time window."""
        for prefix, stats in (("capture", self.observer.stats), ("enrich", self.enricher.stats)):
            for k, v in stats.items():
                name = f"{prefix}.{k}"
                PERF.count(name, v - self._counted.get(name, 0))
                self._counted[name] = v

    async def _draft_loop(self) -> None:
        while True:
            try:
                async with self._ep_lock:
                    await self.episodes.close_idle()
                for res in await self.pipeline.process_ready(limit=3):
                    log.info("drafted episode %s: +%d ~%d !%d rejected=%s", res.episode_id, res.added,
                             res.reinforced, res.invalidated, res.rejected)
            except Exception:
                log.exception("draft loop failed")
                await self.store.bump_metric("draft_loop_errors")
            await asyncio.sleep(5.0)

    async def _maint_loop(self) -> None:
        last_retention = 0.0
        while True:
            try:
                await drain_outbox(self.store)
                last = float(await self.store.get_meta("last_consolidation", "0") or 0)
                if now() - last >= T.CONSOLIDATION_INTERVAL_S:
                    log.info("consolidation: %s", await run_consolidation(self.store))
                if time.time() - last_retention >= 3600:
                    log.info("retention: %s", await run_retention(self.store))
                    last_retention = time.time()
                self._flush_counters()
                await flush_usage(self.store)
            except Exception:
                log.exception("maintenance failed")
                await self.store.bump_metric("maint_errors")
            await asyncio.sleep(60.0)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(T.HEARTBEAT_TICK_S)
            try:
                await self.observer.heartbeat_tick()
            except Exception:
                log.exception("heartbeat failed")


async def run(*, bubble: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    PERF.source = "live"
    store = await Store.open()
    rt = Runtime(store, bubble=bubble)
    await rt.start()
    # SIGTERM (kill, launchd) stops as cleanly as Ctrl-C: work in flight is closed and
    # the last minute of measurements is written.
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    try:
        await stop.wait()
    finally:
        await rt.stop()
