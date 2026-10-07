"""Runtime orchestration: capture → store → enrich → embed + segment → drafting → consolidation.

    observer ──queue──▶ ingest ──▶ store row (basic caption) ──▶ enricher (vision, bounded pool)
                                         │                               │
                                         └──── segmenter (oldest row) ◀──┘ waits for its row's caption
                                                     │
                                                     ▼
                                   embed once (vector stored) → episode tracker

Ingest never waits on a model call. The database is the segmentation queue: the
segmenter always takes the oldest unsegmented row and waits while its caption is
pending, so episodes see final captions in capture order. A caption backlog stays
on disk (row + saved frame), never in memory, and resumes after a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time

from .aio import daemon_lease, spawn
from .capture.enrich import Capture, Enricher
from .capture.events import EventTriggers
from .capture.retention import run_retention
from .capture.screen import ScreenObserver
from .capture.session import SessionTracker
from .config import T
from .llm import flush_usage
from .perf import PERF, since_ms
from .process.consolidate import run_consolidation
from .process.dream import due as dream_due
from .process.dream import run_dream
from .process.embed_worker import drain_outbox, embed_row, observation_text
from .process.entities import promote_recurring
from .process.episodes import EpisodeTracker
from .process.pipeline import ProcessPipeline
from .process.provenance import build_provenance
from .process.trails import build_trails
from .store import Store
from .timeutil import now

log = logging.getLogger("discrete.app")


class CaptureQueue(asyncio.Queue):
    async def put(self, item) -> None:
        # Overload drops the new capture explicitly rather than retaining waiting
        # producers and their decoded frames outside the bounded queue.
        try:
            self.put_nowait(item)
        except asyncio.QueueFull:
            PERF.count("health.capture_queue_dropped")
            raise


class Runtime:
    def __init__(self, store: Store, *, bubble: bool = True):
        self.store = store
        self.bubble = None
        if bubble:
            from .bubble.service import BubbleService  # AppKit/Quartz: only when Bubble is on

            self.bubble = BubbleService(store)
        self.queue: asyncio.Queue[Capture] = CaptureQueue(maxsize=T.CAPTURE_QUEUE_MAX)
        # Set when a row is stored or a caption finishes: segmentation may advance.
        self._wake = asyncio.Event()
        self.session = SessionTracker()
        self.observer = ScreenObserver(queue=self.queue, session=self.session, store=store)
        self.enricher = Enricher(store, on_done=self._wake.set)
        self.episodes = EpisodeTracker(store)
        self.pipeline = ProcessPipeline(store)
        self.events = EventTriggers(on_trigger=self.observer.trigger, on_idle=self._on_idle)
        self._tasks: list[asyncio.Task] = []
        self._bg: set[asyncio.Task] = set()
        self._ep_lock = asyncio.Lock()
        self._counted: dict[str, int] = {}
        self._seg_failures: dict[int, int] = {}
        self._started = 0.0

    async def start(self) -> None:
        await self.store.reset_stuck_drafting()
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

    async def wait(self) -> None:
        """Any essential loop ending is fatal; the caller shuts down and exits loudly."""
        tasks = list(self._tasks)
        tasks += [t for t in (self.observer._frame_task, self.events._task) if t is not None]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        task = next(iter(done))
        if not task.cancelled() and task.exception() is not None:
            raise RuntimeError(f"Essential worker {task.get_name()} failed") from task.exception()
        raise RuntimeError(f"Essential worker {task.get_name()} stopped unexpectedly")

    async def stop(self) -> None:
        try:
            await self._stop()
        finally:
            for task in (*self._tasks, *self._bg):
                task.cancel()
            await asyncio.gather(*self._tasks, *self._bg, return_exceptions=True)
            try:
                await self.enricher.cancel_all()
            finally:
                await self.store.close()

    async def _stop(self) -> None:
        if self.bubble:
            await self.bubble.stop()
        await self.events.stop()
        await self.observer.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Captions in flight stop; their rows stay pending with a saved frame and are
        # captioned next run. Queued captures are stored (no model calls), and rows up
        # to the first pending caption are segmented, keeping capture order. Vectors
        # come from the outbox next run.
        await self.enricher.cancel_all()
        while not self.queue.empty():
            try:
                await self._ingest(self.queue.get_nowait())
            except Exception:
                log.exception("final ingest failed")
        for row in await self.store.unsegmented_observations(limit=T.CAPTURE_QUEUE_MAX):
            if row["enrich_state"] == "pending":
                break
            try:
                await self._segment(row["id"], row["session_id"], embed=False)
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

    def _on_idle(self, away: bool) -> None:
        self.observer.away = away
        self.session.new_session()
        if away:
            if not self._bg:
                spawn(self._close_episode(), name="discrete-idle-close", owner=self._bg)

    async def _close_episode(self) -> None:
        async with self._ep_lock:
            await self.episodes.close_current()

    async def _ingest(self, cap: Capture) -> None:
        oid, is_new = await self.store.add_observation(cap.draft)
        if not (is_new and oid) or cap.draft.origin != "user":
            return  # a duplicate costs no vision call
        PERF.sample("ingest.stored", since_ms(cap.queued_at), ref_id=oid)  # queued → row written
        if cap.job is not None:
            self.enricher.submit(oid, cap.job)  # over the pool limit: resumed from its saved frame
        self._wake.set()

    async def _report_error(self, name: str) -> None:
        try:
            await self.store.bump_metric(name)
        except Exception:
            PERF.count(f"health.{name}")
            log.warning("could not persist runtime error counter %s", name)

    async def _ingest_loop(self) -> None:
        while True:
            cap = await self.queue.get()
            try:
                # Keep ownership of the dequeued capture across transient failures.
                # Persistent failures exit through essential-worker supervision.
                for attempt in range(3):
                    try:
                        await self._ingest(cap)
                        break
                    except Exception:
                        log.exception("ingest failed")
                        await self._report_error("ingest_errors")
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.5 * 2 ** attempt)
            finally:
                self.queue.task_done()

    async def _segment_loop(self) -> None:
        while True:
            self._wake.clear()
            rows = []
            try:
                skip = {oid for oid, n in self._seg_failures.items() if n >= 3}
                rows = [r for r in await self.store.unsegmented_observations(limit=len(skip) + 1)
                        if r["id"] not in skip]
                if rows and rows[0]["enrich_state"] != "pending":
                    await self._segment(rows[0]["id"], rows[0]["session_id"])
                    continue
                if rows:
                    await self.enricher.fill_pending()  # the oldest row waits; keep the pool busy
            except Exception:
                log.exception("segmentation failed")
                await self._report_error("segment_errors")
                if rows:
                    # A row that keeps failing is left unassigned for this run instead of
                    # blocking every later capture.
                    self._seg_failures[rows[0]["id"]] = self._seg_failures.get(rows[0]["id"], 0) + 1
                await asyncio.sleep(1.0)
                continue
            # Woken by ingest or a finished caption; the timeout is only a safety net.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=30.0)

    async def _segment(self, oid: int, session_id: str, *, embed: bool = True) -> None:
        row = await self.store.get_observation(oid)
        if row is None or row["origin"] != "user":
            return  # discarded as useless by the vision call
        vec = None
        if embed:
            with PERF.timer("segment.embed", ref_id=oid):
                vec = await embed_row(self.store, "observation", oid, observation_text(row))
        async with self._ep_lock:
            await self.episodes.on_observation(oid, row, session_id=session_id, vec=vec)
        PERF.sample("segment.lag", max(0.0, (now() - row["created_at"]) * 1000), ref_id=oid)  # stored → in an episode

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
                await self._report_error("draft_loop_errors")
            await asyncio.sleep(5.0)

    async def _maint_loop(self) -> None:
        last = {"retention": 0.0, "entities": 0.0}

        async def consolidation() -> None:
            if now() - float(await self.store.get_meta("last_consolidation", "0") or 0) >= T.CONSOLIDATION_INTERVAL_S:
                log.info("consolidation: %s", await run_consolidation(self.store))

        async def trails() -> None:
            if made := await build_trails(self.store):
                log.info("causal trails: %d links", made)

        async def entities() -> None:
            # Entities and their links between full consolidations, so a session shorter
            # than the merge interval still gets a graph.
            if time.time() - last["entities"] >= T.ENTITY_INTERVAL_MIN * 60:
                last["entities"] = time.time()
                if promoted := await promote_recurring(self.store):
                    log.info("entities: +%d, %d links", promoted, await self.store.rebuild_edges())

        async def dream() -> None:
            if await dream_due(self.store):
                log.info("dream: %s", await run_dream(self.store))

        async def retention() -> None:
            if time.time() - last["retention"] >= 3600:
                last["retention"] = time.time()
                log.info("retention: %s", await run_retention(self.store))

        async def housekeeping() -> None:
            await self.store.retry_failed_drafts(older_than_s=T.FAILED_WORK_RETRY_S)
            await self.store.retry_failed_outbox(older_than_s=T.FAILED_WORK_RETRY_S)
            await drain_outbox(self.store)

        async def provenance() -> None:
            await build_provenance(self.store)

        steps = (provenance, housekeeping, trails, entities, consolidation, dream, retention)
        while True:
            # Each step on its own: one failing (a model outage, a bad reply) must not
            # starve the others, as a single shared try did.
            for step in steps:
                try:
                    await step()
                except Exception:
                    log.exception("maintenance step %s failed", step.__name__)
                    await self._report_error(f"maint_errors_{step.__name__}")
            try:
                self._flush_counters()
                await flush_usage(self.store)
            except Exception:
                log.exception("usage flush failed")
            await asyncio.sleep(60.0)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(T.HEARTBEAT_TICK_S)
            try:
                await self.observer.heartbeat_tick()
            except Exception:
                log.exception("heartbeat failed")


async def run(*, bubble: bool = True) -> None:
    with daemon_lease():
        await _run(bubble=bubble)


async def _run(*, bubble: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    PERF.source = "live"
    store = await Store.open()
    rt = Runtime(store, bubble=bubble)
    # SIGTERM (kill, launchd) stops as cleanly as Ctrl-C: work in flight is closed and
    # the last minute of measurements is written.
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    try:
        await rt.start()
        waiter = asyncio.create_task(rt.wait(), name="discrete-supervisor")
        stopper = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((waiter, stopper), return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                await waiter
        finally:
            waiter.cancel()
            stopper.cancel()
            await asyncio.gather(waiter, stopper, return_exceptions=True)
    finally:
        await rt.stop()
