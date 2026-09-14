"""Runtime orchestration: capture → store → episodes → drafting → consolidation."""

from __future__ import annotations

import asyncio
import logging
import time

from .capture.events import EventTriggers
from .capture.retention import run_retention
from .capture.screen import ScreenObserver
from .capture.session import SessionTracker
from .config import T
from .process.consolidate import run_consolidation
from .process.embed_worker import drain_outbox
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
            from .bubble.service import BubbleService

            self.bubble = BubbleService(store)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.session = SessionTracker()
        self.observer = ScreenObserver(queue=self.queue, session=self.session)
        self.episodes = EpisodeTracker(store)
        self.pipeline = ProcessPipeline(store)
        self.events = EventTriggers(on_trigger=self.observer.trigger, on_idle=self._on_idle)
        self._tasks: list[asyncio.Task] = []
        self._ep_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.store.reset_stuck_drafting()
        recovered = await self.store.recover_open_episodes()
        if recovered:
            log.info("recovered %d open episodes from a previous run", len(recovered))
        await self.observer.start()
        await self.events.start()
        if self.bubble:
            await self.bubble.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="discrete-ingest"),
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
        async with self._ep_lock:
            await self.episodes.close_current()
        log.info("capture stats: %s", self.observer.stats)
        log.info("open work is drafted on the next `discrete run` or `discrete process`")
        await self.store.close()

    def _on_idle(self, away: bool) -> None:
        self.observer.away = away
        self.session.new_session()
        if away:
            asyncio.get_running_loop().create_task(self._close_episode())

    async def _close_episode(self) -> None:
        async with self._ep_lock:
            await self.episodes.close_current()

    async def _ingest_loop(self) -> None:
        while True:
            draft = await self.queue.get()
            try:
                oid, is_new = await self.store.add_observation(draft)
                if not (is_new and oid) or draft.origin != "user":
                    continue
                row = await self.store.get_observation(oid)
                async with self._ep_lock:
                    await self.episodes.on_observation(oid, row, session_id=draft.session_id)
            except Exception:
                log.exception("ingest failed")
                await self.store.bump_metric("ingest_errors")

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
    store = await Store.open()
    rt = Runtime(store, bubble=bubble)
    await rt.start()
    try:
        await asyncio.Event().wait()
    finally:
        await rt.stop()
