"""Runtime orchestration — full planS1-3 pipeline."""

from __future__ import annotations

import asyncio
import logging

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

log = logging.getLogger("discrete.app")


class Runtime:
    def __init__(self, store: Store):
        self.store = store
        self.queue: asyncio.Queue = asyncio.Queue()
        self.session = SessionTracker()
        self.observer = ScreenObserver(queue=self.queue, session=self.session)
        self.episodes = EpisodeTracker(store)
        self.pipeline = ProcessPipeline(store)
        self.events = EventTriggers(on_trigger=self.observer._schedule)
        self._tasks: list[asyncio.Task] = []
        self._last_consolidation = 0.0

    async def start(self) -> None:
        await self.observer.start()
        await self.events.start()
        self._tasks = [
            asyncio.create_task(self._ingest_loop(), name="discrete-ingest"),
            asyncio.create_task(self._draft_loop(), name="discrete-draft"),
            asyncio.create_task(self._maint_loop(), name="discrete-maint"),
            asyncio.create_task(self._heartbeat_loop(), name="discrete-heartbeat"),
        ]
        log.info("discrete runtime started")

    async def stop(self) -> None:
        await self.events.stop()
        await self.observer.stop()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.store.close()

    async def _ingest_loop(self) -> None:
        while True:
            draft = await self.queue.get()
            try:
                oid, is_new = await self.store.add_observation(draft)
                if not is_new or not oid:
                    continue
                row = await self.store.get_observation(oid)
                idle = draft.event_kind == "idle"
                closed = await self.episodes.on_observation(
                    oid, row, session_id=draft.session_id, idle_transition=idle
                )
                for eid in closed:
                    await self.pipeline.draft_episode(eid)
            except Exception:
                log.exception("ingest failed")

    async def _draft_loop(self) -> None:
        while True:
            try:
                for ep in await self.store.episodes_ready_for_draft(limit=3):
                    await self.pipeline.draft_episode(ep["id"])
                closed = await self.episodes.close_idle()
                for eid in closed:
                    await self.pipeline.draft_episode(eid)
            except Exception:
                log.exception("draft loop failed")
            await asyncio.sleep(5.0)

    async def _maint_loop(self) -> None:
        while True:
            try:
                n = await drain_outbox(self.store)
                if n:
                    log.debug("embedded %s rows", n)
                import time

                if time.time() - self._last_consolidation >= T.CONSOLIDATION_INTERVAL_S:
                    rep = await run_consolidation(self.store)
                    self._last_consolidation = time.time()
                    log.info("consolidation: %s", rep)
                await run_retention(self.store)
            except Exception:
                log.exception("maint failed")
            await asyncio.sleep(60.0)

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await self.observer.heartbeat_tick()
            except Exception:
                log.exception("heartbeat failed")
            await asyncio.sleep(T.HEARTBEAT_TICK_S)


async def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    store = await Store.open()
    rt = Runtime(store)
    await rt.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await rt.stop()
