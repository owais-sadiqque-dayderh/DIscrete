"""Native OS event triggers (planS1 §2)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from ..config import T

log = logging.getLogger("discrete.capture.events")

TriggerFn = Callable[[str], None]


class EventTriggers:
    """Fires trigger kinds into the screen observer schedule path."""

    def __init__(self, on_trigger: TriggerFn):
        self.on_trigger = on_trigger
        self._loop: asyncio.AbstractEventLoop | None = None
        self._observers: list = []
        self._task: asyncio.Task | None = None
        self._last_clipboard = 0
        self._away = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            from Foundation import NSWorkspace

            nc = NSWorkspace.sharedWorkspace().notificationCenter()
            for name in (
                "NSWorkspaceDidActivateApplicationNotification",
                "NSWorkspaceDidLaunchApplicationNotification",
            ):
                nc.addObserverForName_object_queue_usingBlock_(name, None, None, self._app_change)
            self._observers.append(nc)
        except Exception as exc:
            log.warning("NSWorkspace notifications unavailable: %s", exc)
        self._task = asyncio.create_task(self._poll_loop(), name="discrete-events")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _fire(self, kind: str) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: self.on_trigger(kind))

    def _app_change(self, _notification) -> None:
        self._fire("app")

    def notify_title_change(self) -> None:
        self._fire("title")

    def notify_clipboard(self) -> None:
        self._fire("clipboard")

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_idle()
                await self._poll_clipboard()
            except Exception as exc:
                log.debug("event poll: %s", exc)
            await asyncio.sleep(T.POLL_S)

    async def _poll_idle(self) -> None:
        try:
            import Quartz

            idle = Quartz.CGEventSourceSecondsSinceLastEventType(
                Quartz.CGEventSourceStateCombinedSessionState,
                Quartz.kCGAnyInputEventType,
            )
            away = idle >= T.IDLE_THRESHOLD_S
            if away != self._away:
                self._away = away
                self._fire("idle")
        except Exception:
            pass

    async def _poll_clipboard(self) -> None:
        try:
            from AppKit import NSPasteboard

            pb = NSPasteboard.generalPasteboard()
            count = int(pb.changeCount())
            if count != self._last_clipboard:
                self._last_clipboard = count
                self._fire("clipboard")
        except Exception:
            pass
