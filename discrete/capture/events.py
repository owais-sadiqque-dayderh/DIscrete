"""Context triggers: app/window/title change, clipboard, idle (planS1 §2, Rev 3).

A poller rather than NSWorkspace/AX notifications: those are delivered on a
Cocoa run loop, which the asyncio daemon does not run, and in v1 they never
fired. The window list is cheap to poll at `POLL_S`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from ..config import T
from . import accessibility as ax

log = logging.getLogger("discrete.capture.events")


class EventTriggers:
    def __init__(self, on_trigger: Callable[[str], None], on_idle: Callable[[bool], None] | None = None):
        self.on_trigger = on_trigger
        self.on_idle = on_idle or (lambda away: None)
        self._task: asyncio.Task | None = None
        self._last_key: tuple | None = None
        self._last_clipboard: int | None = None
        self.away = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop(), name="discrete-events")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _poll_loop(self) -> None:
        while True:
            try:
                self._poll_idle()
                if not self.away:
                    await self._poll_window()
                    self._poll_clipboard()
            except Exception as exc:
                log.debug("event poll: %s", exc)
            await asyncio.sleep(T.POLL_S)

    async def _poll_window(self) -> None:
        ctx = await asyncio.to_thread(ax.window_context)
        if not ctx.pid:
            return
        key = ctx.key
        if self._last_key is not None and key != self._last_key:
            self.on_trigger("app_switch" if key[0] != self._last_key[0] else "window_change")
        self._last_key = key

    def _poll_clipboard(self) -> None:
        from AppKit import NSPasteboard

        count = int(NSPasteboard.generalPasteboard().changeCount())
        if self._last_clipboard is not None and count != self._last_clipboard:
            self.on_trigger("clipboard")
        self._last_clipboard = count

    def _poll_idle(self) -> None:
        import Quartz

        idle = Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGAnyInputEventType
        )
        away = idle >= T.IDLE_THRESHOLD_S
        if away != self.away:
            # Idle transitions change sessions and episodes; they never capture (planS1 §2.7).
            self.away = away
            self.on_idle(away)
