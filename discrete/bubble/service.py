"""Bubble runtime: hotkey → focus snapshot → pill → engine → typed answer.

Capture is held off for the whole run (suppress.hold), and the typed text is
registered as agent output, so DIscrete never models its own answers.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..aio import spawn
from ..capture import accessibility as ax
from ..capture.privacy import denied
from ..capture.redact import redact
from ..capture.suppress import hold, note_agent_output, release
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed, interactive
from ..perf import PERF, since_ms
from ..retrieval.gaps import maybe_record_gap
from ..store import Store
from .engine import BubbleEngine, EmptyField, field_query
from .keys import HotkeyTap, type_text
from .overlay_client import Overlay

log = logging.getLogger("discrete.bubble")


class BubbleService:
    def __init__(self, store: Store, *, overlay: Overlay | None = None):
        self.store = store
        self.engine = BubbleEngine(store)
        self.overlay = overlay or Overlay(enabled=T.BUBBLE_OVERLAY)
        self.tap = HotkeyTap(T.BUBBLE_HOTKEY, on_fire=self._on_hotkey, on_escape=self._on_escape,
                             on_user_key=self._on_user_key, busy=self.busy)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._cancel_reason = ""
        self._bg: set[asyncio.Task] = set()
        self.stats = {"triggers": 0, "completed": 0, "empty": 0, "cancelled": 0, "failed": 0}

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.overlay.start()  # warm: the first trigger shows the pill in a frame
        if not ax.accessibility_granted():
            log.warning("bubble: Accessibility is not granted; the hotkey cannot be armed or fields read")
        if self.tap.start():
            log.info("bubble armed on %s (Esc or typing cancels)", T.BUBBLE_HOTKEY)
        else:
            log.warning("bubble: hotkey could not be armed")
        self._spawn(self._warm())

    async def stop(self) -> None:
        self.tap.stop()
        if self._task and not self._task.done():
            self._task.cancel()
        tasks = [t for t in [self._task, *self._bg] if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.overlay.close()
        log.info("bubble stats: %s", self.stats)

    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def _spawn(self, coro) -> None:
        spawn(coro, owner=self._bg)

    async def _warm(self) -> None:
        """Open the API connection now so the first trigger skips the TLS handshake."""
        try:
            with interactive():
                await embed(["warm"])
        except (NotConfigured, *EXPECTED_ERRORS):
            pass

    # ------------------------------------------------------------ tap thread

    def _on_hotkey(self) -> None:
        if self.busy() or self._loop is None:
            return
        snap = ax.capture_focus()  # synchronous: before the user's attention can move
        self._loop.call_soon_threadsafe(self._launch, snap)

    def _on_escape(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._cancel, "escape")

    def _on_user_key(self) -> None:
        if T.BUBBLE_CANCEL_ON_TYPE and self._loop:
            self._loop.call_soon_threadsafe(self._cancel, "typing")

    # ------------------------------------------------------------ event loop

    def _launch(self, snap: ax.FocusSnapshot) -> None:
        if self.busy():
            return
        self._cancel_reason = ""
        self._task = asyncio.create_task(self._run(snap), name="bubble-run")

    def _cancel(self, reason: str) -> None:
        if self._task is not None and self.busy() and not self._cancel_reason:
            self._cancel_reason = reason
            self._task.cancel()

    async def _watch_focus(self, pid: int) -> None:
        while True:
            await asyncio.sleep(T.BUBBLE_FOCUS_POLL_S)
            cur = ax.frontmost_pid()
            if cur and cur != pid:
                return

    async def _read_field(self, snap: ax.FocusSnapshot) -> tuple[str, str]:
        """(text up to the caret, field description): AX only; unapproved pixels never leave the device."""
        if snap.has_text:
            return snap.text_to_caret, snap.description or snap.role or "text field"
        # Vision is disabled until each outgoing image has local privacy approval.
        return "", ""

    async def _run(self, snap: ax.FocusSnapshot) -> None:
        self.stats["triggers"] += 1
        t0 = time.perf_counter()
        hold()
        watcher = asyncio.create_task(self._watch_focus(snap.pid))
        self.overlay.show(snap.caret)
        typed = ""
        try:
            if denied(snap.app, url=snap.url):
                # DENIED_APPS / DENIED_SITES: the field is not read, nor is the screen.
                self.overlay.message("Bubble is off here", tone="warn")
                return
            with PERF.timer("bubble.field_read"):
                text, desc = await self._read_field(snap)
            if T.REDACT_SECRETS:
                text = redact(text).text  # the field goes to the planner and the writer
            if not field_query(text):
                self.overlay.message("No text in this field", tone="warn")
                return

            def progress(stage: str, detail) -> None:
                if stage == "searching":
                    self.overlay.status("Searching your memory")
                elif stage == "ranking" and detail:
                    self.overlay.status(f"Ranking {detail} moments")

            self.overlay.status("Understanding")
            p = await self.engine.prepare(text, app=snap.app, field_desc=desc, progress=progress)
            if watcher.done():
                self.overlay.message("Focus moved — stopped", tone="muted")
                self.stats["cancelled"] += 1
                return
            n = len(p.result.hits)
            log.info("bubble: mode=%s intent=%s hits=%d retrieve=%.2fs query=%r", p.mode, p.result.plan.intent,
                     n, p.timings["retrieve_s"], "[encrypted store]" if getattr(
                         self.store.conn, "_discrete_encrypted", False) else p.query[:80])
            if p.mode == "empty":
                self.stats["empty"] += 1
                self.overlay.message("Nothing on file about that", tone="muted")
                self._spawn(self._record_gap(p))
                return
            if p.mode == "thin":
                self._spawn(self._record_gap(p))
            self.overlay.status(f"Writing from {n} source{'s' if n != 1 else ''}" if n else "Writing")
            snap.restore()
            typed = await self.engine.write(p, emit=lambda s: type_text(s, pid=snap.pid),
                                            should_continue=lambda: not watcher.done())
            if typed:
                self.stats["completed"] += 1
                PERF.sample("bubble.total", since_ms(t0), n=len(typed))  # hotkey → last word typed
                self.overlay.done(f"Done · {time.perf_counter() - t0:.1f}s")
                log.info("bubble: typed %d chars; first chunk %.2fs after retrieval, total %.2fs", len(typed),
                         p.timings.get("first_chunk_s", 0.0), time.perf_counter() - t0)
            else:
                self.overlay.message("Nothing to add here", tone="muted")
        except EmptyField:
            self.overlay.message("No text in this field", tone="warn")
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            self.overlay.message("Stopped" if self._cancel_reason == "escape" else "Stopped — you kept typing",
                                 tone="muted")
        except (NotConfigured, *EXPECTED_ERRORS) as exc:
            self.stats["failed"] += 1
            log.warning("bubble failed: %s", exc)
            self.overlay.message("Couldn't reach the model", tone="warn")
        except Exception:
            self.stats["failed"] += 1
            log.exception("bubble crashed")
            self.overlay.message("Something went wrong", tone="warn")
        finally:
            watcher.cancel()
            if typed:
                note_agent_output(typed)
            release()

    async def _record_gap(self, p) -> None:
        try:
            out = await maybe_record_gap(self.store, p.query, result=p.result, field_text=p.field_text)
            log.info("bubble gap check: %s %s", out.kind, out.missed or out.gap_id or "")
        except Exception:
            log.exception("gap check failed")
