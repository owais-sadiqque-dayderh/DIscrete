"""Screen observer: debounced before/after capture of the focused window (planS1 §4-6).

Interaction counting happens on the event loop thread inside `arm()`, so the
counts always describe exactly the pending capture's interval (insights.md §7.14).

The capture lock covers only local work (frame hash, AX walk, OCR). The vision
call happens later in capture/enrich.py: every capture is queued at once with a
basic caption, so a slow model never holds up the next capture.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any

import imagehash
from PIL import Image

from .. import textkit as tk
from ..aio import spawn
from ..config import T, caption_endpoint, frames_dir
from ..models import ObservationDraft
from ..perf import PERF, since_ms
from . import accessibility as ax
from . import ocr
from .browser import browser_url
from .enrich import Capture, EnrichJob
from .frames import FrameGrabber, is_blank
from .session import SessionTracker
from .suppress import classify_origin, is_suppressed

log = logging.getLogger("discrete.capture.screen")

_COUNTED = {"click": "clicks", "scroll": "scrolls", "keyboard": "keys", "clipboard": "clipboard"}


def _novel_lines(ocr_text: str, ax_text: str) -> list[str]:
    """OCR lines carrying words the AX text does not already have."""
    known = set(tk.content_tokens(ax_text))
    out: list[str] = []
    for line in dict.fromkeys(x.strip() for x in ocr_text.splitlines()):
        toks = set(tk.content_tokens(line))
        if toks and len(toks - known) / len(toks) >= 0.5:
            out.append(line)
    return out


class ScreenObserver:
    def __init__(self, *, queue: asyncio.Queue, session: SessionTracker | None = None):
        self.queue = queue
        self.session = session or SessionTracker()
        self.dir = frames_dir()
        self.ep = caption_endpoint()
        self.grabber = FrameGrabber()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listeners: list = []
        self._tasks: set[asyncio.Task] = set()
        self._pending: dict | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._capture_lock = asyncio.Lock()
        self._frame_task: asyncio.Task | None = None
        self._latest: tuple[Image.Image, ax.WindowContext] | None = None
        # Last few cached frames: a rotated capture must use a frame of *its* window, and
        # the poller notices a navigation up to POLL_S after the frame cache already has the
        # next page (insights.md §7.37).
        self._recent: collections.deque = collections.deque(maxlen=6)
        self._last_event_mono = time.monotonic()
        # (app, title, url) -> phash of the last kept frame; least recently used evicted.
        self._last_kept: collections.OrderedDict[tuple, str] = collections.OrderedDict()
        self._last_full_capture = 0.0
        self._last_full_text = ""
        self._warned_blank = False
        self.away = False
        self.stats = {k: 0 for k in ("captured", "light", "frame_dup", "dropped_failed", "degraded",
                                     "suppressed", "aborted_drift", "self")}

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        ax.warn_permissions()
        from pynput import keyboard, mouse

        # Mouse *moves* are neither triggers nor interaction (insights.md §7.14).
        ml = mouse.Listener(
            on_click=lambda x, y, b, pressed: self.trigger("click") if pressed else None,
            on_scroll=lambda x, y, dx, dy: self.trigger("scroll"),
        )
        ml.daemon = True
        ml.start()
        self._listeners.append(ml)
        if T.KEYBOARD_CAPTURE:
            kl = keyboard.Listener(on_press=lambda k: self.trigger("keyboard"))
            kl.daemon = True
            kl.start()
            self._listeners.append(kl)
        self._frame_task = asyncio.create_task(self._frame_loop(), name="discrete-frames")
        log.info("screen observer armed")

    async def stop(self) -> None:
        self._loop = None
        if self._frame_task:
            self._frame_task.cancel()
            await asyncio.gather(self._frame_task, return_exceptions=True)
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception:
                pass
        self._listeners.clear()
        if self._timer:
            self._timer.cancel()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.grabber.close()

    # -------------------------------------------------------------- triggers

    def trigger(self, kind: str) -> None:
        """Thread-safe entry point for every trigger source."""
        if self._loop:
            self._loop.call_soon_threadsafe(self.arm, kind)

    def arm(self, kind: str) -> None:
        if self._loop is None:
            return  # a listener callback queued before stop()
        mono = time.monotonic()
        if mono - self._last_event_mono > T.SESSION_GAP_S:
            self.session.new_session()
        self._last_event_mono = mono
        if is_suppressed():
            self.stats["suppressed"] += 1
            return
        # A navigation to a different window/app is a capture boundary: finalize the
        # article in flight *now* and open a fresh pair, instead of letting the debounce
        # coalesce a whole browse into one before/after (planS1 §5, insights.md §7.32).
        cur = None
        if kind in ("app_switch", "window_change"):
            cur = ax.window_context()
            if self._pending is not None and cur.pid and cur.key != self._pending["before_ctx"].key:
                self._rotate()
        if self._pending is None:
            before = self._latest
            if cur is not None and before is not None and before[1].key != cur.key:
                before = None  # the cached frame shows the page we just left, not this one
            self._pending = {
                "kind": kind,
                "started": time.time(),
                "before_img": before[0] if before else None,
                # A navigation trigger anchors the pair to the window we just landed on;
                # other triggers keep the cached context of the frame already in hand.
                "before_ctx": cur or (before[1] if before else ax.window_context()),
                "counts": {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0},
            }
        elif kind == "keyboard":
            self._pending["kind"] = "keyboard"
        field = _COUNTED.get(kind)
        if field:
            self._pending["counts"][field] += 1
        delay = T.KEY_IDLE_DEBOUNCE_S if self._pending["kind"] == "keyboard" else T.MOUSE_DEBOUNCE_S
        if self._timer:
            self._timer.cancel()
        assert self._loop is not None, "arm() before start()"
        self._timer = self._loop.call_later(delay, self._flush_soon)

    def _flush_soon(self) -> None:
        spawn(self._flush(), name="discrete-flush", owner=self._tasks)

    # ---------------------------------------------------------------- frames

    async def _frame_loop(self) -> None:
        while True:
            try:
                ctx = await asyncio.to_thread(ax.window_context)
                img = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH, window_id=ctx.window_id)
                if img is not None:
                    self._latest = (img, ctx)
                    self._recent.append((img, ctx))
                    if not self._warned_blank and is_blank(img):
                        self._warned_blank = True
                        log.warning("near-flat frame — check Screen Recording permission")
            except Exception as exc:
                log.debug("frame loop: %s", exc)
            await asyncio.sleep(T.FRAME_CACHE_S)

    def _save(self, img: Image.Image | None, tag: str) -> str:
        if img is None:
            return ""
        path = self.dir / f"{int(time.time() * 1000)}_{tag}.jpg"
        img.save(path, "JPEG", quality=T.FRAME_JPEG_QUALITY)
        return str(path)

    # ----------------------------------------------------------------- flush

    def _rotate(self) -> None:
        """Finalize the in-flight pair immediately at a navigation boundary. The
        window has already moved on, so the previous article's `after` is its last
        cached frame and its text is read from that image, not from the live (now
        different) window (planS1 §5, insights.md §7.32)."""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        pending, self._pending = self._pending, None
        if not pending:
            return
        ctx: ax.WindowContext = pending["before_ctx"]
        own = next((img for img, c in reversed(self._recent) if c.key == ctx.key), None)
        after = own if own is not None else pending["before_img"]
        if after is None:
            self.stats["dropped_failed"] += 1  # degraded: no frame, but still segment the boundary
        spawn(self._finalize(pending, ctx, after, stale=True), name="discrete-rotate", owner=self._tasks)

    async def _flush(self) -> None:
        pending, self._pending = self._pending, None
        if not pending:
            return
        ctx = await asyncio.to_thread(ax.window_context)
        before_ctx: ax.WindowContext = pending["before_ctx"]
        if before_ctx.pid and ctx.pid and before_ctx.pid != ctx.pid:
            # Focus drift: a pair spanning two apps is not one moment (planS1 §5).
            self.stats["aborted_drift"] += 1
            self.arm("app")
            return
        after = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH, window_id=ctx.window_id)
        await self._finalize(pending, ctx, after, stale=False)

    async def _finalize(self, pending: dict, ctx: ax.WindowContext, after, *, stale: bool) -> None:
        async with self._capture_lock:
            await self._capture(pending, ctx, pending["before_img"], after, stale=stale)

    async def _capture(self, pending: dict, ctx: ax.WindowContext, before, after, *,
                       heartbeat: bool = False, stale: bool = False) -> None:
        # A stale (rotated) capture describes a window we already left; its live AX/URL
        # would read the new window, so trust the context frozen at arm time instead.
        # The tab URL comes from the web view (window_context); AppleScript is a slow fallback.
        t0 = time.perf_counter()
        url = ctx.url or ("" if stale else await asyncio.to_thread(browser_url, ctx.app))
        counts: dict[str, Any] = pending["counts"]
        started, ended = pending["started"], time.time()
        phash = str(imagehash.phash(after)) if after is not None else ""
        sig = (ctx.app, ctx.window_title, url)
        prev = self._last_kept.get(sig)
        if phash and prev and imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(prev) <= T.FRAME_PHASH_MAX_DIST:
            self.stats["frame_dup"] += 1
            await self.queue.put(Capture(ObservationDraft(
                source="screen", event_kind=pending["kind"], session_id=self.session.session_id,
                started_at=started, ended_at=ended, app=ctx.app, bundle_id=ctx.bundle_id,
                window_title=ctx.window_title, url=url, frame_duplicate=True, **counts,
            )))
            return
        self._remember(sig, phash)

        t_text = time.perf_counter()
        screen_text, text_source = await asyncio.to_thread(self._screen_text, ctx, after, stale)
        PERF.sample("capture.text", since_ms(t_text), n=len(screen_text))
        origin = classify_origin(ctx.app, ctx.window_title, screen_text)
        base: dict[str, Any] = dict(
            origin=origin, event_kind=pending["kind"], session_id=self.session.session_id,
            started_at=started, ended_at=ended, app=ctx.app, bundle_id=ctx.bundle_id,
            window_title=ctx.window_title, url=url, site=tk.site_from_url(url),
            screen_text=screen_text, text_source=text_source, frame_phash=phash, **counts,
        )
        light_caption = f"{ctx.app}: {ctx.window_title}".strip(": ")
        if origin != "user":
            # The system's own output is stored for audit only; no model call.
            self.stats["self"] += 1
            await self.queue.put(Capture(ObservationDraft(source="screen", caption=light_caption, **base)))
            return

        text_change = 1.0 - tk.jaccard(tk.trigrams(screen_text), tk.trigrams(self._last_full_text))
        since_full = time.time() - self._last_full_capture
        if heartbeat and not (
            (text_change >= T.HEARTBEAT_TEXT_CHANGE and since_full >= T.HEARTBEAT_MIN_GAP_S)
            or since_full >= T.HEARTBEAT_MAX_GAP_S
        ):
            self.stats["light"] += 1
            await self.queue.put(Capture(ObservationDraft(source="heartbeat", caption=light_caption, **base)))
            return

        after_path = await asyncio.to_thread(self._save, after, "after")
        self._last_full_capture = time.time()
        self._last_full_text = screen_text
        if not self.ep.configured:
            self.stats["degraded"] += 1
            await self.queue.put(Capture(ObservationDraft(
                source="degraded", caption=light_caption, frame_path=after_path, **base)))
            return
        # Stored now with the basic caption; the vision caption follows (enrich.py).
        self.stats["captured"] += 1
        job = EnrichJob(frames=[f for f in (before, after) if f is not None], app=ctx.app,
                        window_title=ctx.window_title, url=url, screen_text=screen_text, counts=dict(counts))
        PERF.sample("capture.local", since_ms(t0))  # hash, text, frame save: all work before the queue
        await self.queue.put(Capture(ObservationDraft(
            source="screen", caption=light_caption, frame_path=after_path, enrich_state="pending", **base), job))

    def _remember(self, sig: tuple, phash: str) -> None:
        self._last_kept[sig] = phash
        self._last_kept.move_to_end(sig)
        while len(self._last_kept) > T.LAST_KEPT_MAX:
            self._last_kept.popitem(last=False)

    def _screen_text(self, ctx: ax.WindowContext, after, stale: bool = False) -> tuple[str, str]:
        # For a stale (rotated) capture the live AX tree is the *new* window, so the
        # only faithful text for the old article is OCR of its captured frame.
        text = "" if stale else ax.read_screen_text(
            ctx.pid, max_nodes=T.AX_WALK_MAX_NODES, timeout_ms=T.AX_WALK_TIMEOUT_MS,
            max_chars=T.SCREEN_TEXT_MAX_CHARS, viewport=ctx.bounds)
        # A long AX tree can still be only chrome while the document is a scanned page
        # or a canvas, so its length cannot establish coverage: OCR every frame
        # (~0.2-0.3 s, off the event loop). Only OCR lines that AX lacks are kept, and
        # they go first so a long chrome tree cannot truncate them away; on an ordinary
        # page that is a few lines, and the exact AX text follows.
        ocr_lines = _novel_lines(ocr.ocr_image(after), text)
        if not ocr_lines:
            return tk.truncate(text, T.SCREEN_TEXT_MAX_CHARS), "ax" if text else ""
        merged = "\n".join(ocr_lines + ([text] if text else []))
        return tk.truncate(merged, T.SCREEN_TEXT_MAX_CHARS), "ax+ocr" if text else "ocr"

    # ------------------------------------------------------------- heartbeat

    async def heartbeat_tick(self) -> None:
        """Passive-session coverage (planS1 §6.4). Skipped while away or mid-capture."""
        if self.away or is_suppressed() or self._pending is not None or self._capture_lock.locked():
            return
        async with self._capture_lock:
            ctx = await asyncio.to_thread(ax.window_context)
            img = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH, window_id=ctx.window_id)
            if img is None:
                return
            pending = {"kind": "heartbeat", "started": time.time(),
                       "counts": {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0}}
            await self._capture(pending, ctx, None, img, heartbeat=True)
