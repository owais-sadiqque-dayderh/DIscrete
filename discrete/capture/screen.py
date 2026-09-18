"""Screen observer: debounced before/after capture of the focused window (planS1 §4-6).

Interaction counting happens on the event loop thread inside `arm()`, so the
counts always describe exactly the pending capture's interval (insights.md §7.14).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

import imagehash
from PIL import Image

from .. import textkit as tk
from ..config import T, caption_endpoint, frames_dir
from ..llm import EXPECTED_ERRORS, complete_json, image_part
from ..store import ObservationDraft
from . import accessibility as ax
from . import ocr
from .browser import browser_url
from .frames import FrameGrabber, downscale, is_blank
from .prompts import CAPTURE_PROMPT
from .session import SessionTracker
from .suppress import classify_origin, is_suppressed

log = logging.getLogger("discrete.capture.screen")

_COUNTED = {"click": "clicks", "scroll": "scrolls", "keyboard": "keys", "clipboard": "clipboard"}


class ScreenObserver:
    def __init__(self, *, queue: asyncio.Queue, session: SessionTracker | None = None):
        self.queue = queue
        self.session = session or SessionTracker()
        self.dir = frames_dir()
        self.ep = caption_endpoint()
        self.grabber = FrameGrabber()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listeners: list = []
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
        self._last_kept: dict[tuple, str] = {}  # (app, title, url) -> phash of last kept frame
        self._last_full_capture = 0.0
        self._last_full_text = ""
        self._warned_blank = False
        self.away = False
        self.stats = {k: 0 for k in ("captured", "light", "frame_dup", "dropped_failed", "dropped_useless",
                                     "degraded", "suppressed", "aborted_drift", "self")}

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
        self.grabber.close()

    # -------------------------------------------------------------- triggers

    def trigger(self, kind: str) -> None:
        """Thread-safe entry point for every trigger source."""
        if self._loop:
            self._loop.call_soon_threadsafe(self.arm, kind)

    def arm(self, kind: str) -> None:
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
        self._timer = self._loop.call_later(delay, lambda: asyncio.create_task(self._flush()))

    # ---------------------------------------------------------------- frames

    async def _frame_loop(self) -> None:
        while True:
            try:
                ctx = await asyncio.to_thread(ax.window_context)
                img = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH)
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
        asyncio.create_task(self._finalize(pending, ctx, after, stale=True))

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
        after = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH)
        await self._finalize(pending, ctx, after, stale=False)

    async def _finalize(self, pending: dict, ctx: ax.WindowContext, after, *, stale: bool) -> None:
        async with self._capture_lock:
            await self._capture(pending, ctx, pending["before_img"], after, stale=stale)

    async def _capture(self, pending: dict, ctx: ax.WindowContext, before, after, *,
                       heartbeat: bool = False, stale: bool = False) -> None:
        # A stale (rotated) capture describes a window we already left; its live AX/URL
        # would read the new window, so trust the context frozen at arm time instead.
        # The tab URL comes from the web view (window_context); AppleScript is a slow fallback.
        url = ctx.url or ("" if stale else await asyncio.to_thread(browser_url, ctx.app))
        counts = pending["counts"]
        started, ended = pending["started"], time.time()
        phash = str(imagehash.phash(after)) if after is not None else ""
        sig = (ctx.app, ctx.window_title, url)
        prev = self._last_kept.get(sig)
        if phash and prev and imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(prev) <= T.FRAME_PHASH_MAX_DIST:
            self.stats["frame_dup"] += 1
            await self.queue.put(ObservationDraft(
                source="screen", event_kind=pending["kind"], session_id=self.session.session_id,
                started_at=started, ended_at=ended, app=ctx.app, bundle_id=ctx.bundle_id,
                window_title=ctx.window_title, url=url, frame_duplicate=True, **counts,
            ))
            return
        self._last_kept[sig] = phash

        screen_text, text_source = await asyncio.to_thread(self._screen_text, ctx, after, stale)
        origin = classify_origin(ctx.app, ctx.window_title, screen_text)
        base = dict(
            origin=origin, event_kind=pending["kind"], session_id=self.session.session_id,
            started_at=started, ended_at=ended, app=ctx.app, bundle_id=ctx.bundle_id,
            window_title=ctx.window_title, url=url, site=tk.site_from_url(url),
            screen_text=screen_text, text_source=text_source, frame_phash=phash, **counts,
        )
        light_caption = f"{ctx.app}: {ctx.window_title}".strip(": ")
        if origin != "user":
            # The system's own output is stored for audit only; no model call.
            self.stats["self"] += 1
            await self.queue.put(ObservationDraft(source="screen", caption=light_caption, **base))
            return

        text_change = 1.0 - tk.jaccard(tk.trigrams(screen_text), tk.trigrams(self._last_full_text))
        since_full = time.time() - self._last_full_capture
        if heartbeat and not (
            (text_change >= T.HEARTBEAT_TEXT_CHANGE and since_full >= T.HEARTBEAT_MIN_GAP_S)
            or since_full >= T.HEARTBEAT_MAX_GAP_S
        ):
            self.stats["light"] += 1
            await self.queue.put(ObservationDraft(source="heartbeat", caption=light_caption, **base))
            return

        after_path = self._save(after, "after")
        vlm = await self._vlm([before, after], ctx, url, screen_text, counts)
        self._last_full_capture = time.time()
        self._last_full_text = screen_text
        if vlm is None:
            self.stats["degraded"] += 1
            await self.queue.put(ObservationDraft(source="degraded", caption=light_caption, frame_path=after_path, **base))
            return
        caption = str(vlm.get("caption") or "").strip()
        if not vlm.get("useful", True) or (len(screen_text) < 40 and len(caption) < 5):
            self.stats["dropped_useless"] += 1
            return
        self.stats["captured"] += 1
        await self.queue.put(ObservationDraft(
            source="screen", caption=caption or light_caption,
            entities=[str(e).strip() for e in (vlm.get("entities") or []) if str(e).strip()],
            frame_path=after_path, **base,
        ))

    def _screen_text(self, ctx: ax.WindowContext, after, stale: bool = False) -> tuple[str, str]:
        # For a stale (rotated) capture the live AX tree is the *new* window, so the
        # only faithful text for the old article is OCR of its captured frame.
        text = "" if stale else ax.read_screen_text(
            ctx.pid, max_nodes=T.AX_WALK_MAX_NODES, timeout_ms=T.AX_WALK_TIMEOUT_MS,
            max_chars=T.SCREEN_TEXT_MAX_CHARS)
        source = "ax" if text else ""
        canvas = any(c in (ctx.app or "").lower() for c in T.CANVAS_APPS)
        if stale or len(text) < T.AX_TEXT_MIN_CHARS or canvas:
            ocr_text = ocr.ocr_image(after)
            if ocr_text:
                text = f"{text}\n{ocr_text}".strip()
                source = "ax+ocr" if source else "ocr"
        return tk.truncate(text, T.SCREEN_TEXT_MAX_CHARS), source

    async def _vlm(self, frames: list, ctx: ax.WindowContext, url: str, screen_text: str, counts: dict) -> dict | None:
        if not self.ep.configured:
            return None
        imgs = [image_part(_jpeg(downscale(f, T.VLM_MAX_WIDTH)), detail=T.VLM_IMAGE_DETAIL) for f in frames if f is not None]
        prompt = CAPTURE_PROMPT.format(
            app=ctx.app, window_title=ctx.window_title, url=url,
            screen_text=tk.truncate(screen_text, 4000), **counts,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *imgs]}]
        for attempt in range(T.CAPTURE_MAX_RETRIES):
            try:
                data = await complete_json(self.ep, messages, max_tokens=400)
                if isinstance(data, dict):
                    return data
            except EXPECTED_ERRORS as exc:
                log.warning("VLM attempt %s failed: %s", attempt + 1, exc)
            await asyncio.sleep(T.CAPTURE_RETRY_BASE_S * (2**attempt))
        self.stats["dropped_failed"] += 1
        return None

    # ------------------------------------------------------------- heartbeat

    async def heartbeat_tick(self) -> None:
        """Passive-session coverage (planS1 §6.4). Skipped while away or mid-capture."""
        if self.away or is_suppressed() or self._pending is not None or self._capture_lock.locked():
            return
        async with self._capture_lock:
            ctx = await asyncio.to_thread(ax.window_context)
            img = await asyncio.to_thread(self.grabber.grab, ctx.bounds, max_width=T.FRAME_MAX_WIDTH)
            if img is None:
                return
            pending = {"kind": "heartbeat", "started": time.time(),
                       "counts": {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0}}
            await self._capture(pending, ctx, None, img, heartbeat=True)


def _jpeg(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=T.FRAME_JPEG_QUALITY)
    return buf.getvalue()
