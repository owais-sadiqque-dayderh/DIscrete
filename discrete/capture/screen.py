"""Screen observer — debounced before/after capture with AX-first text."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from pathlib import Path

import imagehash
import mss
import numpy as np
from PIL import Image

from .. import textkit as tk
from ..config import T, caption_endpoint, frames_dir
from ..llm import EXPECTED_ERRORS, LLMError, complete_json, image_part
from ..store import ObservationDraft
from . import accessibility as ax
from . import ocr
from .browser import browser_url
from .prompts import CAPTURE_PROMPT
from .session import SessionTracker
from .suppress import classify_origin, is_suppressed

log = logging.getLogger("discrete.capture.screen")


class ScreenObserver:
    def __init__(self, *, queue: asyncio.Queue, session: SessionTracker | None = None):
        self.queue = queue
        self.session = session or SessionTracker()
        self.dir = frames_dir()
        self.ep = caption_endpoint()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listeners: list = []
        self._pending: dict | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._before_ctx: ax.WindowContext | None = None
        self._before_path: str | None = None
        self._last_event_ts = 0.0
        self._stats = {"captured": 0, "dropped_failed": 0, "dropped_useless": 0, "degraded": 0, "suppressed": 0, "aborted_drift": 0}
        self._warned_blank = False
        self._capture_lock = asyncio.Lock()
        self._latest_frame = None
        self._frame_task: asyncio.Task | None = None
        self._interaction = {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0}
        self._last_phash: str = ""
        self._last_full_capture = 0.0
        self._last_screen_text = ""

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        ax.warn_permissions()
        from pynput import keyboard, mouse

        def bump(kind: str) -> None:
            if kind == "keyboard":
                self._interaction["keys"] += 1
            elif kind == "mouse":
                self._interaction["clicks"] += 1
            elif kind == "scroll":
                self._interaction["scrolls"] += 1
            elif kind == "clipboard":
                self._interaction["clipboard"] += 1

        def on_mouse(*_a, **_k):
            bump("mouse")
            self._schedule("mouse")

        def on_scroll(*_a, **_k):
            bump("scroll")
            self._schedule("mouse")

        def on_key(*_a, **_k):
            if T.KEYBOARD_CAPTURE:
                bump("keyboard")
                self._schedule("keyboard")

        ml = mouse.Listener(on_move=lambda x, y: on_mouse(), on_click=lambda x, y, b, p: on_mouse() if p else None, on_scroll=lambda x, y, dx, dy: on_scroll())
        ml.daemon = True
        ml.start()
        self._listeners.append(ml)
        if T.KEYBOARD_CAPTURE:
            kl = keyboard.Listener(on_press=lambda k: on_key())
            kl.daemon = True
            kl.start()
            self._listeners.append(kl)
        self._frame_task = asyncio.create_task(self._frame_loop(), name="discrete-frames")
        log.info("screen observer armed")

    async def stop(self) -> None:
        if self._frame_task:
            self._frame_task.cancel()
            try:
                await self._frame_task
            except asyncio.CancelledError:
                pass
        for l in self._listeners:
            try:
                l.stop()
            except Exception:
                pass
        self._listeners.clear()
        if self._timer:
            self._timer.cancel()

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _schedule(self, kind: str) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._arm, kind)

    def _arm(self, kind: str) -> None:
        now = time.monotonic()
        if now - self._last_event_ts > T.SESSION_GAP_S:
            self.session.new_session()
        self._last_event_ts = now
        if is_suppressed():
            self._stats["suppressed"] += 1
            return
        delay = T.KEY_IDLE_DEBOUNCE_S if kind == "keyboard" else T.MOUSE_DEBOUNCE_S
        if self._pending is None:
            self._before_ctx = ax.window_context()
            self._before_path = self._save_frame(self._latest_frame, "before")
            self._pending = {"kind": kind, "started": time.time(), "interaction": dict(self._interaction)}
        else:
            self._pending["kind"] = "keyboard" if kind == "keyboard" else self._pending["kind"]
        if self._timer:
            self._timer.cancel()
        self._timer = self._loop.call_later(delay, lambda: asyncio.create_task(self._flush()))

    async def _flush(self) -> None:
        async with self._capture_lock:
            pending = self._pending
            self._pending = None
            if not pending:
                return
            after_ctx = ax.window_context()
            if self._before_ctx and after_ctx.app and self._before_ctx.app and after_ctx.app != self._before_ctx.app:
                self._stats["aborted_drift"] += 1
                return
            after_path = self._save_frame(self._latest_frame, "after")
            paths = [p for p in (self._before_path, after_path) if p]
            await self._process_capture(pending, after_ctx, paths)
            self._interaction = {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0}

    async def _frame_loop(self) -> None:
        with mss.mss() as sct:
            while True:
                try:
                    shot = sct.grab(sct.monitors[1])
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    if img.width > T.FRAME_MAX_WIDTH:
                        h = int(img.height * T.FRAME_MAX_WIDTH / img.width)
                        img = img.resize((T.FRAME_MAX_WIDTH, h), Image.Resampling.LANCZOS)
                    self._latest_frame = img
                except Exception as exc:
                    log.debug("frame loop: %s", exc)
                await asyncio.sleep(0.2)

    def _save_frame(self, img: Image.Image | None, tag: str) -> str | None:
        if img is None:
            return None
        arr = np.array(img.convert("L"))
        if arr.std() < 2.0 and not self._warned_blank:
            self._warned_blank = True
            log.warning("near-flat frame — check Screen Recording permission")
        ts = int(time.time() * 1000)
        path = self.dir / f"{ts}_{tag}.jpg"
        img.save(path, "JPEG", quality=T.FRAME_JPEG_QUALITY)
        return str(path)

    async def heartbeat_tick(self) -> None:
        if is_suppressed():
            return
        ctx = ax.window_context()
        screen_text = ax.read_screen_text(max_nodes=T.AX_WALK_MAX_NODES, timeout_ms=T.AX_WALK_TIMEOUT_MS)
        text_change = 1.0 - tk.jaccard(tk.trigrams(screen_text), tk.trigrams(self._last_screen_text))
        since_full = time.time() - self._last_full_capture
        force_full = (
            text_change >= T.HEARTBEAT_TEXT_CHANGE and since_full >= T.HEARTBEAT_MIN_GAP_S
        ) or since_full >= T.HEARTBEAT_MAX_GAP_S
        if force_full:
            self._schedule("heartbeat")
            return
        if screen_text and screen_text != self._last_screen_text:
            draft = ObservationDraft(
                source="heartbeat",
                origin=classify_origin(ctx.app, screen_text),
                event_kind="heartbeat",
                session_id=self.session.session_id,
                started_at=time.time(),
                ended_at=time.time(),
                app=ctx.app,
                bundle_id=ctx.bundle_id,
                window_title=ctx.window_title,
                url=self._resolve_url(ctx),
                site=tk.site_from_url(self._resolve_url(ctx)),
                screen_text=screen_text,
                text_source="ax",
                caption=f"{ctx.app}: {ctx.window_title}".strip(": "),
            )
            await self.queue.put(draft)
        self._last_screen_text = screen_text

    def _resolve_url(self, ctx: ax.WindowContext) -> str:
        return browser_url(ctx.app) or ctx.url or ""

    async def _process_capture(self, pending: dict, ctx: ax.WindowContext, paths: list[str]) -> None:
        ctx_url = self._resolve_url(ctx)
        if ctx_url:
            ctx = ax.WindowContext(
                app=ctx.app,
                bundle_id=ctx.bundle_id,
                window_title=ctx.window_title,
                url=ctx_url,
                pid=ctx.pid,
            )
        started = pending["started"]
        ended = time.time()
        screen_text = ax.read_screen_text(max_nodes=T.AX_WALK_MAX_NODES, timeout_ms=T.AX_WALK_TIMEOUT_MS)
        text_source = "ax"
        after_jpeg = b""
        if paths:
            with open(paths[-1], "rb") as f:
                after_jpeg = f.read()
        app_l = (ctx.app or "").lower()
        if len(screen_text) < T.AX_TEXT_MIN_CHARS or any(c in app_l for c in T.CANVAS_APPS):
            ocr_text = ocr.ocr_jpeg(after_jpeg)
            if ocr_text:
                screen_text = tk.truncate(f"{screen_text}\n{ocr_text}".strip(), T.SCREEN_TEXT_MAX_CHARS)
                text_source = "ax+ocr" if text_source == "ax" else "ocr"
        screen_text = tk.truncate(screen_text, T.SCREEN_TEXT_MAX_CHARS)
        phash = ""
        if after_jpeg:
            try:
                phash = str(imagehash.phash(Image.open(io.BytesIO(after_jpeg))))
                if self._last_phash and phash:
                    dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(self._last_phash)
                    if dist <= T.FRAME_PHASH_MAX_DIST and ctx.app == (self._before_ctx.app if self._before_ctx else ctx.app):
                        log.debug("phash duplicate, skipping VLM")
                        draft = self._light_draft(pending, ctx, screen_text, text_source, phash, paths, started, ended)
                        await self.queue.put(draft)
                        return
                self._last_phash = phash
            except Exception:
                pass
        inter = pending.get("interaction", {})
        vlm = await self._vlm(paths, ctx, screen_text, inter)
        if vlm is None:
            self._stats["degraded"] += 1
            draft = self._light_draft(pending, ctx, screen_text, text_source, phash, paths, started, ended)
            await self.queue.put(draft)
            return
        if not vlm.get("useful", True):
            self._stats["dropped_useless"] += 1
            return
        caption = str(vlm.get("caption") or "").strip()
        if len(screen_text) < 40 and len(caption) < 5:
            self._stats["dropped_useless"] += 1
            return
        entities = [str(e) for e in (vlm.get("entities") or []) if str(e).strip()]
        origin = classify_origin(ctx.app, screen_text)
        draft = ObservationDraft(
            source="screen",
            origin=origin,
            event_kind=pending["kind"],
            session_id=self.session.session_id,
            started_at=started,
            ended_at=ended,
            app=ctx.app,
            bundle_id=ctx.bundle_id,
            window_title=ctx.window_title,
            url=ctx.url,
            site=tk.site_from_url(ctx.url),
            screen_text=screen_text,
            text_source=text_source,
            caption=caption,
            entities=entities,
            keys=inter.get("keys", 0),
            clicks=inter.get("clicks", 0),
            scrolls=inter.get("scrolls", 0),
            clipboard=inter.get("clipboard", 0),
            frame_phash=phash,
            frame_path=paths[-1] if paths else "",
        )
        self._stats["captured"] += 1
        self._last_full_capture = time.time()
        self._last_screen_text = screen_text
        await self.queue.put(draft)

    def _light_draft(self, pending, ctx, screen_text, text_source, phash, paths, started, ended) -> ObservationDraft:
        url = self._resolve_url(ctx)
        caption = f"{ctx.app}: {ctx.window_title}".strip(": ")
        return ObservationDraft(
            source="screen",
            origin=classify_origin(ctx.app, screen_text),
            event_kind=pending["kind"],
            session_id=self.session.session_id,
            started_at=started,
            ended_at=ended,
            app=ctx.app,
            bundle_id=ctx.bundle_id,
            window_title=ctx.window_title,
            url=url,
            site=tk.site_from_url(url),
            screen_text=screen_text,
            text_source=text_source,
            caption=caption,
            frame_phash=phash,
            frame_path=paths[-1] if paths else "",
            keys=pending.get("interaction", {}).get("keys", 0),
            clicks=pending.get("interaction", {}).get("clicks", 0),
            scrolls=pending.get("interaction", {}).get("scrolls", 0),
            clipboard=pending.get("interaction", {}).get("clipboard", 0),
        )

    async def _vlm(self, paths: list[str], ctx: ax.WindowContext, screen_text: str, inter: dict) -> dict | None:
        if not self.ep.configured:
            return None
        imgs: list[dict] = []
        for p in paths[-2:]:
            try:
                with open(p, "rb") as f:
                    imgs.append(image_part(f.read()))
            except OSError:
                pass
        prompt = CAPTURE_PROMPT.format(
            app=ctx.app,
            window_title=ctx.window_title,
            url=ctx.url,
            screen_text=tk.truncate(screen_text, 4000),
            keys=inter.get("keys", 0),
            clicks=inter.get("clicks", 0),
            scrolls=inter.get("scrolls", 0),
            clipboard=inter.get("clipboard", 0),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *imgs]}]
        for attempt in range(T.CAPTURE_MAX_RETRIES):
            try:
                data = await complete_json(self.ep, messages, max_tokens=800)
                if isinstance(data, dict):
                    return data
            except EXPECTED_ERRORS as exc:
                log.warning("VLM attempt %s failed: %s", attempt + 1, exc)
                await asyncio.sleep(T.CAPTURE_RETRY_BASE_S * (2**attempt))
        self._stats["dropped_failed"] += 1
        return None
