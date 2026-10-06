"""Window-cropped frame grabs (planS1 §5).

The grab targets the display containing the focused window and crops to the
window's frame, so menu bars, docks and neighbouring windows (including the
terminal running this daemon) never reach OCR or the VLM.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time

import mss
import numpy as np
from PIL import Image

log = logging.getLogger("discrete.capture.frames")


class FrameGrabber:
    def __init__(self) -> None:
        self._sct = None
        self._windows: dict = {}
        self._windows_at = 0.0
        self._window_lock = threading.Lock()
        self._sck = None
        if sys.platform == "darwin":
            try:
                import ScreenCaptureKit
                from AppKit import NSApplication

                if hasattr(ScreenCaptureKit, "SCScreenshotManager"):
                    # AppKit's WindowServer connection must exist before converting
                    # CGImages in a callback; otherwise NSBitmapImageRep aborts Python.
                    NSApplication.sharedApplication()
                    self._sck = ScreenCaptureKit
            except ImportError:
                pass

    def _mss(self):
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def grab(self, bounds: tuple[float, float, float, float] | None, *, max_width: int,
             window_id: int = 0) -> Image.Image | None:
        """Grab the window region (global points); whole main display if no bounds."""
        if self._sck is not None and window_id:
            with self._window_lock:
                return self._grab_window(window_id, max_width)
        sct = self._mss()
        monitors = sct.monitors[1:] or sct.monitors
        region = None
        if bounds:
            x, y, w, h = bounds
            cx, cy = x + w / 2, y + h / 2
            mon = next(
                (m for m in monitors if m["left"] <= cx < m["left"] + m["width"] and m["top"] <= cy < m["top"] + m["height"]),
                monitors[0],
            )
            left = max(int(x), mon["left"])
            top = max(int(y), mon["top"])
            right = min(int(x + w), mon["left"] + mon["width"])
            bottom = min(int(y + h), mon["top"] + mon["height"])
            if right - left >= 40 and bottom - top >= 40:
                region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        if region is None:
            region = monitors[0]
        try:
            shot = sct.grab(region)
        except Exception as exc:
            log.debug("grab failed: %s", exc)
            return None
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return downscale(img, max_width)

    def _grab_window(self, window_id: int, max_width: int) -> Image.Image | None:
        """Window-isolated ScreenCaptureKit capture, including occluded window content.

        A known window failing capture returns None: substituting the display
        would pair another app's pixels with this app's metadata.
        """
        from AppKit import NSBitmapImageRep, NSPNGFileType

        sc = self._sck
        if sc is None:
            return None
        ready = threading.Event()
        out: list = []
        try:
            if time.monotonic() - self._windows_at > 2 or window_id not in self._windows:
                def content_done(content, error):
                    out.extend([content, error])
                    ready.set()

                sc.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                    True, True, content_done)
                if not ready.wait(2) or not out or out[0] is None:
                    return None
                self._windows = {int(w.windowID()): w for w in out[0].windows()}
                self._windows_at = time.monotonic()
            win = self._windows.get(window_id)
            if win is None:
                return None
            filt = sc.SCContentFilter.alloc().initWithDesktopIndependentWindow_(win)
            cfg = sc.SCStreamConfiguration.alloc().init()
            frame = win.frame()
            width = max(1, min(max_width, int(frame.size.width * 2)))
            cfg.setWidth_(width)
            cfg.setHeight_(max(1, int(width * frame.size.height / max(1, frame.size.width))))
            cfg.setShowsCursor_(False)
            # A distinct event/result prevents a timed-out earlier callback from
            # satisfying this request with the wrong frame.
            shot_ready = threading.Event()
            result: list = []

            def shot_done(image, error):
                try:
                    if image is not None and error is None:
                        rep = NSBitmapImageRep.alloc().initWithCGImage_(image)
                        result.append(bytes(rep.representationUsingType_properties_(NSPNGFileType, {})))
                finally:
                    shot_ready.set()

            sc.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(filt, cfg, shot_done)
            if not shot_ready.wait(2) or not result:
                return None
            with Image.open(io.BytesIO(result[0])) as im:
                return im.convert("RGB")
        except Exception as exc:
            log.debug("window capture failed: %s", exc)
            return None


def downscale(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img
    h = max(1, int(img.height * max_width / img.width))
    return img.resize((max_width, h), Image.Resampling.BILINEAR)


def is_blank(img: Image.Image | None) -> bool:
    if img is None:
        return True
    return float(np.array(img.convert("L")).std()) < 2.0
