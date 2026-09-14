"""Window-cropped frame grabs (planS1 §5).

The grab targets the display containing the focused window and crops to the
window's frame, so menu bars, docks and neighbouring windows (including the
terminal running this daemon) never reach OCR or the VLM.
"""

from __future__ import annotations

import logging

import mss
import numpy as np
from PIL import Image

log = logging.getLogger("discrete.capture.frames")


class FrameGrabber:
    def __init__(self) -> None:
        self._sct = None

    def _mss(self):
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def grab(self, bounds: tuple[float, float, float, float] | None, *, max_width: int) -> Image.Image | None:
        """Grab the window region (global points); whole main display if no bounds."""
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


def downscale(img: Image.Image, max_width: int) -> Image.Image:
    if img.width <= max_width:
        return img
    h = max(1, int(img.height * max_width / img.width))
    return img.resize((max_width, h), Image.Resampling.BILINEAR)


def is_blank(img: Image.Image | None) -> bool:
    if img is None:
        return True
    return float(np.array(img.convert("L")).std()) < 2.0
