"""Daemon side of the status pill: spawn the overlay process once, send it JSON lines.

Every call is fire-and-forget and never raises: a missing or crashed overlay
costs the user the animation, never the answer.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

from ..config import data_dir

log = logging.getLogger("discrete.bubble.overlay")


class Overlay:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.platform == "darwin"
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not self.enabled or (self._proc and self._proc.poll() is None):
            return
        try:
            # The child inherits its own copy of the log descriptor; ours closes here.
            with open(data_dir() / "overlay.log", "a") as err:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "discrete.bubble.overlay"],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, text=True, bufsize=1,
                )
        except OSError as exc:
            log.warning("overlay unavailable: %s", exc)
            self.enabled = False

    def _send(self, **msg) -> None:
        if not self.enabled:
            return
        if self._proc is None or self._proc.poll() is not None:
            self.start()
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError, AttributeError) as exc:
            log.debug("overlay send failed: %s", exc)
            self._proc = None

    def show(self, anchor=None) -> None:
        self._send(op="show", anchor=list(anchor) if anchor else None)

    def status(self, text: str) -> None:
        self._send(op="status", text=text)

    def done(self, text: str) -> None:
        self._send(op="done", text=text)

    def message(self, text: str, tone: str = "info") -> None:
        self._send(op="message", text=text, tone=tone)

    def hide(self) -> None:
        self._send(op="hide")

    def close(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            self._send(op="quit")
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        self._proc = None
