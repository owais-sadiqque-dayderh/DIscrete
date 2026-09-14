"""Global hotkey (CGEventTap) and atomic text insertion (CGEvent).

The tap *consumes* the chord, so the host app never sees Option+Tab. While a
run is in flight it also watches the keyboard: Escape cancels (and is
swallowed), and any other key the user types cancels too (and passes through).
Our own synthetic keystrokes carry EVENT_TAG, so they never cancel the run.

Text goes in as one `CGEventKeyboardSetUnicodeString` event per chunk, posted
to the target pid. There is no spinner typed into the field any more, so
there are no cleanup backspaces to race with (WorkA/WorkAA insights).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger("discrete.bubble.keys")

try:
    import Quartz

    _OK = True
except ImportError:  # non-macOS
    _OK = False

EVENT_TAG = 0xD15C4E7E
ESCAPE = 53

_MODS = {"cmd": 1 << 20, "alt": 1 << 19, "option": 1 << 19, "ctrl": 1 << 18, "shift": 1 << 17}
_RELEVANT = _MODS["cmd"] | _MODS["alt"] | _MODS["ctrl"] | _MODS["shift"]
_KEYCODES = {
    "tab": 48, "space": 49, "return": 36, "escape": 53, "backslash": 42, "grave": 50, "period": 47,
    "comma": 43, "slash": 44, "semicolon": 41, "quote": 39, "j": 38, "k": 40, "l": 37, "b": 11, "e": 14,
}


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'<alt>+<tab>' -> (modifier mask, keycode); keycode -1 if unparseable."""
    mask, code = 0, -1
    for part in spec.replace(" ", "").split("+"):
        name = part.strip("<>").lower()
        if name in _MODS:
            mask |= _MODS[name]
        elif name in _KEYCODES:
            code = _KEYCODES[name]
    return mask, code


class HotkeyTap:
    def __init__(self, spec: str, *, on_fire: Callable[[], None], on_escape: Callable[[], None],
                 on_user_key: Callable[[], None], busy: Callable[[], bool]):
        self.mask, self.keycode = parse_hotkey(spec)
        self.spec = spec
        self.on_fire, self.on_escape, self.on_user_key, self.busy = on_fire, on_escape, on_user_key, busy
        self.available = _OK and self.keycode >= 0
        self._tap = None
        self._runloop = None
        self._ready = threading.Event()
        self.armed = False

    def start(self) -> bool:
        if not self.available:
            log.warning("hotkey %r unavailable (no Quartz or unparseable)", self.spec)
            return False
        threading.Thread(target=self._run, name="discrete-hotkey", daemon=True).start()
        self._ready.wait(2.0)
        return self.armed

    def _callback(self, proxy, etype, event, refcon):  # pragma: no cover - needs a live session
        try:
            if etype == Quartz.kCGEventKeyDown:
                if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == EVENT_TAG:
                    return event
                code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                flags = Quartz.CGEventGetFlags(event) & _RELEVANT
                if code == self.keycode and flags == self.mask:
                    self.on_fire()
                    return None
                if self.busy():
                    if code == ESCAPE:
                        self.on_escape()
                        return None
                    self.on_user_key()
            elif etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(self._tap, True)
                log.warning("event tap was disabled by the system; re-enabled")
        except Exception:
            log.exception("event tap callback failed")
        return event

    def _run(self) -> None:  # pragma: no cover
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap, Quartz.kCGEventTapOptionDefault,
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown), self._callback, None,
        )
        if not self._tap:
            log.error("could not create the event tap: grant Accessibility (and Input Monitoring) "
                      "to the terminal/app running discrete, then restart")
            self._ready.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._runloop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._runloop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        self.armed = True
        self._ready.set()
        Quartz.CFRunLoopRun()

    def stop(self) -> None:
        if self._runloop is not None:  # pragma: no cover
            try:
                Quartz.CFRunLoopStop(self._runloop)
            except Exception:
                pass


_lock = threading.Lock()


def type_text(text: str, *, pid: int | None = None) -> bool:
    """Insert `text` at the cursor as one atomic unicode event."""
    if not text or not _OK:
        return False
    with _lock:
        try:
            src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
            for down in (True, False):
                ev = Quartz.CGEventCreateKeyboardEvent(src, 0, down)
                Quartz.CGEventKeyboardSetUnicodeString(ev, len(text), text)
                Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, EVENT_TAG)
                if pid:
                    Quartz.CGEventPostToPid(pid, ev)
                else:
                    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)
            return True
        except Exception as exc:
            log.warning("type_text failed: %s", exc)
            return False
