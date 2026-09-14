"""macOS window context and screen text.

Window identity (app, pid, bounds, title) comes from the Quartz window list,
which needs neither Accessibility trust nor a Cocoa run loop; the asyncio
daemon has no run loop, so NSWorkspace's cached frontmost app can go stale
(insights.md §7.11). AX is layered on top when the process is trusted: the
focused window's title, document/URL, and the text walk (planS1 §6.1).

AX attribute names are passed as plain strings. pyobjc does not export every
`kAX…` constant, and importing a missing one disabled all of AX in v1.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

log = logging.getLogger("discrete.capture.ax")

try:
    import Quartz
    from AppKit import NSRunningApplication
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXUIElementCopyAttributeValue,
        AXUIElementCopyParameterizedAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
        AXUIElementSetAttributeValue,
        AXValueGetValue,
        kAXValueCFRangeType,
        kAXValueCGPointType,
        kAXValueCGRectType,
        kAXValueCGSizeType,
    )

    _HAS_QUARTZ = True
except ImportError:  # non-macOS
    _HAS_QUARTZ = False

_AX_OK = 0  # kAXErrorSuccess
_CHROME_ROLES = frozenset({"AXMenuBar", "AXMenu", "AXToolbar", "AXScrollBar", "AXTabGroup", "AXButton"})
_MIN_WINDOW_SIDE = 80


@dataclass
class WindowContext:
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    pid: int = 0
    # Window frame in global display points (top-left origin), or None.
    bounds: tuple[float, float, float, float] | None = None

    @property
    def key(self) -> tuple:
        return (self.pid, self.window_title)


def accessibility_granted() -> bool:
    if not _HAS_QUARTZ:
        return False
    try:
        return bool(AXIsProcessTrusted())
    except Exception:
        return False


def screen_recording_preflight() -> bool:
    """The OS preflight. Known to report False for terminal-launched Python
    even when capture works; `doctor` verifies with a real frame instead."""
    if not _HAS_QUARTZ:
        return False
    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def _front_window() -> dict | None:
    """Frontmost normal-layer window, from the front-to-back window list."""
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
        if int(w.get("kCGWindowLayer", 1)) != 0:
            continue
        if float(w.get("kCGWindowAlpha", 1.0)) <= 0.01:
            continue
        b = w.get("kCGWindowBounds") or {}
        if float(b.get("Width", 0)) < _MIN_WINDOW_SIDE or float(b.get("Height", 0)) < _MIN_WINDOW_SIDE:
            continue
        return w
    return None


def _ax_get(el, attr: str):
    if el is None:
        return None
    try:
        err, val = AXUIElementCopyAttributeValue(el, attr, None)
        return val if err == _AX_OK else None
    except Exception:
        return None


def _ax_url(raw) -> str:
    if raw is None:
        return ""
    if hasattr(raw, "absoluteString"):
        return str(raw.absoluteString() or "")
    return str(raw or "")


def window_context() -> WindowContext:
    if not _HAS_QUARTZ:
        return WindowContext()
    try:
        w = _front_window()
        if w is None:
            return WindowContext()
        pid = int(w.get("kCGWindowOwnerPID", 0))
        b = w.get("kCGWindowBounds") or {}
        bounds = (float(b.get("X", 0)), float(b.get("Y", 0)), float(b.get("Width", 0)), float(b.get("Height", 0)))
        title = str(w.get("kCGWindowName") or "")
        name = str(w.get("kCGWindowOwnerName") or "")
        bundle = ""
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is not None:
            name = str(app.localizedName() or name)
            bundle = str(app.bundleIdentifier() or "")
        url = ""
        if accessibility_granted():
            win = _ax_get(AXUIElementCreateApplication(pid), "AXFocusedWindow")
            title = str(_ax_get(win, "AXTitle") or title)
            for attr in ("AXDocument", "AXURL"):
                url = _ax_url(_ax_get(win, attr))
                if url:
                    break
        return WindowContext(app=name, bundle_id=bundle, window_title=title, url=url, pid=pid, bounds=bounds)
    except Exception as exc:
        log.debug("window_context failed: %s", exc)
        return WindowContext()


_enhanced_pids: set[int] = set()


def _enable_full_tree(app_el, pid: int) -> None:
    """Chromium/Electron expose a full AX tree only when asked (planS1 §6.1)."""
    if pid in _enhanced_pids:
        return
    _enhanced_pids.add(pid)
    for attr in ("AXManualAccessibility", "AXEnhancedUserInterface"):
        try:
            AXUIElementSetAttributeValue(app_el, attr, True)
        except Exception:
            pass


def _walk_text(el, parts: list[str], *, depth: int, max_nodes: int, deadline: float, seen: list[int]) -> None:
    if depth > 40 or seen[0] >= max_nodes or time.monotonic() > deadline:
        return
    seen[0] += 1
    role = str(_ax_get(el, "AXRole") or "")
    if role in _CHROME_ROLES:
        return
    for attr in ("AXValue", "AXTitle", "AXDescription"):
        val = _ax_get(el, attr)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
            break
    for child in _ax_get(el, "AXChildren") or []:
        _walk_text(child, parts, depth=depth + 1, max_nodes=max_nodes, deadline=deadline, seen=seen)


def read_screen_text(pid: int, *, max_nodes: int, timeout_ms: int, max_chars: int) -> str:
    """AX text walk of the focused window of `pid`. Empty when AX is not trusted."""
    if not _HAS_QUARTZ or not pid or not accessibility_granted():
        return ""
    try:
        app_el = AXUIElementCreateApplication(pid)
        _enable_full_tree(app_el, pid)
        win = _ax_get(app_el, "AXFocusedWindow")
        if win is None:
            return ""
        parts: list[str] = []
        deadline = time.monotonic() + timeout_ms / 1000.0
        _walk_text(win, parts, depth=0, max_nodes=max_nodes, deadline=deadline, seen=[0])
        # Adjacent duplicates are common (title + value of one element).
        out: list[str] = []
        for p in parts:
            if not out or out[-1] != p:
                out.append(p)
        return "\n".join(out)[:max_chars]
    except Exception as exc:
        log.debug("read_screen_text failed: %s", exc)
        return ""


# ------------------------------------------------------------ focused field (Bubble)

_VALUE_ATTRS = ("AXValue", "AXSelectedText")
_DESC_ATTRS = ("AXPlaceholderValue", "AXDescription", "AXTitle", "AXHelp", "AXRoleDescription")


@dataclass
class FocusSnapshot:
    """The focused element when the hotkey fired: identity, text, caret position."""

    pid: int = 0
    app: str = ""
    window_title: str = ""
    url: str = ""
    role: str = ""
    value: str = ""
    caret_index: int | None = None
    description: str = ""
    # Caret (or field) rect in global top-left display points, for the overlay.
    caret: tuple[float, float, float, float] | None = None
    window_bounds: tuple[float, float, float, float] | None = None
    element: object = None

    @property
    def has_text(self) -> bool:
        return bool(self.value.strip())

    @property
    def text_to_caret(self) -> str:
        if self.caret_index is not None and 0 <= self.caret_index <= len(self.value):
            return self.value[: self.caret_index]
        return self.value

    def restore(self) -> None:
        """Re-activate the target app and re-focus the field before typing. Chromium
        and Electron apply posted keys only while their window is key."""
        if not _HAS_QUARTZ or not self.pid or frontmost_pid() == self.pid:
            return
        try:
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(self.pid)
            if app is not None:
                app.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
        except Exception:
            pass
        if self.element is not None:
            try:
                AXUIElementSetAttributeValue(self.element, "AXFocused", True)
            except Exception:
                pass


def frontmost_pid() -> int:
    if not _HAS_QUARTZ:
        return 0
    try:
        w = _front_window()
        return int(w.get("kCGWindowOwnerPID", 0)) if w else 0
    except Exception:
        return 0


def _ax_value(raw, kind):
    try:
        ok, val = AXValueGetValue(raw, kind, None)
        return val if ok else None
    except Exception:
        return None


def _caret_rect(el) -> tuple[tuple[float, float, float, float] | None, int | None]:
    rng = _ax_get(el, "AXSelectedTextRange")
    index = None
    if rng is not None:
        r = _ax_value(rng, kAXValueCFRangeType)
        if r is not None:
            index = int(r.location)
        try:
            err, b = AXUIElementCopyParameterizedAttributeValue(el, "AXBoundsForRange", rng, None)
            rect = _ax_value(b, kAXValueCGRectType) if err == _AX_OK else None
            if rect is not None and (rect.size.height > 0 or rect.origin.x > 0):
                return (rect.origin.x, rect.origin.y, rect.size.width, max(rect.size.height, 14.0)), index
        except Exception:
            pass
    pos = _ax_get(el, "AXPosition")
    size = _ax_get(el, "AXSize")
    p = _ax_value(pos, kAXValueCGPointType) if pos is not None else None
    s = _ax_value(size, kAXValueCGSizeType) if size is not None else None
    if p is not None:
        h = s.height if s is not None else 20.0
        # Field rect; the pill goes under the field's first line.
        return (p.x, p.y, s.width if s is not None else 0.0, min(h, 28.0)), index
    return None, index


def capture_focus() -> FocusSnapshot:
    """Everything Bubble needs about the focused field, in one pass (~2-5 ms)."""
    ctx = window_context()
    snap = FocusSnapshot(pid=ctx.pid, app=ctx.app, window_title=ctx.window_title, url=ctx.url,
                         window_bounds=ctx.bounds)
    if not _HAS_QUARTZ or not accessibility_granted():
        return snap
    el = _ax_get(AXUIElementCreateApplication(ctx.pid), "AXFocusedUIElement") if ctx.pid else None
    if el is None:
        el = _ax_get(AXUIElementCreateSystemWide(), "AXFocusedUIElement")
    if el is None:
        return snap
    snap.element = el
    snap.role = str(_ax_get(el, "AXRole") or "")
    for attr in _VALUE_ATTRS:
        val = _ax_get(el, attr)
        if isinstance(val, str) and val.strip():
            snap.value = val
            break
    for attr in _DESC_ATTRS:
        val = _ax_get(el, attr)
        if isinstance(val, str) and val.strip():
            snap.description = val.strip()[:120]
            break
    snap.caret, snap.caret_index = _caret_rect(el)
    return snap


def check_permissions() -> list[str]:
    issues: list[str] = []
    if not accessibility_granted():
        issues.append(
            "Accessibility not granted to this terminal/app: screen text falls back to OCR and "
            "window titles come from the window list (System Settings → Privacy → Accessibility)"
        )
    return issues


def warn_permissions() -> None:
    for msg in check_permissions():
        print(f"[capture] WARNING: {msg}", file=sys.stderr, flush=True)
