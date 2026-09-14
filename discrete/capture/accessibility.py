"""macOS Accessibility reads — metadata and screen text."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

log = logging.getLogger("discrete.capture.ax")

try:
    import Quartz
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        kAXErrorSuccess,
        kAXFocusedUIElementAttribute,
        kAXTitleAttribute,
        kAXValueAttribute,
        kAXDescriptionAttribute,
        kAXRoleAttribute,
        kAXChildrenAttribute,
        kAXDocumentAttribute,
        kAXURLAttribute,
        kAXPositionAttribute,
        kAXSizeAttribute,
        kAXManualAccessibilityAttribute,
        kAXEnhancedUserInterfaceAttribute,
    )
    from AppKit import NSWorkspace
    from Foundation import NSURL

    _HAS_AX = True
except ImportError:
    _HAS_AX = False


@dataclass
class WindowContext:
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    pid: int = 0


def screen_recording_granted() -> bool:
    if not _HAS_AX:
        return False
    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def accessibility_granted() -> bool:
    if not _HAS_AX:
        return False
    try:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())
    except Exception:
        return False


def _ax_get(el, attr: str):
    if not _HAS_AX or el is None:
        return None
    try:
        err, val = AXUIElementCopyAttributeValue(el, attr, None)
        if err != kAXErrorSuccess or val is None:
            return None
        return val
    except Exception:
        return None


def window_context() -> WindowContext:
    if not _HAS_AX:
        return WindowContext()
    try:
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return WindowContext()
        pid = int(app.processIdentifier())
        name = str(app.localizedName() or "")
        bundle = str(app.bundleIdentifier() or "")
        app_el = AXUIElementCreateApplication(pid)
        win = _ax_get(app_el, kAXFocusedUIElementAttribute)
        title = str(_ax_get(win, kAXTitleAttribute) or "") if win else ""
        url = ""
        for attr in (kAXDocumentAttribute, kAXURLAttribute):
            raw = _ax_get(win, attr)
            if raw is None:
                continue
            if hasattr(raw, "absoluteString"):
                url = str(raw.absoluteString() or "")
            else:
                url = str(raw or "")
            if url:
                break
        return WindowContext(app=name, bundle_id=bundle, window_title=title, url=url, pid=pid)
    except Exception as exc:
        log.debug("window_context failed: %s", exc)
        return WindowContext()


def _walk_text(el, parts: list[str], *, depth: int, max_nodes: int, deadline: float) -> None:
    if depth > 40 or len(parts) >= max_nodes or time.monotonic() > deadline:
        return
    role = str(_ax_get(el, kAXRoleAttribute) or "")
    skip = {"AXMenuBar", "AXToolbar", "AXScrollBar", "AXTabGroup", "AXButton"}
    if role in skip:
        return
    for attr in (kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute):
        val = _ax_get(el, attr)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
            break
    children = _ax_get(el, kAXChildrenAttribute) or []
    for child in children:
        _walk_text(child, parts, depth=depth + 1, max_nodes=max_nodes, deadline=deadline)


def read_screen_text(*, max_nodes: int, timeout_ms: int) -> str:
    if not _HAS_AX:
        return ""
    from ..config import T

    ctx = window_context()
    if not ctx.pid:
        return ""
    try:
        app_el = AXUIElementCreateApplication(ctx.pid)
        _ax_get(app_el, kAXManualAccessibilityAttribute)
        _ax_get(app_el, kAXEnhancedUserInterfaceAttribute)
        win = _ax_get(app_el, kAXFocusedUIElementAttribute)
        if win is None:
            return ""
        parts: list[str] = []
        deadline = time.monotonic() + timeout_ms / 1000.0
        _walk_text(win, parts, depth=0, max_nodes=max_nodes, deadline=deadline)
        text = "\n".join(parts)
        return text[: T.SCREEN_TEXT_MAX_CHARS]
    except Exception as exc:
        log.debug("read_screen_text failed: %s", exc)
        return ""


def check_permissions() -> list[str]:
    issues: list[str] = []
    if not screen_recording_granted():
        issues.append("Screen Recording permission missing")
    if not accessibility_granted():
        issues.append("Accessibility permission missing")
    return issues


def warn_permissions() -> None:
    for msg in check_permissions():
        print(f"[capture] WARNING: {msg}", file=sys.stderr, flush=True)
