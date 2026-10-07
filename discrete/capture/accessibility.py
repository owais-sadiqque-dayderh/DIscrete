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
import re
import sys
import time
from dataclasses import dataclass

from ..config import T
from ..perf import PERF

log = logging.getLogger("discrete.capture.ax")

try:
    import Quartz
    from AppKit import NSMakeRect, NSRunningApplication, NSValue
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXTextMarkerRangeCreate,
        AXUIElementCopyAttributeValue,
        AXUIElementCopyParameterizedAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
        AXUIElementSetAttributeValue,
        AXUIElementSetMessagingTimeout,
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
_CHROME_ROLES = frozenset({"AXMenuBar", "AXMenu", "AXToolbar", "AXScrollBar", "AXTabGroup"})
_NAV_ROLES = frozenset({"AXLandmarkNavigation"})
_TEXT_ROLES = frozenset({"AXStaticText", "AXTextArea", "AXTextView", "AXTextField"})
_MIN_WINDOW_SIDE = 80
# Unlocated/offscreen text is a small fallback, never an unbounded second walk.
_MIN_VISIBLE_CHARS = 80
_MAX_DEFERRED_CHARS = 256
_MAX_DEFERRED_LINES = 8
_TABLE_ROLES = frozenset({"AXTable", "AXGrid"})
_CELL_ROLES = frozenset({"AXCell", "AXGridCell", "AXColumnHeader", "AXRowHeader"})
_CONTROL_LABEL_ROLES = frozenset({"AXRadioButton", "AXCheckBox"})


@dataclass
class WindowContext:
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    pid: int = 0
    # Window frame in global display points (top-left origin), or None.
    bounds: tuple[float, float, float, float] | None = None
    window_id: int = 0

    @property
    def key(self) -> tuple:
        # The URL is part of identity: in-tab navigation often keeps the window title.
        return (self.pid, self.window_title, self.url.split("#")[0])


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


# Per-call ceiling for one AX attribute read. The system default (~6 s) lets one hung
# app stall a capture or Bubble; Chromium can need ~100 ms while it builds its tree.
_AX_TIMEOUT_S = 0.25


def _ax_get(el, attr: str):
    if el is None:
        return None
    try:
        AXUIElementSetMessagingTimeout(el, _AX_TIMEOUT_S)
        err, val = AXUIElementCopyAttributeValue(el, attr, None)
        return val if err == _AX_OK else None
    except Exception:
        return None


def _ax_get_param(el, attr: str, parameter):
    try:
        AXUIElementSetMessagingTimeout(el, _AX_TIMEOUT_S)
        err, val = AXUIElementCopyParameterizedAttributeValue(el, attr, parameter, None)
        return val if err == _AX_OK else None
    except Exception:
        return None


def _has_text_child(children, deadline: float) -> bool:
    """Check aria containers for text descendants, including clipped ones."""
    stack = list(children)
    checked = 0
    while stack and checked < 25 and time.monotonic() < deadline:
        child = stack.pop()
        checked += 1
        if _ax_get(child, "AXRole") in _TEXT_ROLES:
            return True
        if any(isinstance(_ax_get(child, attr), str) for attr in ("AXValue", "AXTitle")):
            return True
        stack.extend(_ax_get(child, "AXChildren") or [])
    return bool(stack)  # uncertainty is safer than exposing an aria summary


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
            if win is not None and bundle.startswith(_BROWSER_BUNDLES):
                page_url, page_title = _web_page(pid, win, title)
                if page_url:
                    # The tab, not the window: Arc titles windows by Space ("Space 1").
                    url, title = page_url, page_title or title
        return WindowContext(app=name, bundle_id=bundle, window_title=title, url=url, pid=pid, bounds=bounds,
                             window_id=int(w.get("kCGWindowNumber", 0)))
    except Exception as exc:
        log.debug("window_context failed: %s", exc)
        return WindowContext()


_BROWSER_BUNDLES = (
    "company.thebrowser.", "com.google.Chrome", "com.apple.Safari", "com.microsoft.edgemac", "com.brave.Browser",
    "com.vivaldi.Vivaldi", "com.operasoftware.Opera", "org.chromium.Chromium",
)
_WEB_REFIND_S = 2.0
# pid -> (window title at lookup, AXWebArea or None, monotonic time of lookup)
_web_cache: dict[int, tuple[str, object, float]] = {}


def _find_role(el, role: str, *, max_nodes: int = 1500, budget_s: float = 0.08,
               seen_out: list[int] | None = None):
    deadline = time.monotonic() + budget_s
    stack, seen = [el], 0
    while stack and seen < max_nodes and time.monotonic() < deadline:
        cur = stack.pop()
        seen += 1
        if _ax_get(cur, "AXRole") == role:
            if seen_out is not None:
                seen_out[0] = seen
            return cur
        stack.extend(reversed(_ax_get(cur, "AXChildren") or []))
    if seen_out is not None:
        seen_out[0] = seen
    return None


def _web_page(pid: int, win, win_title: str) -> tuple[str, str]:
    """(url, page title) of the visible tab, from its AXWebArea (~0 ms when cached,
    ~30 ms to find). AppleScript is 0.25-4 s per call, too slow to poll (insights.md §7.33).
    The element is re-found when the window title changes and every `_WEB_REFIND_S`,
    because a background tab's web area stays valid and would report the old page."""
    now_m = time.monotonic()
    ent = _web_cache.get(pid)
    wa = None
    if ent and ent[0] == win_title and now_m - ent[2] < _WEB_REFIND_S:
        wa = ent[1]
        if wa is None:
            return "", ""
    if wa is None:
        _enable_full_tree(AXUIElementCreateApplication(pid), pid)
        wa = _find_role(win, "AXWebArea")
        _web_cache[pid] = (win_title, wa, now_m)
        if wa is None:
            return "", ""
    url = _ax_url(_ax_get(wa, "AXURL"))
    if not url.startswith("http"):
        return "", ""
    return url, str(_ax_get(wa, "AXTitle") or "")


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


def _node_frame(el):
    """Chromium supplies position/size; some native adapters supply only frame."""
    p = _point_xy(_ax_value(_ax_get(el, "AXPosition"), kAXValueCGPointType))
    s = _size_wh(_ax_value(_ax_get(el, "AXSize"), kAXValueCGSizeType))
    if p is None or s is None:
        frame = _rect_xywh(_ax_value(_ax_get(el, "AXFrame"), kAXValueCGRectType))
        if frame is not None:
            p, s = frame[:2], frame[2:]
    return p, s


def _intersects(point, size, viewport) -> bool:
    x, y, w, h = viewport
    # Chromium clamps clipped text to a one-pixel strip at the viewport edge.
    return (size[0] > 1 and size[1] > 1 and point[0] + size[0] > x
            and point[0] < x + w and point[1] + size[1] > y and point[1] < y + h)


def _deferred_full(values: list[str]) -> bool:
    return (sum(map(len, values)) >= _MAX_DEFERRED_CHARS
            or sum(bool(v) for v in values) >= _MAX_DEFERRED_LINES)


def _walk_text(el, parts: list[str], *, depth: int, max_nodes: int, deadline: float, seen: list[int],
               viewport: tuple[float, float, float, float] | None = None,
               deferred: list[str] | None = None, in_banner: bool = False,
               clipped_parent: bool = False, parent=None, grandparent=None,
               meta: list[tuple] | None = None, deferred_meta: list[tuple] | None = None,
               in_table: bool = False, row=None, cell=None, row_table: bool = False) -> None:
    if depth > 40:
        PERF.count("health.capture_ax_depth_limit")
        return
    if seen[0] >= max_nodes or time.monotonic() >= deadline:
        return
    seen[0] += 1
    role = str(_ax_get(el, "AXRole") or "")
    if role in _CHROME_ROLES or role in _NAV_ROLES or (in_banner and role == "AXList"):
        return
    # AXColumn is another projection of the same cells, not additional content.
    # Outside a semantic table it may be the only available content, so retain it.
    if row_table and role == "AXColumn":
        return
    p, s = _node_frame(el) if viewport else (None, None)
    located = p is not None and s is not None
    visible = _intersects(p, s, viewport) if located else not clipped_parent
    if not visible and deferred is not None and (
        sum(map(len, parts)) >= _MIN_VISIBLE_CHARS
        or _deferred_full(deferred)
    ):
        return
    if role in {"AXWebArea", "AXScrollArea"} and visible and located:
        x, y, w, h = viewport
        left, top = max(x, p[0]), max(y, p[1])
        viewport = (left, top, min(x + w, p[0] + s[0]) - left,
                    min(y + h, p[1] + s[1]) - top)
    if role in _TABLE_ROLES:
        in_table, row, cell = True, None, None
    elif in_table and role == "AXRow":
        row, cell = el, None
    elif row is not None and role in _CELL_ROLES:
        cell = el
    target = parts if visible and (located or viewport is None) else deferred
    target_meta = meta if target is parts else deferred_meta

    def emit(value):
        if target is None:
            return
        if target is deferred and _deferred_full(target):
            return
        value = value.strip()
        if target is deferred:
            value = value[:_MAX_DEFERRED_CHARS - sum(map(len, target))]
        target.append(value)
        if target_meta is not None:
            target_meta.append((role, parent, p, s, el, row, cell, grandparent))

    range_used = False
    value = None
    if role in {"AXTextArea", "AXTextView"} and visible:
        char_range = _ax_get(el, "AXVisibleCharacterRange")
        if char_range is not None:
            value = _ax_get_param(el, "AXStringForRange", char_range)
            range_used = isinstance(value, str)
    children = [] if range_used else (_ax_get(el, "AXChildren") or [])
    if role in _TABLE_ROLES:
        row_table = bool(_ax_get(el, "AXRows")) or any(
            time.monotonic() < deadline and _ax_get(c, "AXRole") == "AXRow" for c in children)
    # Chromium exposes some row headers (Wikipedia's rank) as a plain group.
    # A wrapper containing semantic cells is not itself a cell.
    if row is not None and parent == row and role == "AXGroup" and not any(
        time.monotonic() < deadline and _ax_get(c, "AXRole") in _CELL_ROLES for c in children
    ):
        cell = el
    # Preserve empty cells too: "A |  | C" must not shift C into column two.
    if cell is el and target is not None:
        emit("")
    before = (len(parts), len(deferred) if deferred is not None else 0)
    if role in _TEXT_ROLES:
        if not range_used:
            value = _ax_get(el, "AXValue")
            if not isinstance(value, str) or not value.strip():
                value = _ax_get(el, "AXTitle")
        if isinstance(value, str) and value.strip():
            emit(value)
    for child in children:
        if not visible and deferred is not None and _deferred_full(deferred):
            break
        _walk_text(child, parts, depth=depth + 1, max_nodes=max_nodes, deadline=deadline, seen=seen,
                   viewport=viewport, deferred=deferred, in_banner=in_banner or role == "AXLandmarkBanner",
                   clipped_parent=not visible, parent=el, grandparent=parent, meta=meta, deferred_meta=deferred_meta,
                   in_table=in_table, row=row, cell=cell, row_table=row_table)
    if time.monotonic() >= deadline:
        return
    # Read container labels only if descendants cannot supply painted text.
    if visible and role not in _TEXT_ROLES and not range_used:
        has_child_text = before != (len(parts), len(deferred) if deferred is not None else 0)
        if not has_child_text and not _has_text_child(children, deadline):
            selected = _ax_get(el, "AXValue")
            if not isinstance(selected, str) or not selected.strip():
                selected = _ax_get(el, "AXTitle")
            from_description = not isinstance(selected, str) or not selected.strip()
            if from_description:
                selected = _ax_get(el, "AXDescription")
            icon = ((from_description and role in {"AXButton", "AXLink"} and (s is None or s[0] < 48))
                    or (role == "AXImage" and _ax_get(parent, "AXRole") in {"AXButton", "AXLink"}))
            if not icon and isinstance(selected, str) and selected.strip():
                emit(selected)


def _local_unique(parts: list[str], meta: list[tuple]) -> tuple[list[str], list[tuple]]:
    """Remove duplicate AX projections in eight neighbouring emissions.

    Equal words at different positions (or in distinct comments/cells) survive.
    Missing geometry alone is not evidence that repeated text is redundant.
    """
    out, locations = [], []
    recent: list[tuple[str, tuple]] = []
    for value, current in zip(parts, meta, strict=True):
        key = " ".join(value.split())
        point, size = current[2:4]
        duplicate = False
        for old_key, old in recent[-8:]:
            if not key or key != old_key or current[6] != old[6]:
                continue
            same_element = current[4] == old[4]
            same_frame = (point is not None and size is not None and old[2] is not None and old[3] is not None
                          and all(abs(a - b) <= 2 for a, b in zip((*point, *size), (*old[2], *old[3]), strict=True)))
            # A radio/checkbox's aria title can name text in an adjacent
            # label sibling. Require both a shared local group and geometry.
            control, label = (current, old) if current[0] in _CONTROL_LABEL_ROLES else (old, current)
            sibling_label = (
                control[0] in _CONTROL_LABEL_ROLES and label[0] == "AXStaticText"
                and control[2] is not None and control[3] is not None and label[2] is not None
                and (control[1] == label[1] or control[1] == label[7])
                and abs(control[2][1] - label[2][1]) <= 4
                and 0 <= label[2][0] - (control[2][0] + control[3][0]) <= 32
            )
            if same_element or same_frame or sibling_label:
                duplicate = True
                break
        recent.append((key, current))
        if len(recent) > 8:
            del recent[0]
        if not duplicate:
            out.append(value)
            locations.append(current)
    return out, locations


def _merge_static_runs(parts: list[str], meta: list[tuple]) -> list[str]:
    """Join visual text runs, then cells within each semantic row."""
    parts, meta = _local_unique(parts, meta)
    out: list[str] = []
    prior = None
    active_row = active_cell = None
    cells: list[str] = []

    def flush_row():
        if cells and any(cells):
            out.append(" | ".join(cells))
        cells.clear()

    for value, current in zip(parts, meta, strict=True):
        role, parent, point, size, _, row, cell, _ = current
        if row != active_row:
            flush_row()
            active_row, active_cell, prior = row, None, None
        if row is not None and cell is not None:
            if cell != active_cell:
                cells.append("")
                active_cell = cell
            if value:
                cells[-1] = " ".join(filter(None, (cells[-1], " ".join(value.split()))))
            prior = None
            continue
        if not value:
            continue
        if prior is not None and role == prior[0] == "AXStaticText" and "\n" not in value and "\n" not in out[-1]:
            old_point, old_size = prior[2], prior[3]
            same_line = point is not None and old_point is not None and abs(point[1] - old_point[1]) <= 4
            if same_line:
                adjacent = point[0] >= old_point[0] and (old_size is None or point[0] <= old_point[0] + old_size[0] + 30)
            else:
                adjacent = point is None or old_point is None
            if (same_line and adjacent) or (parent is prior[1] and adjacent):
                out[-1] = f"{out[-1]} {value}"
                prior = current
                continue
        out.append(value)
        prior = current
    flush_row()
    return out


def _web_visible_text(web_area, viewport: tuple[float, float, float, float], deadline: float) -> str:
    """Read Chromium's line-structured text between markers inside the window."""
    if time.monotonic() >= deadline:
        return ""
    try:
        bounds = NSValue.valueWithRect_(NSMakeRect(*viewport))
        start = _ax_get_param(web_area, "AXStartTextMarkerForBounds", bounds)
        end = _ax_get_param(web_area, "AXEndTextMarkerForBounds", bounds)
        if start is None or end is None or time.monotonic() >= deadline:
            return ""
        marker_range = AXTextMarkerRangeCreate(None, start, end)
    except Exception:
        return ""
    if time.monotonic() >= deadline:
        return ""
    raw_frame = _ax_get_param(web_area, "AXBoundsForTextMarkerRange", marker_range)
    frame = _rect_xywh(_ax_value(raw_frame, kAXValueCGRectType))
    if frame is None:
        return ""
    x, y, w, h = frame
    vx, vy, vw, vh = viewport
    if w <= 0 or h <= 0 or x + w <= vx or x >= vx + vw or y < vy - 8 or y + h > vy + vh + 8:
        return ""
    if time.monotonic() >= deadline:
        return ""
    value = _ax_get_param(web_area, "AXStringForTextMarkerRange", marker_range)
    if not isinstance(value, str):
        return ""
    # Some sites flatten an entire article into one line; the node walk retains
    # visual line boundaries there. Empty text is also a fallback signal.
    text = value.strip()
    return text if text.count("\n") >= 2 or len(text) < 200 else ""


def read_screen_text(pid: int, *, max_nodes: int, timeout_ms: int, max_chars: int,
                     viewport: tuple[float, float, float, float] | None = None) -> str:
    """AX text walk of the focused window of `pid`. Empty when AX is not trusted."""
    if not _HAS_QUARTZ:
        PERF.count("health.capture_ax_empty.unavailable")
        return ""
    if not pid:
        PERF.count("health.capture_ax_empty.no_pid")
        return ""
    if not accessibility_granted():
        PERF.count("health.capture_ax_empty.untrusted")
        return ""
    try:
        app_el = AXUIElementCreateApplication(pid)
        _enable_full_tree(app_el, pid)
        win = _ax_get(app_el, "AXFocusedWindow")
        if win is None:
            PERF.count("health.capture_ax_empty.no_window")
            return ""
        if viewport:
            p, s = _node_frame(win)
            if p is not None and s is not None and any(abs(a - b) > 2 for a, b in zip((*p, *s), viewport, strict=True)):
                PERF.count("health.capture_ax_empty.viewport_mismatch")
                return ""  # another window of the same app gained focus; trust captured pixels
        if viewport is None:
            p, s = _node_frame(win)
            if p is not None and s is not None:
                viewport = (*p, *s)
        deadline = time.monotonic() + timeout_ms / 1000.0
        searched = [0]
        if T.AX_TEXT_MARKERS and viewport and max_nodes >= 4:
            # Experimental (DISCRETE_AX_TEXT_MARKERS): not yet verified live on enough pages.
            # Reuse the remaining node budget if marker reads fail.
            web_area = _find_role(win, "AXWebArea", max_nodes=max_nodes // 2,
                                  budget_s=min(0.05, max(0.0, deadline - time.monotonic())), seen_out=searched)
            if web_area is not None:
                visible_text = _web_visible_text(web_area, viewport, deadline)
                if visible_text:
                    if len(visible_text) >= max_chars:
                        PERF.count("health.capture_ax_text_truncated")
                    return visible_text[:max_chars]
        parts: list[str] = []
        deferred: list[str] = []
        meta: list[tuple] = []
        deferred_meta: list[tuple] = []
        seen = [0]
        remaining_nodes = max(0, max_nodes - searched[0])
        _walk_text(win, parts, depth=0, max_nodes=remaining_nodes,
                   deadline=deadline, seen=seen,
                   viewport=viewport, deferred=deferred, meta=meta, deferred_meta=deferred_meta)
        out = _merge_static_runs(parts, meta)
        text = "\n".join(out)
        if viewport and len(text) < _MIN_VISIBLE_CHARS:
            tail = "\n".join(_merge_static_runs(deferred, deferred_meta))[:_MAX_DEFERRED_CHARS]
            text = "\n".join(filter(None, (text, tail)))
        expired = time.monotonic() > deadline
        exhausted = seen[0] >= remaining_nodes
        if expired:
            PERF.count("health.capture_ax_deadline")
        if exhausted:
            PERF.count("health.capture_ax_node_limit")
        if not text:
            reason = "deadline" if expired else "node_limit" if exhausted else "no_text"
            PERF.count(f"health.capture_ax_empty.{reason}")
        if len(text) >= max_chars:
            PERF.count("health.capture_ax_text_truncated")
        return text[:max_chars]
    except Exception as exc:
        log.debug("read_screen_text failed: %s", exc)
        PERF.count("health.capture_ax_empty.error")
        return ""


# ------------------------------------------------------------ focused field (Bubble)

_VALUE_ATTRS = ("AXValue", "AXSelectedText")
_INVISIBLE = re.compile(r"[​-‏ - ⁠-⁤﻿￼]")
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
    window_id: int = 0
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


# pyobjc may bridge AXValue structs either as objects (NSPoint/NSSize/NSRect/CFRange,
# with .x/.width/.location attributes) or as plain (nested) tuples, depending on the
# installed pyobjc version. These normalizers accept both so a tuple never raises
# AttributeError (insights.md §caret) and vice versa.
def _range_location(r) -> int | None:
    if r is None:
        return None
    try:
        return int(r.location)
    except AttributeError:
        try:
            return int(r[0])
        except (TypeError, IndexError, ValueError):
            return None


def _point_xy(p) -> tuple[float, float] | None:
    if p is None:
        return None
    try:
        return float(p.x), float(p.y)
    except AttributeError:
        try:
            return float(p[0]), float(p[1])
        except (TypeError, IndexError, ValueError):
            return None


def _size_wh(s) -> tuple[float, float] | None:
    if s is None:
        return None
    try:
        return float(s.width), float(s.height)
    except AttributeError:
        try:
            return float(s[0]), float(s[1])
        except (TypeError, IndexError, ValueError):
            return None


def _rect_xywh(r) -> tuple[float, float, float, float] | None:
    if r is None:
        return None
    try:
        return float(r.origin.x), float(r.origin.y), float(r.size.width), float(r.size.height)
    except AttributeError:
        try:
            (ox, oy), (w, h) = r
            return float(ox), float(oy), float(w), float(h)
        except (TypeError, ValueError):
            return None


def _caret_rect(el) -> tuple[tuple[float, float, float, float] | None, int | None]:
    rng = _ax_get(el, "AXSelectedTextRange")
    index = None
    if rng is not None:
        index = _range_location(_ax_value(rng, kAXValueCFRangeType))
        try:
            err, b = AXUIElementCopyParameterizedAttributeValue(el, "AXBoundsForRange", rng, None)
            rect = _rect_xywh(_ax_value(b, kAXValueCGRectType)) if err == _AX_OK else None
            if rect is not None and (rect[3] > 0 or rect[0] > 0):
                x, y, w, h = rect
                return (x, y, w, max(h, 14.0)), index
        except Exception:
            pass
    p = _point_xy(_ax_value(_ax_get(el, "AXPosition"), kAXValueCGPointType))
    s = _size_wh(_ax_value(_ax_get(el, "AXSize"), kAXValueCGSizeType))
    if p is not None:
        h = s[1] if s is not None else 20.0
        # Field rect; the pill goes under the field's first line.
        return (p[0], p[1], s[0] if s is not None else 0.0, min(h, 28.0)), index
    return None, index


def capture_focus() -> FocusSnapshot:
    """Everything Bubble needs about the focused field, in one pass (~2-5 ms)."""
    ctx = window_context()
    snap = FocusSnapshot(pid=ctx.pid, app=ctx.app, window_title=ctx.window_title, url=ctx.url,
                         window_bounds=ctx.bounds, window_id=ctx.window_id)
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
        # Canvas editors (Google Docs) report a zero-width space as the field value;
        # str.strip() keeps it, so it looked like text (insights.md §7.35).
        val = _INVISIBLE.sub("", val) if isinstance(val, str) else ""
        if val.strip():
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
