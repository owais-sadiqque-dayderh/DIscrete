"""Focused tests for visible AX extraction and native-pixel OCR handoff."""

import asyncio
import io

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import ocr
from discrete.capture.frames import _storage_frame
from discrete.capture.screen import ScreenObserver


def _read(monkeypatch, nodes, *, max_chars=2000):
    monkeypatch.setattr(ax, "_HAS_QUARTZ", True)
    monkeypatch.setattr(ax, "accessibility_granted", lambda: True)
    monkeypatch.setattr(ax, "AXUIElementCreateApplication", lambda pid: "app", raising=False)
    monkeypatch.setattr(ax, "_enable_full_tree", lambda app, pid: None)
    monkeypatch.setattr(ax, "_ax_get", lambda el, attr: nodes.get(el, {}).get(attr))
    monkeypatch.setattr(ax, "_ax_value", lambda value, kind: value)
    return ax.read_screen_text(1, max_nodes=100, timeout_ms=1000, max_chars=max_chars,
                               viewport=(0, 0, 200, 200))


def _root(*children):
    return {"app": {"AXFocusedWindow": "root"},
            "root": {"AXRole": "AXWindow", "AXPosition": (0, 0), "AXSize": (200, 200),
                     "AXChildren": list(children)}}


def test_static_runs_merge_on_one_line_but_not_across_lines(monkeypatch):
    nodes = _root("parent")
    nodes["parent"] = {"AXRole": "AXGroup", "AXChildren": ["a", "b", "c"]}
    for name, word, y in (("a", "So", 20), ("b", "imagine", 21), ("c", "Next", 60)):
        nodes[name] = {"AXRole": "AXStaticText", "AXValue": word,
                       "AXPosition": (10, y), "AXSize": (70, 15)}
    assert _read(monkeypatch, nodes) == "So imagine\nNext"


def test_description_does_not_repeat_text_children(monkeypatch):
    nodes = _root("article", "hidden_article", "leaf")
    nodes["article"] = {"AXRole": "AXGroup", "AXDescription": "Repeated aria label",
                        "AXChildren": ["text"]}
    nodes["text"] = {"AXRole": "AXStaticText", "AXValue": "Visible article"}
    nodes["hidden_article"] = {"AXRole": "AXGroup", "AXDescription": "Hidden aria summary",
                               "AXChildren": ["hidden_text"]}
    nodes["hidden_text"] = {"AXRole": "AXStaticText", "AXValue": "Offscreen text",
                            "AXPosition": (240, 10), "AXSize": (100, 20)}
    nodes["leaf"] = {"AXRole": "AXImage", "AXDescription": "Chart"}
    assert _read(monkeypatch, nodes) == "Visible article\nChart"


def test_text_area_uses_visible_range_and_falls_back_when_unsupported(monkeypatch):
    nodes = _root("editor")
    nodes["editor"] = {"AXRole": "AXTextArea", "AXValue": "old scrollback\nNewest output",
                       "AXVisibleCharacterRange": "range", "AXPosition": (0, 10), "AXSize": (100, 80),
                       "AXChildren": ["scrollback_child"]}
    nodes["scrollback_child"] = {"AXRole": "AXStaticText", "AXValue": "old scrollback"}
    calls = []
    def visible(el, attr, parameter):
        calls.append((el, attr, parameter))
        return "Newest output"
    monkeypatch.setattr(ax, "_ax_get_param", visible)
    assert _read(monkeypatch, nodes) == "Newest output"
    assert calls == [("editor", "AXStringForRange", "range")]
    monkeypatch.setattr(ax, "_ax_get_param", lambda *args: None)
    assert _read(monkeypatch, nodes) == "old scrollback\nNewest output\nold scrollback"
    monkeypatch.setattr(ax, "_ax_get_param", lambda *args: "")
    assert _read(monkeypatch, nodes) == ""
    del nodes["editor"]["AXVisibleCharacterRange"]
    assert _read(monkeypatch, nodes) == "old scrollback\nNewest output\nold scrollback"


def test_viewport_culls_offscreen_and_navigation_but_keeps_main(monkeypatch):
    nodes = _root("nav", "banner", "offscreen", "offscreen_group", "main")
    nodes["nav"] = {"AXRole": "AXLandmarkNavigation", "AXChildren": ["nav_text"]}
    nodes["nav_text"] = {"AXRole": "AXStaticText", "AXValue": "Menu"}
    nodes["banner"] = {"AXRole": "AXLandmarkBanner", "AXChildren": ["banner_list"]}
    nodes["banner_list"] = {"AXRole": "AXList", "AXChildren": ["banner_text"]}
    nodes["banner_text"] = {"AXRole": "AXStaticText", "AXValue": "Site links"}
    nodes["offscreen"] = {"AXRole": "AXStaticText", "AXValue": "Hidden preview",
                          "AXPosition": (250, 0), "AXSize": (100, 30)}
    nodes["offscreen_group"] = {"AXRole": "AXGroup", "AXPosition": (250, 40),
                                "AXSize": (100, 30), "AXChildren": ["unlocated_child"]}
    nodes["unlocated_child"] = {"AXRole": "AXStaticText", "AXValue": "Hidden child"}
    nodes["main"] = {"AXRole": "AXLandmarkMain", "AXChildren": ["body"]}
    nodes["body"] = {"AXRole": "AXStaticText", "AXValue": "Main content",
                     "AXPosition": (10, 20), "AXSize": (100, 20)}
    assert _read(monkeypatch, nodes) == "Main content"


def test_truncation_prefers_located_viewport_text(monkeypatch):
    nodes = _root("unknown", "located")
    nodes["unknown"] = {"AXRole": "AXStaticText", "AXValue": "unknown geometry"}
    nodes["located"] = {"AXRole": "AXStaticText", "AXValue": "visible first",
                        "AXPosition": (10, 10), "AXSize": (100, 20)}
    assert _read(monkeypatch, nodes, max_chars=13) == "visible first"


def test_static_runs_on_same_line_merge_across_parents(monkeypatch):
    nodes = _root("left", "right")
    nodes["left"] = {"AXRole": "AXGroup", "AXChildren": ["a"]}
    nodes["right"] = {"AXRole": "AXGroup", "AXChildren": ["b"]}
    nodes["a"] = {"AXRole": "AXStaticText", "AXValue": "One",
                  "AXPosition": (10, 15), "AXSize": (30, 15)}
    nodes["b"] = {"AXRole": "AXStaticText", "AXValue": "line",
                  "AXPosition": (42, 15), "AXSize": (30, 15)}
    assert _read(monkeypatch, nodes) == "One line"


def test_web_area_uses_bounded_markers_before_word_nodes(monkeypatch):
    monkeypatch.setattr(ax.T, "AX_TEXT_MARKERS", True)  # experimental path, off by default
    nodes = _root("web")
    nodes["web"] = {"AXRole": "AXWebArea", "AXChildren": ["article"]}
    nodes["article"] = {"AXRole": "AXGroup", "AXValue": "So imagine an agent has memory",
                        "AXChildren": ["heading", "link", "last"]}
    nodes["heading"] = {"AXRole": "AXHeading", "AXChildren": ["so"]}
    nodes["link"] = {"AXRole": "AXLink", "AXChildren": ["imagine"]}
    for name, word, x in (("so", "So", 10), ("imagine", "imagine", 35), ("last", "an agent", 105)):
        nodes[name] = {"AXRole": "AXStaticText", "AXValue": word,
                       "AXPosition": (x, 20), "AXSize": (65, 18)}
    calls = []
    monkeypatch.setattr(ax, "_web_visible_text",
                        lambda web, viewport, deadline: calls.append(web) or "So imagine\nan agent\nNext line")
    assert _read(monkeypatch, nodes) == "So imagine\nan agent\nNext line"
    assert calls == ["web"]
    monkeypatch.setattr(ax, "_web_visible_text", lambda *args: "")
    assert _read(monkeypatch, nodes) == "So imagine an agent"


def test_bounded_marker_read_requires_structured_text(monkeypatch):
    class Value:
        @staticmethod
        def valueWithRect_(rect):
            return rect

    monkeypatch.setattr(ax, "NSValue", Value)
    monkeypatch.setattr(ax, "NSMakeRect", lambda *args: args)
    monkeypatch.setattr(ax, "AXTextMarkerRangeCreate", lambda allocator, start, end: (start, end))
    monkeypatch.setattr(ax, "_ax_value", lambda value, kind: value)
    calls = []

    def param(element, attr, value):
        calls.append((attr, value))
        return {"AXStartTextMarkerForBounds": "start", "AXEndTextMarkerForBounds": "end",
                "AXBoundsForTextMarkerRange": ((0, 0), (200, 100)),
                "AXStringForTextMarkerRange": "First line\nSecond line\nThird line"}[attr]

    monkeypatch.setattr(ax, "_ax_get_param", param)
    assert ax._web_visible_text("web", (0, 0, 200, 200), float("inf")) == "First line\nSecond line\nThird line"
    assert calls == [("AXStartTextMarkerForBounds", (0, 0, 200, 200)),
                     ("AXEndTextMarkerForBounds", (0, 0, 200, 200)),
                     ("AXBoundsForTextMarkerRange", ("start", "end")),
                     ("AXStringForTextMarkerRange", ("start", "end"))]
    monkeypatch.setattr(ax, "_ax_get_param", lambda element, attr, value:
                        ((0, 0), (200, 100)) if attr == "AXBoundsForTextMarkerRange" else "word " * 100)
    assert ax._web_visible_text("web", (0, 0, 200, 200), float("inf")) == ""
    monkeypatch.setattr(ax, "_ax_get_param", lambda element, attr, value:
                        ((0, 0), (200, 300)) if attr == "AXBoundsForTextMarkerRange" else "One\nTwo\nThree")
    assert ax._web_visible_text("web", (0, 0, 200, 200), float("inf")) == ""


def test_chromium_article_aria_and_decorative_icons(monkeypatch):
    nodes = _root("article", "text_button", "short_button", "icon_button", "chart")
    nodes["article"] = {"AXRole": "AXGroup", "AXValue": "Whole tweet repeats its descendants",
                        "AXDescription": "Whole tweet repeats its descendants", "AXChildren": ["tweet"]}
    nodes["tweet"] = {"AXRole": "AXStaticText", "AXValue": "Whole tweet repeats its descendants",
                      "AXPosition": (10, 10), "AXSize": (180, 20)}
    nodes["text_button"] = {"AXRole": "AXButton", "AXTitle": "Continue",
                            "AXPosition": (10, 40), "AXSize": (100, 30)}
    nodes["short_button"] = {"AXRole": "AXButton", "AXTitle": "OK",
                             "AXPosition": (120, 40), "AXSize": (35, 30)}
    nodes["icon_button"] = {"AXRole": "AXButton", "AXDescription": "Like",
                            "AXPosition": (10, 80), "AXSize": (32, 32), "AXChildren": ["icon"]}
    nodes["icon"] = {"AXRole": "AXImage", "AXDescription": "Heart"}
    nodes["chart"] = {"AXRole": "AXImage", "AXDescription": "Revenue chart"}
    assert _read(monkeypatch, nodes) == "Whole tweet repeats its descendants\nContinue\nOK\nRevenue chart"


def test_storage_frame_keeps_native_pixels_only_in_memory():
    native = Image.new("RGB", (2400, 1200), "white")
    frame = _storage_frame(native, 1600)
    assert frame.size == (1600, 800)
    assert frame.info["_ocr_native"] is native
    encoded = io.BytesIO()
    frame.save(encoded, "JPEG")
    encoded.seek(0)
    with Image.open(encoded) as stored:
        assert stored.size == (1600, 800)
        assert "_ocr_native" not in stored.info


def test_screen_text_ocr_receives_native_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCRETE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ax, "read_screen_text", lambda *args, **kwargs: "")
    received = []
    monkeypatch.setattr(ocr, "ocr_image", lambda image: received.append(image) or "Visible text")
    observer = ScreenObserver(queue=asyncio.Queue())
    try:
        native = Image.new("RGB", (2400, 1200), "white")
        frame = _storage_frame(native, 1600)
        text, source = observer._screen_text(ax.WindowContext(), frame)
        assert received == [native]
        assert (text, source) == ("Visible text", "ocr")
    finally:
        observer.grabber.close()
