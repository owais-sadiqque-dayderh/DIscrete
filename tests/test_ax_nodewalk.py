"""Synthetic AX trees exercise spatial redundancy and semantic table extraction."""

import pytest

from discrete.capture import accessibility as ax


def read(monkeypatch, nodes, *, timeout_ms=1000, max_nodes=500, viewport=(0, 0, 200, 200)):
    monkeypatch.setattr(ax, "_HAS_QUARTZ", True)
    monkeypatch.setattr(ax, "accessibility_granted", lambda: True)
    monkeypatch.setattr(ax, "AXUIElementCreateApplication", lambda pid: "app", raising=False)
    monkeypatch.setattr(ax, "_enable_full_tree", lambda *args: None)
    monkeypatch.setattr(ax, "_ax_get", lambda el, attr: nodes.get(el, {}).get(attr))
    monkeypatch.setattr(ax, "_ax_value", lambda value, kind: value)
    monkeypatch.setattr(ax.T, "AX_TEXT_MARKERS", False)
    return ax.read_screen_text(1, max_nodes=max_nodes, timeout_ms=timeout_ms, max_chars=16000,
                               viewport=viewport)


def root(*children):
    return {"app": {"AXFocusedWindow": "window"},
            "window": {"AXRole": "AXWindow", "AXPosition": (0, 0), "AXSize": (200, 200),
                       "AXChildren": list(children)}}


def text(value, x=10, y=20, w=100, h=15):
    return {"AXRole": "AXStaticText", "AXValue": value,
            "AXPosition": (x, y), "AXSize": (w, h)}


def test_neighbouring_duplicate_projections_and_link_label(monkeypatch):
    nodes = root("link", "other", "copy")
    nodes["link"] = {"AXRole": "AXLink", "AXTitle": "Story headline",
                     "AXDescription": "Story headline", "AXChildren": ["painted"]}
    nodes["painted"] = text("Story headline")
    nodes["other"] = text("Read more", y=45)
    nodes["copy"] = text("Story   headline")
    assert read(monkeypatch, nodes) == "Story headline\nRead more"


def test_repeated_short_comments_and_unlocated_siblings_survive(monkeypatch):
    nodes = root("comment1", "comment2")
    nodes["comment1"] = {"AXRole": "AXGroup", "AXChildren": ["a"]}
    nodes["comment2"] = {"AXRole": "AXGroup", "AXChildren": ["b"]}
    nodes["a"] = text("Thanks", y=20)
    nodes["b"] = text("Thanks", y=80)
    assert read(monkeypatch, nodes) == "Thanks\nThanks"
    nodes["a"].pop("AXPosition")
    nodes["b"].pop("AXPosition")
    assert read(monkeypatch, nodes) == "Thanks\nThanks"


def test_repeated_words_at_different_horizontal_positions_survive(monkeypatch):
    nodes = root("a", "b")
    nodes["a"] = text("very", x=10, w=30)
    nodes["b"] = text("very", x=42, w=30)
    assert read(monkeypatch, nodes) == "very very"


def test_dedup_neighbourhood_is_bounded(monkeypatch):
    names = ["first", *(f"middle{i}" for i in range(9)), "last"]
    nodes = root(*names)
    for i, name in enumerate(names):
        nodes[name] = text(f"Item {i}", y=5 + i * 15)
    nodes["first"] = text("Same content")
    nodes["last"] = text("Same content")
    assert read(monkeypatch, nodes).count("Same content") == 2


def test_frame_fallback_and_web_viewport_cull_above_content(monkeypatch):
    nodes = root("web")
    nodes["web"] = {"AXRole": "AXWebArea", "AXFrame": ((0, 50), (200, 150)),
                    "AXChildren": ["above", "body", "below", "strip", "unknown"]}
    nodes["above"] = text("Behind browser chrome", y=20)
    nodes["body"] = {"AXRole": "AXStaticText", "AXValue": "Visible body " * 10,
                     "AXFrame": ((10, 60), (150, 80))}
    nodes["below"] = text("Below viewport", y=220)
    nodes["strip"] = text("Clipped to edge", y=199, h=1)
    nodes["unknown"] = {"AXRole": "AXStaticText", "AXValue": "Missing geometry"}
    assert read(monkeypatch, nodes) == ("Visible body " * 10).strip()


def test_tiny_visible_text_gets_only_bounded_fallback(monkeypatch):
    nodes = root("hidden", "visible", "unknown")
    nodes["hidden"] = text("Offscreen remainder " * 100, y=220)
    nodes["visible"] = text("Loading")
    nodes["unknown"] = {"AXRole": "AXStaticText", "AXValue": "Another hidden remainder"}
    result = read(monkeypatch, nodes)
    assert result.startswith("Loading\nOffscreen remainder")
    assert len(result) <= len("Loading\n") + ax._MAX_DEFERRED_CHARS
    assert "Another hidden" not in result
    nodes["hidden"]["AXValue"] = "One offscreen line"
    nodes["visible"]["AXValue"] = "Visible content " * 20
    assert read(monkeypatch, nodes) == ("Visible content " * 20).strip()


@pytest.mark.parametrize("role", ["AXTable", "AXGrid"])
def test_table_rows_cells_empty_columns_and_projection_dedup(monkeypatch, role):
    nodes = root("table")
    nodes["table"] = {"AXRole": role, "AXChildren": ["column", "row1", "row2"]}
    nodes["column"] = {"AXRole": "AXColumn", "AXChildren": ["c11", "c21"]}
    for row, y in (("row1", 20), ("row2", 60)):
        names = [f"c{row[-1]}{j}" for j in range(1, 4)]
        nodes[row] = {"AXRole": "AXRow", "AXPosition": (0, y), "AXSize": (190, 20),
                      "AXChildren": names}
        for j, name in enumerate(names):
            nodes[name] = {"AXRole": "AXCell", "AXPosition": (j * 60, y), "AXSize": (60, 20),
                           "AXChildren": [] if j == 1 else [name + "text"]}
            nodes[name + "text"] = text("Same" if j == 0 else "Last", x=j * 60, y=y, w=50)
    assert read(monkeypatch, nodes) == "Same |  | Last\nSame |  | Last"


def test_column_only_table_is_not_discarded(monkeypatch):
    nodes = root("table")
    nodes["table"] = {"AXRole": "AXTable", "AXChildren": ["column"]}
    nodes["column"] = {"AXRole": "AXColumn", "AXChildren": ["a"]}
    nodes["a"] = text("Column-only content")
    assert read(monkeypatch, nodes) == "Column-only content"


def test_expired_deadline_and_node_budget_stop_walk(monkeypatch):
    nodes = root("a", "b")
    nodes["a"] = text("First")
    nodes["b"] = text("Second", y=60)
    assert read(monkeypatch, nodes, timeout_ms=0) == ""
    assert read(monkeypatch, nodes, max_nodes=2) == "First"


def test_window_frame_is_used_when_no_viewport_is_passed(monkeypatch):
    nodes = root("visible", "offscreen")
    nodes["visible"] = text("Visible text " * 10)
    nodes["offscreen"] = text("Offscreen", y=250)
    assert read(monkeypatch, nodes, viewport=None) == ("Visible text " * 10).strip()


def test_control_label_sibling_dedup_stays_within_its_local_group(monkeypatch):
    nodes = root("choice1", "choice2")
    for i, y in ((1, 20), (2, 70)):
        group, control, label, painted = (f"choice{i}", f"control{i}", f"label{i}", f"painted{i}")
        nodes[group] = {"AXRole": "AXGroup", "AXChildren": [control, label]}
        nodes[control] = {"AXRole": "AXRadioButton", "AXTitle": "Standard",
                          "AXPosition": (10, y), "AXSize": (20, 20)}
        nodes[label] = {"AXRole": "AXGroup", "AXChildren": [painted]}
        nodes[painted] = text("Standard", x=34, y=y + 2, w=60)
    assert read(monkeypatch, nodes) == "Standard\nStandard"


def test_plain_group_row_header_is_a_cell(monkeypatch):
    nodes = root("table")
    nodes["table"] = {"AXRole": "AXTable", "AXChildren": ["row"]}
    nodes["row"] = {"AXRole": "AXRow", "AXChildren": ["rank", "cell"]}
    nodes["rank"] = {"AXRole": "AXGroup", "AXPosition": (0, 20), "AXSize": (30, 20),
                     "AXChildren": ["ranktext"]}
    nodes["ranktext"] = text("1", x=10, w=10)
    nodes["cell"] = {"AXRole": "AXCell", "AXPosition": (30, 20), "AXSize": (150, 20),
                     "AXChildren": ["country"]}
    nodes["country"] = text("India", x=40)
    assert read(monkeypatch, nodes) == "1 | India"


def test_wrapper_around_semantic_cells_does_not_add_a_phantom_column(monkeypatch):
    nodes = root("table")
    nodes["table"] = {"AXRole": "AXTable", "AXChildren": ["row"]}
    nodes["row"] = {"AXRole": "AXRow", "AXChildren": ["wrapper"]}
    nodes["wrapper"] = {"AXRole": "AXGroup", "AXChildren": ["a", "b"]}
    for name, x in (("a", 0), ("b", 80)):
        nodes[name] = {"AXRole": "AXCell", "AXPosition": (x, 20), "AXSize": (80, 20),
                       "AXChildren": [name + "text"]}
        nodes[name + "text"] = text(name.upper(), x=x)
    assert read(monkeypatch, nodes) == "A | B"


def test_hidden_subtree_cannot_consume_budget_after_fallback_is_full(monkeypatch):
    nodes = root("hidden", "visible")
    names = [f"hidden{i}" for i in range(100)]
    nodes["hidden"] = {"AXRole": "AXGroup", "AXPosition": (0, 250), "AXSize": (100, 100),
                       "AXChildren": names}
    for name in names:
        nodes[name] = text("One short offscreen line", y=250)
    nodes["visible"] = text("Visible content " * 10)
    assert read(monkeypatch, nodes, max_nodes=20) == ("Visible content " * 10).strip()
