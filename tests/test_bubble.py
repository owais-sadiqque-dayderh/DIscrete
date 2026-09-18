"""Bubble (planS3 §12): query extraction, evidence pack, streaming insertion, grounding path."""

from __future__ import annotations

import pytest

from discrete.bubble.engine import BubbleEngine, field_query, needs_space, render_records
from discrete.bubble.keys import parse_hotkey
from discrete.bubble.overlay_client import Overlay
from discrete.capture import suppress
from discrete.process.embed_worker import drain_outbox
from discrete.store import ClaimDraft


def test_field_query_takes_last_paragraph_and_sentence_boundary():
    draft = "Dear team,\n\nThanks for the notes. " + "Some filler words here. " * 40 + "What did I decide about the logo?"
    q = field_query(draft, max_chars=120)
    assert q.endswith("What did I decide about the logo?")
    assert len(q) <= 120 and not q.startswith("ome")  # starts on a sentence, not mid-word
    assert field_query("  what editor do I use? ") == "what editor do I use?"


def test_needs_space():
    assert needs_space("My editor is", "Neovim")
    assert not needs_space("My editor is ", "Neovim")
    assert not needs_space("Hello", ", world")
    assert not needs_space("", "x")


def test_parse_hotkey():
    assert parse_hotkey("<alt>+<tab>") == (1 << 19, 48)
    assert parse_hotkey("<cmd>+<nope>")[1] == -1


def test_suppress_hold_spans_whole_run():
    suppress.hold()
    assert suppress.is_suppressed()
    suppress.release(tail_s=0.0)
    assert not suppress.is_suppressed()


def test_disabled_overlay_is_a_noop():
    ov = Overlay(enabled=False)
    ov.start()
    ov.show((1, 2, 3, 4))
    ov.status("x")
    ov.close()


async def _seed(store):
    cid = await store.create_claim(ClaimDraft(kind="preference", modality="engaged",
                                              text="The user's preferred editor for Python is Neovim"))
    await drain_outbox(store)
    return cid


async def test_grounded_answer_uses_records_and_types_after_space(store, fake):
    cid = await _seed(store)
    engine = BubbleEngine(store, user_name="Owais")
    p = await engine.prepare("what is my preferred editor for python?", app="Notes")
    assert p.mode == "grounded"
    assert any(h.ref_kind == "claim" and h.ref_id == cid for h in p.result.hits)
    msgs = engine.messages(p)
    assert f"[claim {cid}" in msgs[1]["content"] and "Neovim" in msgs[1]["content"]

    fake.writer_reply = "Neovim."
    typed: list[str] = []
    out = await engine.write(p, emit=lambda s: typed.append(s) or True)
    assert out == " Neovim." and "".join(typed) == out


async def test_empty_store_skips_writer(store, fake):
    p = await BubbleEngine(store).prepare("who is my dentist?")
    assert p.mode == "empty"
    assert "(no records)" in render_records(p.result)


async def test_streamed_chunks_break_on_word_boundaries(store, fake):
    await _seed(store)
    engine = BubbleEngine(store)
    p = await engine.prepare("My python editor is", app="Notes")
    pieces = ["Neo", "vim, a", " modal edi", "tor I con", "figured last", " week."]

    async def stream(ep, messages, *, temperature, max_tokens):
        for x in pieces:
            yield x

    fake.stream = stream
    typed: list[str] = []
    out = await engine.write(p, emit=lambda s: typed.append(s) or True)
    assert out == " " + "".join(pieces)
    for chunk in typed[:-1]:
        nxt = typed[typed.index(chunk) + 1]
        assert nxt[0] in " \n", f"split mid-word: {chunk!r}|{nxt!r}"


async def test_emit_failure_stops_writing(store, fake):
    await _seed(store)
    engine = BubbleEngine(store)
    p = await engine.prepare("what is my preferred editor for python?")
    fake.writer_reply = "one two three four five six seven eight nine ten eleven twelve"
    calls = []
    out = await engine.write(p, emit=lambda s: calls.append(s) and False)
    assert out == "" and len(calls) == 1


@pytest.mark.parametrize("q", ["fix the grammar in this paragraph"])
async def test_edit_skips_retrieval(store, fake, q):
    p = await BubbleEngine(store).prepare(q)
    assert p.mode == "edit" and not p.result.hits


# --- AX struct normalizers (accessibility.py): pyobjc may bridge AXValue structs as
# objects (.location/.origin) or as plain (nested) tuples; both must be accepted so the
# hotkey handler never crashes (insights.md §caret). ---

class _Loc:
    def __init__(self, location):
        self.location = location


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Sz:
    def __init__(self, w, h):
        self.width, self.height = w, h


class _Rect:
    def __init__(self, x, y, w, h):
        self.origin, self.size = _Pt(x, y), _Sz(w, h)


def test_range_location_accepts_object_and_tuple():
    from discrete.capture.accessibility import _range_location

    assert _range_location(_Loc(7)) == 7
    assert _range_location((7, 3)) == 7  # (location, length) tuple form
    assert _range_location(None) is None
    assert _range_location(()) is None


def test_point_size_rect_accept_object_and_tuple():
    from discrete.capture.accessibility import _point_xy, _rect_xywh, _size_wh

    assert _point_xy(_Pt(1.0, 2.0)) == (1.0, 2.0)
    assert _point_xy((1.0, 2.0)) == (1.0, 2.0)
    assert _size_wh(_Sz(3.0, 4.0)) == (3.0, 4.0)
    assert _size_wh((3.0, 4.0)) == (3.0, 4.0)
    assert _rect_xywh(_Rect(1, 2, 3, 4)) == (1.0, 2.0, 3.0, 4.0)
    assert _rect_xywh(((1, 2), (3, 4))) == (1.0, 2.0, 3.0, 4.0)
    assert _point_xy(None) is None and _size_wh(None) is None and _rect_xywh(None) is None


# --- insights.md §7.32-7.33 ---------------------------------------------------

def test_invisible_only_field_is_empty():
    assert field_query("​") == ""
    assert field_query("﻿  ​\n") == ""
    assert field_query("hi​ there") == "hi there"


async def test_empty_field_never_retrieves(store, fake):
    from discrete.bubble.engine import EmptyField

    with pytest.raises(EmptyField):
        await BubbleEngine(store).prepare("​")
    assert "writer" not in fake.calls


def test_page_label_from_navigation_caption():
    from discrete import textkit as tk

    cap = "The user switched from viewing Hacker News threads to a GitHub repository page for 'frank-386'."
    assert tk.page_label("Space 1", caption=cap) == "frank-386 (GitHub)"
    assert tk.destination_from_caption(
        "The user clicked on a link to view details about a specific post regarding auto-vectorization."
    ) == "auto-vectorization"


async def test_enumerate_answer_is_deterministic_list(store, fake):
    from discrete.bubble.engine import format_enumerate_answer
    from discrete.retrieval.search import retrieve
    from discrete.store import ObservationDraft

    t = 1_789_000_000.0
    await store.add_observation(ObservationDraft(
        app="Arc", window_title="Space 1", url="",
        screen_text="Hacker News new past comments",
        caption="The user switched from Hacker News to the Linux From Scratch website.",
        started_at=t, ended_at=t + 10, session_id="s"))
    await drain_outbox(store)
    res = await retrieve(store, "list all articles I opened from hacker news")
    text = format_enumerate_answer(res)
    assert text.startswith("- Linux From Scratch")
    assert "writer" not in fake.calls


async def test_enumerate_lists_the_trail_after_a_matched_listing(store, fake):
    """Pages opened *from* a listing never mention it; they come from the trail."""
    from discrete.retrieval.search import retrieve
    from discrete.store import ObservationDraft

    t = 1_789_000_000.0

    def page(dt, title, url, text, app="Arc"):
        return ObservationDraft(app=app, window_title=title, url=url, site=url.split("/")[2] if url else "",
                                screen_text=text, caption=f"viewing {title}", started_at=t + dt,
                                ended_at=t + dt + 20, clicks=1, session_id="s")

    for d in (
        page(0, "Hacker News", "https://news.ycombinator.com/news", "Hacker News new past comments ask show jobs"),
        page(40, "Blueprint | Founders, Inc.", "https://f.inc/blueprint", "Founders program for builders"),
        page(90, "Hacker News", "https://news.ycombinator.com/news", "Hacker News new past comments ask show"),
        page(130, "Linux From Scratch", "https://www.linuxfromscratch.org/", "build your own linux system"),
        page(9000, "Weather today", "https://weather.example/", "sunny forecast"),
        page(60, "notes.md", "", "todo list", app="Code"),
    ):
        await store.add_observation(d)
    await drain_outbox(store)
    res = await retrieve(store, "list all the articles I opened from hacker news")
    assert res.plan.intent == "enumerate"
    titles = [(h.title, h.status) for h in res.hits]
    assert ("Hacker News", "matched") in titles
    assert ("Blueprint | Founders, Inc.", "trail") in titles and ("Linux From Scratch", "trail") in titles
    assert all(t != "Weather today" for t, _ in titles)      # outside the trail window
    assert all(t != "notes.md" for t, _ in titles)           # another app, not a trail
    assert sum(1 for t, _ in titles if t == "Hacker News") == 1  # repeat visits collapse by URL
    pack = render_records(res)
    assert "★" in pack and "→" in pack and "Blueprint" in pack
