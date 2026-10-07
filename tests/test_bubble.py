"""Bubble (planS3 §12): query extraction, evidence pack, streaming insertion, grounding path."""

from __future__ import annotations

import pytest

from discrete.bubble.engine import BubbleEngine, field_query, needs_space, render_records
from discrete.bubble.keys import parse_hotkey
from discrete.bubble.overlay_client import Overlay
from discrete.capture import suppress
from discrete.models import ClaimDraft
from discrete.process.embed_worker import drain_outbox


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


def test_page_label_comes_from_structured_fields():
    """Labels come from the vision call's page_title and ground truth, never caption prose."""
    from discrete.retrieval.pages import page_identity, page_label

    assert page_label("Space 1", page_title="frank-386") == "frank-386"
    assert page_label("Space 1", url="https://github.com/acme/frank-386") == "frank 386 (github.com)"
    assert page_label("Rust Book", url="https://doc.rust-lang.org/book/") == "Rust Book"
    # A caption is never read for the label, however it is worded.
    assert page_label("Space 1", site="example.org") == "example.org"
    # A generic window title with no page title is never a shared identity.
    assert page_identity("Space 1", caption="a") != page_identity("Space 1", caption="b")
    assert page_identity("Space 1", page_title="Linux From Scratch", caption="a") == \
        page_identity("Space 1", page_title="Linux From Scratch", caption="b")


async def test_enumerate_answer_is_deterministic_list(store, fake):
    from discrete.bubble.engine import format_enumerate_answer
    from discrete.models import ObservationDraft
    from discrete.retrieval.hybrid import retrieve

    t = 1_789_000_000.0
    await store.add_observation(ObservationDraft(
        app="Arc", window_title="Space 1", url="",
        screen_text="build your own linux system from source",
        caption="The user is reading a guide.", page_title="Linux From Scratch", navigated_from="Hacker News",
        started_at=t, ended_at=t + 10, session_id="s"))
    await drain_outbox(store)
    res = await retrieve(store, "list all articles I opened from hacker news")
    text = format_enumerate_answer(res)
    assert text.startswith("- Linux From Scratch")
    assert "writer" not in fake.calls


async def test_enumerate_lists_the_trail_after_a_matched_listing(store, fake):
    """Pages opened *from* a listing never mention it; they come from the trail."""
    from discrete.models import ObservationDraft
    from discrete.retrieval.hybrid import retrieve

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


def test_opened_items_drop_listings_and_pages_from_other_sources():
    from discrete.models import Hit, QueryPlan
    from discrete.retrieval.enumerate import shape_enumerate_items

    def hit(title, status, **kw):
        return Hit("observation", abs(hash(title)) % 1000, title, "", status=status, strong=True, **kw)

    hits = [hit("Front Page", "matched", is_listing=True),
            hit("A deep dive into allocators", "trail", navigated_from="Front Page"),
            hit("Search results", "trail", navigated_from="Web search"),
            hit("Pricing", "trail"),
            hit("Space 1", "trail")]
    plan = QueryPlan(raw="list the articles I opened from front page", intent="enumerate")
    # "Pricing" was only open nearby in time; the allocators page has a recorded source.
    assert [h.title for h in shape_enumerate_items(hits, plan)] == ["A deep dive into allocators"]
    # With no provenance recorded anywhere, the time trail is all there is.
    bare = [hit("Front Page", "matched", is_listing=True), hit("Pricing", "trail"), hit("Changelog", "trail")]
    assert [h.title for h in shape_enumerate_items(bare, plan)] == ["Pricing", "Changelog"]
    # A question that is not about opened items keeps every line.
    assert len(shape_enumerate_items(hits, QueryPlan(raw="list every page about allocators"))) == 5


def test_every_cli_command_has_a_handler():
    from discrete.cli import build_parser

    p = build_parser()
    sub = next(a for a in p._actions if a.dest == "cmd")
    needs = {"ask": ["x"], "query": ["x"], "why": ["x"], "answer": ["1", "x"], "history": ["1"], "revert": ["1"],
             "forget": ["1"], "correct": ["1", "x"]}
    for name in sub.choices:
        args = p.parse_args([name, *needs.get(name, [])])
        assert callable(args.func), name


@pytest.mark.parametrize("mode", ["grounded", "thin", "empty"])
@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("whitespace", [True, False])
async def test_empty_writer_retries_then_answers_or_quotes_record(store, fake, monkeypatch, mode, recover, whitespace):
    from discrete import llm
    from discrete.bubble.engine import Prepared
    from discrete.config import T
    from discrete.models import Hit, QueryPlan, RetrievalResult
    from discrete.retrieval.grounding import GroundingDecision

    query = "What size range does EmbeddingGemma 2 come in?"
    hits = [] if mode == "empty" else [Hit("observation", 1, "X post", "Menu\nEmbeddingGemma 2 comes in 300M and 1B sizes.",
                                          occurred_at=1_791_300_000)]
    p = Prepared(query, query, "terminal", "answer", RetrievalResult(query, QueryPlan(raw=query, terms=["embeddinggemma"]), hits),
                 GroundingDecision(mode, 1, 0))
    calls = 0
    elapsed = 0.0

    async def stream(ep, messages, **kwargs):
        nonlocal calls, elapsed
        calls += 1
        elapsed += 2.0
        if calls == 2 and recover:
            yield "300M and 1B."
        elif whitespace:
            yield " \n\t"

    monkeypatch.setattr(fake, "stream", stream, raising=False)
    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 0.0)
    monkeypatch.setattr(llm.time, "perf_counter", lambda: elapsed)
    typed = []
    out = await BubbleEngine(store).write(p, lambda text: typed.append(text) or True)
    assert calls == 2 and "".join(typed) == out
    assert p.timings["first_chunk_s"] == 4.0
    if recover:
        assert out.strip() == "300M and 1B."
    elif mode == "empty":
        assert out.strip() == "(nothing on file about that)"
    else:
        assert out.strip().startswith("From X post (")
        assert out.endswith(": EmbeddingGemma 2 comes in 300M and 1B sizes.")
        assert "Menu" not in out
