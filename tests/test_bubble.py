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
