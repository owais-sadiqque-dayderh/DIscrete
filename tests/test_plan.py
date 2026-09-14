"""Additional plan coverage tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from discrete.process.confidence import compute_confidence
from discrete.process.validation import atomic, subject_ok, verify_quote
from discrete.retrieval.query import fallback_plan
from discrete.store import ClaimDraft, ObservationDraft, Store


@pytest.mark.asyncio
async def test_episode_lifecycle(tmp_db):
    store = await Store.open(tmp_db)
    try:
        eid = await store.open_episode("sess", 1000.0)
        d = ObservationDraft(app="Code", window_title="main.py", screen_text="def foo(): pass", session_id="sess")
        oid, _ = await store.add_observation(d)
        await store.assign_episode([oid], eid)
        await store.mark_episode_ready(eid)
        ready = await store.episodes_ready_for_draft()
        assert len(ready) == 1
    finally:
        await store.close()


def test_confidence_derivation():
    assert compute_confidence(["engaged"], 1) > 0.2
    assert compute_confidence(["stated"], 1) > compute_confidence(["passive"], 1)


def test_query_aggregate_intent():
    plan = fallback_plan("what design tools have I used")
    assert plan.intent == "aggregate"


def test_atomic_rejects_long():
    assert not atomic("x" * 300)


def test_quote_verification():
    body = "Adobe Illustrator Artboard 3"
    q, ok = verify_quote("Artboard 3", body, "")
    assert ok
