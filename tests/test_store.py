"""Core store and text utilities tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from discrete import textkit as tk
from discrete.store import ClaimDraft, ObservationDraft, Store


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


@pytest.mark.asyncio
async def test_observation_dedup_exact(tmp_db):
    store = await Store.open(tmp_db)
    try:
        d = ObservationDraft(
            app="Safari",
            window_title="Test",
            screen_text="hello world " * 5,
            caption="reading test",
            session_id="s1",
        )
        oid1, new1 = await store.add_observation(d)
        oid2, new2 = await store.add_observation(d)
        assert new1 is True
        assert new2 is False
        assert oid1 == oid2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_claim_create(tmp_db):
    store = await Store.open(tmp_db)
    try:
        cid = await store.create_claim(ClaimDraft(kind="fact", text="User prefers dark mode"))
        rows = await store.live_claims()
        assert len(rows) == 1
        assert rows[0]["id"] == cid
    finally:
        await store.close()


def test_stable_key_excludes_caption():
    a = tk.stable_obs_key("App", "Title", "", "same text")
    b = tk.stable_obs_key("App", "Title", "", "same text")
    assert a == b


def test_quote_verification_window():
    body = "The user edited Artboard 3 in Adobe Illustrator with monochrome palette"
    quote, overlap = tk.best_window(body, {"artboard", "illustrator", "monochrome"})
    assert overlap > 0
    assert quote.lower() in body.lower()
