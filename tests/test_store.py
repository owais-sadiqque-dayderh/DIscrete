"""Core store and text utilities tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from discrete import textkit as tk
from discrete.models import ClaimDraft, ObservationDraft
from discrete.store import Store


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


# --- docs/C6.md: storage layer ------------------------------------------------

async def test_reads_do_not_wait_for_the_writer(tmp_db):
    """Bubble's retrieval reads on its own connection (WAL), not behind background writes."""
    import asyncio

    store = await Store.open(tmp_db)
    try:
        await store.add_observation(ObservationDraft(app="A", window_title="T", screen_text="some text here",
                                                     session_id="s"))
        async with store._lock:  # a long write in progress
            rows = await asyncio.wait_for(store.q("SELECT COUNT(*) FROM observations"), 1.0)
        assert rows[0][0] == 1
    finally:
        await store.close()


async def test_minhash_signatures_are_raw_bytes(tmp_db):
    from discrete.config import T

    store = await Store.open(tmp_db)
    try:
        oid, _ = await store.add_observation(ObservationDraft(app="A", screen_text="alpha beta gamma delta",
                                                              session_id="s"))
        (sig,) = (await store.q1("SELECT sig FROM obs_minhash WHERE obs_id=?", (oid,)))
        assert len(sig) == 8 * T.DEDUP_MINHASH_PERMUTATIONS and sig[:1] != b"\x80"
    finally:
        await store.close()


async def test_numbered_migrations_upgrade_a_pre_c6_store(tmp_path):
    """A store written before numbered migrations (user_version 0, pickled MinHash,
    no page columns) upgrades in place and keeps working."""
    import pickle
    import sqlite3

    from datasketch import MinHash

    from discrete.store import db

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    for stmt in db._statements(db._BASE):
        conn.execute(stmt)
    db.m002_draft_attempts_and_digest_counts(conn)
    conn.execute(
        "INSERT INTO observations(id, source, event_kind, session_id, started_at, ended_at, app, window_title, "
        "screen_text, caption, content_hash, text_hash, created_at, updated_at) "
        "VALUES (1, 'screen', 'click', 's', 1, 2, 'Code', 'main.py', 'refactor the storage layer', "
        "'editing code', 'h', 'th', 1, 1)")
    mh = MinHash(num_perm=128)
    mh.update(b"refactor")
    conn.execute("INSERT INTO obs_minhash(obs_id, sig) VALUES (1, ?)", (pickle.dumps(mh),))
    conn.commit()
    conn.close()

    store = await Store.open(path)
    try:
        assert (await store.q1("PRAGMA user_version"))[0] == db.SCHEMA_VERSION
        assert await store.get_meta("schema_version") == str(db.SCHEMA_VERSION)
        row = await store.get_observation(1)
        assert row["page_title"] == "" and row["enrich_state"] == ""
        (sig,) = await store.q1("SELECT sig FROM obs_minhash WHERE obs_id=1")
        assert (tk.minhash_from_bytes(sig) == mh.hashvalues).all()
        assert await store.fts_ids("observation", "storage layer") == [1]  # FTS rebuilt with page_title
    finally:
        await store.close()
    again = await Store.open(path)  # re-opening runs nothing new
    await again.close()


async def test_episode_draft_state_migration_moves_meta_to_columns(tmp_path, monkeypatch):
    from discrete.store import db

    path = str(tmp_path / "episode-meta.db")
    with monkeypatch.context() as legacy:
        legacy.setattr(db, "MIGRATIONS", db.MIGRATIONS[:-1])
        conn = db.connect(path)
    try:
        conn.executemany(
            "INSERT INTO episodes(id,session_id,started_at,ended_at,status,created_at,updated_at) "
            "VALUES (?,'s',1,2,?,3,4)", [(1, 'ready'), (2, 'ready'), (3, 'merged'), (4, 'open')])
        conn.executemany("INSERT INTO meta(key,value) VALUES (?,?)", [
            ('episode_ready_since:1', '123.25'),
            ('episode_ready_since:2', None),
            ('episode_ready_since:3', '12.5'),
            ('episode_boundary_locked:1', '0'),  # Key presence was the lock, not its value.
            ('episode_boundary_locked:3', None),
            ('episode_ready_since:999', '100'),  # Orphans must be removed too.
            ('episode_boundary_locked:999', '1'),
            ('unrelated', 'keep'),
            ('episodeXreadyXsince:1', 'keep'),
        ])
    finally:
        conn.close()

    for _ in range(2):  # Reopening must not overwrite the migrated state.
        store = await Store.open(path)
        try:
            assert (await store.q1("PRAGMA user_version"))[0] == db.SCHEMA_VERSION
            assert await store.get_meta("schema_version") == str(db.SCHEMA_VERSION)
            assert [tuple(r) for r in await store.q(
                "SELECT id,ready_since,boundary_locked,updated_at FROM episodes ORDER BY id")] == [
                (1, 123.25, 1, 4), (2, None, 0, 4), (3, 12.5, 1, 4), (4, None, 0, 4)]
            assert await store.q("SELECT key FROM meta WHERE key GLOB 'episode_ready_since:*' "
                                 "OR key GLOB 'episode_boundary_locked:*'") == []
            assert await store.get_meta('unrelated') == 'keep'
            assert await store.get_meta('episodeXreadyXsince:1') == 'keep'
        finally:
            await store.close()


async def test_model_usage_is_recorded_per_component(tmp_db):
    from discrete.llm import METER, flush_usage
    from discrete.retrieval.store_card import usage_table

    store = await Store.open(tmp_db)
    try:
        METER.drain()
        METER.add("caption", calls=2)
        METER.add_usage("caption", {"prompt_tokens": 1200, "completion_tokens": 80, "cost": 0.0003})
        METER.add("embedding", calls=1, prompt_tokens=30)
        await flush_usage(store)
        await flush_usage(store)  # drained: nothing is counted twice
        usage = await store.usage_by_component()
        assert usage["caption"] == {"calls": 2, "prompt_tokens": 1200, "completion_tokens": 80, "cost": 0.0003}
        table = await usage_table(store)
        assert "caption" in table and "1,200" in table and "total" in table and "0.0003" in table
    finally:
        await store.close()


async def test_alias_cache_sees_new_aliases(tmp_db):
    store = await Store.open(tmp_db)
    try:
        assert await store.all_entity_aliases() == []
        eid = await store.create_entity("Grainient", "software", "grainient")
        assert [r["entity_id"] for r in await store.all_entity_aliases()] == [eid]
        await store.add_alias(eid, "Grainient App", "grainient")
        assert len(await store.all_entity_aliases()) == 2
    finally:
        await store.close()
