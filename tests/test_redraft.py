"""Draft-only rebuilds preserve captures and episode boundaries on migrated stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.redraft import DERIVED, DERIVED_KINDS, redraft
from discrete.evaluation.invariants import check_store
from discrete.models import ClaimDraft
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval.store_card import store_card
from discrete.store import Store, db
from tests.test_tasks import ready


async def test_redraft_clears_tasks_and_preserves_merged_episode_boundaries(store, tmp_db, tmp_path, fake):
    first, first_ids = await ready(store)
    tombstone, merged_ids = await ready(store, dt=20)
    fake.draft = {"title": "Agent Memory Repo research", "summary": "Researched the design", "claims": []}
    assert await ProcessPipeline(store).draft_episode(first)
    assert (await store.get_episode(tombstone))["status"] == "merged"
    # A formerly drafted neighbor would merge with the survivor after resetting
    # attempts, unless the redraft explicitly keeps the original boundaries.
    neighbor, neighbor_ids = await ready(store, dt=40)
    await store.close_episode(neighbor, title=fake.draft["title"], summary=fake.draft["summary"],
                              apps=[], sites=[], engaged_fraction=1)
    await store.create_claim(ClaimDraft(kind="event", text="Researched Agent Memory Repo", episode_id=first,
                                      evidence=[{"observation_id": first_ids[0], "quote": "Unique screen"}]))
    entity = await store.create_entity("Stale Entity", "software", "stale-entity")
    assert await store.fts_ids("entity", "Stale") == [entity]
    await store.execute("INSERT INTO entity_candidates VALUES ('repo','Repo','[]',100,200)")
    await store.save_draft_chunk(neighbor, 0, "stale", {"title": "stale"})
    await store.record_draft_coverage(neighbor, "stale", 0, neighbor_ids)
    for kind in DERIVED_KINDS:
        await store.save_embeddings([(kind, first, [1., 0.], "stale")])
        await store.execute("INSERT OR REPLACE INTO index_outbox VALUES (?,?,100,1)", (kind, first))
    await store.execute("INSERT INTO index_outbox VALUES ('observation',?,100,0)", (first_ids[0],))
    await store.set_meta("last_dream", "100")
    await store.set_meta("last_consolidation", "100")
    assert (await store.stats())["tasks"] > 0

    preserved_tables = ("observations", "sightings", "obs_minhash", "token_df")
    preserved = {table: [tuple(r) for r in await store.q(f"SELECT * FROM {table}")] for table in preserved_tables}
    boundaries = [tuple(r) for r in await store.q("SELECT id,session_id,started_at,ended_at FROM episodes ORDER BY id")]
    original_episodes = [tuple(r) for r in await store.q("SELECT * FROM episodes ORDER BY id")]
    original_tasks = [tuple(r) for r in await store.q("SELECT * FROM tasks")]
    vectors_version = int(await store.get_meta("vectors:episode"))

    output = tmp_path / "redrafted.db"
    counts = redraft(Path(tmp_db), output)
    assert counts["episodes"] == 2 and counts["tasks"] == len(original_tasks)
    assert counts["episode_tasks"] == 2
    copied = await Store.open(str(output))
    try:
        for table in DERIVED:
            assert await copied.q(f"SELECT * FROM {table}") == []
        for table in preserved_tables:
            assert [tuple(r) for r in await copied.q(f"SELECT * FROM {table}")] == preserved[table]
        assert [tuple(r) for r in await copied.q(
            "SELECT id,session_id,started_at,ended_at FROM episodes ORDER BY id")] == boundaries
        assert [(r["status"], r["draft_attempts"]) for r in await copied.q(
            "SELECT * FROM episodes ORDER BY id")] == [("ready", 0), ("merged", 0), ("ready", 0)]
        assert {r[0] for r in await copied.q("SELECT DISTINCT ref_kind FROM embeddings")} == {"observation"}
        assert {r[0] for r in await copied.q("SELECT DISTINCT ref_kind FROM index_outbox")} == {"observation"}
        assert int(await copied.get_meta("vectors:episode")) == vectors_version + 1
        assert await copied.q("SELECT key FROM meta WHERE key LIKE 'draft_coverage:%'") == []
        assert not await copied.get_meta("last_dream")
        assert not await copied.get_meta("last_consolidation")
        assert await copied.fts_ids("episode", "research") == []
        assert await copied.fts_ids("claim", "Researched") == []
        assert await copied.fts_ids("entity", "Stale") == []
        assert await copied.fts_ids("observation", "Unique screen")
        assert tuple(await copied.q1("PRAGMA integrity_check")) == ("ok",)
        assert await copied.q("PRAGMA foreign_key_check") == []

        outcomes = await ProcessPipeline(copied).process_ready()
        assert [o.episode_id for o in outcomes] == [first, neighbor]
        assert (await copied.get_episode(tombstone))["status"] == "merged"
        assert [tuple(r) for r in await copied.q(
            "SELECT id,session_id,started_at,ended_at FROM episodes ORDER BY id")] == boundaries
        assert {r["episode_id"] for r in await copied.get_observations(first_ids + merged_ids)} == {first}
        assert {r["episode_id"] for r in await copied.get_observations(neighbor_ids)} == {neighbor}
        assert check_store(output).passed
        stats = await copied.stats()
        assert stats["task_episodes"] == 2 and stats["tasks"] > 0
        assert "task coverage 2/2 closed episodes" in await store_card(copied)
        assert not await copied.get_meta(f"draft_coverage:{tombstone}")
    finally:
        await copied.close()
    assert [tuple(r) for r in await store.q("SELECT * FROM episodes ORDER BY id")] == original_episodes
    assert [tuple(r) for r in await store.q("SELECT * FROM tasks")] == original_tasks
    assert await store.fts_ids("entity", "Stale") == [entity]


def test_redraft_handles_pre_task_migrations_and_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "MIGRATIONS", db.MIGRATIONS[:-1])
    source = tmp_path / "legacy.db"
    conn = db.connect(str(source))
    conn.close()
    output = tmp_path / "redrafted.db"
    assert redraft(source, output)["episodes"] == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        redraft(source, output)
    assert output.read_bytes() == before
