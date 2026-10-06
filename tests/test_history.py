"""The memory change log: every decision is recorded with a reason and can be undone."""

from __future__ import annotations

import json

import pytest

from discrete.models import ClaimDraft, ClaimWrite, ObservationDraft
from discrete.store import RevertConflict


async def _obs(store, text: str, n: int = 0) -> int:
    oid, _ = await store.add_observation(ObservationDraft(
        app="Arc", window_title=f"Page {n}", screen_text=f"{text} {n} " * 4, caption=text, session_id="s",
        started_at=1000.0 + n, ended_at=1001.0 + n, keys=1))
    return oid


async def _claim(store, text: str, kind: str = "preference", n: int = 0) -> int:
    oid = await _obs(store, text, n)
    return await store.create_claim(ClaimDraft(kind=kind, text=text, modality="engaged",
                                               evidence=[{"observation_id": oid, "quote": text}]))


async def test_every_add_is_logged_with_its_actor(store):
    cid = await _claim(store, "The user prefers dark mode")
    events = await store.events_for_claim(cid)
    assert [(e["op"], e["actor"]) for e in events] == [("add", "user")]
    assert json.loads(events[0]["created"]) == [cid]


async def test_a_merge_is_logged_and_reverting_it_restores_the_inputs(store):
    a = await _claim(store, "The user prefers dark mode in editors", n=1)
    b = await _claim(store, "The user likes dark themes in code editors", n=2)
    cluster = await store.get_claims([a, b])
    [merged] = await store.consolidate_cluster(cluster, [{"text": "The user prefers dark mode in code editors",
                                                          "input_ids": [a, b]}])
    assert {r["status"] for r in await store.get_claims([a, b])} == {"superseded"}
    [merge] = await store.recent_events(op="merge")
    assert merge["actor"] == "consolidate" and json.loads(merge["created"]) == [merged]

    rid = await store.revert_event(merge["id"])

    assert {r["status"] for r in await store.get_claims([a, b])} == {"live"}
    assert (await store.get_claim(merged))["status"] == "reverted"
    assert (await store.get_event(merge["id"]))["undone_by"] == rid
    # The revived inputs are searchable again: queued for re-embedding.
    queued = {r[0] for r in await store.q("SELECT ref_id FROM index_outbox WHERE ref_kind='claim'")}
    assert {a, b} <= queued


async def test_a_revert_can_be_reverted(store):
    a = await _claim(store, "The user prefers tea", n=1)
    b = await _claim(store, "The user drinks tea daily", n=2)
    [merged] = await store.consolidate_cluster(await store.get_claims([a, b]),
                                               [{"text": "The user prefers tea and drinks it daily",
                                                 "input_ids": [a, b]}])
    [merge] = await store.recent_events(op="merge")
    rid = await store.revert_event(merge["id"])
    await store.revert_event(rid)
    assert (await store.get_claim(merged))["status"] == "live"
    assert {r["status"] for r in await store.get_claims([a, b])} == {"superseded"}


async def test_revert_refuses_when_a_later_change_touched_the_same_claims(store):
    a = await _claim(store, "The user prefers vim", n=1)
    eid = await store.forget_claim(a)
    first = (await store.events_for_claim(a))[0]["id"]  # the add
    with pytest.raises(RevertConflict):
        await store.revert_event(first)
    await store.revert_event(first, force=True)
    assert (await store.get_claim(a))["status"] == "reverted"
    assert eid


async def test_forget_retires_suppresses_and_reverting_lifts_both(store):
    cid = await _claim(store, "The user prefers aisle seats")
    eid = await store.forget_claim(cid, reason="test note, not mine")
    assert (await store.get_claim(cid))["status"] == "forgotten"
    assert await store.suppressed("preference", "The user prefers aisle seats")
    assert await store.suppressed("preference", "the user prefers aisle seats.")
    assert not await store.suppressed("preference", "The user prefers window seats on long flights")
    assert not await store.suppressed("fact", "The user prefers aisle seats")

    await store.revert_event(eid)
    assert (await store.get_claim(cid))["status"] == "live"
    assert not await store.suppressed("preference", "The user prefers aisle seats")


async def test_correct_makes_a_stated_version_in_the_same_lineage(store):
    cid = await _claim(store, "The user uses Arc as their main browser")
    nid, _ = await store.correct_claim(cid, "The user switched to Helium as their main browser")
    old, new = await store.get_claim(cid), await store.get_claim(nid)
    assert old["status"] == "superseded" and old["superseded_by"] == nid
    assert new["lineage_id"] == old["lineage_id"] and new["version"] == old["version"] + 1
    assert new["modality"] == "stated" and new["status"] == "live"
    assert [c["id"] for c in await store.lineage(nid)] == [cid, nid]


async def test_draft_decisions_are_logged_and_reinforcement_is_undoable(store):
    cid = await _claim(store, "The user prefers dark mode", n=1)
    oid = await _obs(store, "dark mode toggle", n=5)
    ep = await store.q1("SELECT id FROM episodes LIMIT 1")
    if ep is None:
        await store.execute("INSERT INTO episodes(session_id, started_at, ended_at, status, created_at, updated_at) "
                            "VALUES ('s', 1000, 1010, 'drafting', 0, 0)")
        ep = await store.q1("SELECT id FROM episodes LIMIT 1")
    else:
        await store.execute("UPDATE episodes SET status='drafting' WHERE id=?", (ep[0],))
    before = (await store.get_claim(cid))["confidence"]
    draft = ClaimDraft(kind="preference", text="The user prefers dark mode", modality="engaged",
                       evidence=[{"observation_id": oid, "quote": "dark mode toggle"}])
    await store.commit_episode_draft(ep[0], title="t", summary="s", apps=[], sites=[], engaged_fraction=1.0,
                                     writes=[ClaimWrite(draft, operation="reinforced", target_id=cid)],
                                     candidates=[], rejected={})
    [reinforce] = await store.recent_events(op="reinforce")
    assert json.loads(reinforce["changes"]) == [{"id": cid, "evidence_added": [oid]}]
    assert (await store.get_claim(cid))["confidence"] > before

    await store.revert_event(reinforce["id"])
    assert oid not in {r["observation_id"] for r in await store.evidence_for_claim(cid)}
    assert (await store.get_claim(cid))["confidence"] == pytest.approx(before)
