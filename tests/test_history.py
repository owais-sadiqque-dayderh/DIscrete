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
    [change] = json.loads(reinforce["changes"])
    assert [r["observation_id"] for r in change["evidence_added_rows"]] == [oid]
    assert (await store.get_claim(cid))["confidence"] > before

    await store.revert_event(reinforce["id"])
    assert oid not in {r["observation_id"] for r in await store.evidence_for_claim(cid)}
    assert (await store.get_claim(cid))["confidence"] == pytest.approx(before)


async def test_reverting_a_reinforce_preserves_full_support_membership(store, monkeypatch):
    """Review #3: the cap's evicted rows were lost for good on revert."""
    from discrete.config import T

    monkeypatch.setattr(T, "EVIDENCE_MAX_SUPPORTS", 1)
    cid = await _claim(store, "The user prefers dark mode", n=1)
    old = [r["observation_id"] for r in await store.evidence_for_claim(cid)]
    new = await _obs(store, "dark mode again", n=9)
    draft = ClaimDraft(kind="preference", text="The user prefers dark mode", modality="engaged",
                       evidence=[{"observation_id": new, "quote": "dark mode again"}])

    def _go():
        with store._tx() as conn:
            change = store._reinforce_claim(conn, cid, draft)
            return store._log_event(conn, "reinforce", "draft", claim_id=cid, changes=[change])

    eid = await store._call(_go)
    assert [r["observation_id"] for r in await store.evidence_for_claim(cid)] == [new]  # quote sample
    assert {r[0] for r in await store.q("SELECT observation_id FROM evidence WHERE claim_id=?", (cid,))} == {*old, new}
    await store.revert_event(eid)
    assert [r["observation_id"] for r in await store.evidence_for_claim(cid)] == old


async def test_an_old_profile_cannot_overwrite_a_newer_one(store):
    """Review #6."""
    a = await store.save_profile({"written_at": 1.0, "sections": []}, run_id="r1")
    await store.save_profile({"written_at": 2.0, "sections": []}, run_id="r2")
    with pytest.raises(RevertConflict):
        await store.revert_event(a)
    assert (await store.get_profile())["written_at"] == 2.0


async def test_redo_of_a_forget_restores_its_suppression_and_the_forget_can_be_reverted_again(store):
    """Review #7."""
    cid = await _claim(store, "The user prefers aisle seats")
    forget = await store.forget_claim(cid)
    undo = await store.revert_event(forget)
    redo = await store.revert_event(undo)
    assert (await store.get_event(redo))["op"] == "redo"
    assert (await store.get_claim(cid))["status"] == "forgotten"
    assert await store.suppressed("preference", "The user prefers aisle seats")
    assert (await store.get_event(forget))["undone_by"] is None
    with pytest.raises(RevertConflict):
        await store.revert_event(redo)  # revert the original instead
    await store.revert_event(forget)
    assert (await store.get_claim(cid))["status"] == "live"
    assert not await store.suppressed("preference", "The user prefers aisle seats")


async def test_correcting_an_old_version_corrects_the_live_one_and_keeps_the_action(store):
    """Review #5."""
    oid = await _obs(store, "Thank you for downloading Helium", n=3)
    cid = await store.create_claim(ClaimDraft(kind="event", text="The user downloaded Helium", modality="engaged",
                                              action="downloaded",
                                              evidence=[{"observation_id": oid, "quote": "downloading Helium"}]))
    v2, _ = await store.correct_claim(cid, "The user downloaded the Helium browser")
    v3, _ = await store.correct_claim(cid, "The user downloaded Helium 0.18 for macOS")  # names the old id
    live = [c for c in await store.lineage(cid) if c["status"] == "live"]
    assert [c["id"] for c in live] == [v3] and live[0]["action"] == "downloaded"
    assert v2 != v3


async def test_a_forgotten_memory_leaves_the_profile_at_once(store):
    """Review #4."""
    a = await _claim(store, "The user follows football on FotMob", n=1)
    b = await _claim(store, "The user uses Arc", n=2)
    await store.save_profile({"written_at": 1.0, "sections": [
        {"heading": "Interests", "points": [{"text": "They follow football.", "claims": [a]}]},
        {"heading": "Tools", "points": [{"text": "They use Arc.", "claims": [b]}]}]}, run_id="r")
    await store.forget_claim(a)
    profile = await store.get_profile()
    assert [s["heading"] for s in profile["sections"]] == ["Tools"]


async def test_older_evidence_cannot_invalidate_newer_claim(store):
    old = await store.create_claim(ClaimDraft(kind="fact", text="The user used the old editor", occurred_from=10, occurred_to=20))
    new = await store.create_claim(ClaimDraft(kind="fact", text="The user uses the new editor", occurred_from=30, occurred_to=40))

    def invalidate():
        with store._tx() as conn:
            return store._invalidate_claim(conn, new, by_id=old, at=10)

    assert await store._call(invalidate) is None
    assert (await store.get_claim(new))["status"] == "live"
    assert (await store.get_claim(new))["invalidates_id"] is None


async def test_checked_verdict_expires_when_claim_revision_changes(store):
    a = await _claim(store, "The user prefers tea", n=1)
    b = await _claim(store, "The user prefers coffee", n=2)
    await store.mark_pair_checked(a, b, "compatible", "different settings")
    assert (a, b) in await store.checked_pairs()
    await store.execute("UPDATE claims SET occurred_to=occurred_to+1, updated_at=updated_at+1 WHERE id=?", (a,))
    assert (a, b) not in await store.checked_pairs()
    await store.mark_pair_checked(a, b, "nonsense", "invalid model output")
    assert (a, b) not in await store.checked_pairs()


async def test_merge_preserves_stated_authority_only_for_its_proposition(store):
    a = await _claim(store, "The user likes the editor", n=1)
    b = await store.create_claim(ClaimDraft(kind="preference", text="The user prefers the editor", modality="stated"))
    [same] = await store.consolidate_cluster(await store.get_claims([a, b]), [{
        "text": "The user prefers the editor", "input_ids": [a, b]}])
    assert (await store.get_claim(same))["modality"] == "stated"
    [general] = await store.consolidate_cluster(await store.get_claims([same]), [{
        "text": "The user likes software tools", "input_ids": [same]}])
    assert (await store.get_claim(general))["modality"] != "stated"
    assert (await store.get_claim(same))["status"] == "live"


async def test_profile_redo_refuses_after_newer_profile(store):
    first = await store.save_profile({"written_at": 1, "sections": []}, run_id="r")
    undo = await store.revert_event(first)
    await store.save_profile({"written_at": 2, "sections": []}, run_id="r")
    with pytest.raises(RevertConflict):
        await store.revert_event(undo)
    assert (await store.get_profile())["written_at"] == 2


async def test_evidence_update_restores_predecessor_quote_on_undo_and_redo(store):
    cid = await _claim(store, "The user likes tea", n=1)
    [before] = await store.evidence_for_claim(cid)

    def reinforce():
        with store._tx() as conn:
            change = store._reinforce_claim(conn, cid, ClaimDraft(kind="preference", text="The user likes tea",
                evidence=[{"observation_id": before["observation_id"], "quote": "updated quote", "field": "caption"}]))
            return store._log_event(conn, "reinforce", "draft", claim_id=cid, changes=[change])

    event = await store._call(reinforce)
    undo = await store.revert_event(event)
    assert dict((await store.evidence_for_claim(cid))[0]) == dict(before)
    await store.revert_event(undo)
    assert (await store.evidence_for_claim(cid))[0]["quote"] == "updated quote"


async def test_overwritten_trail_restores_predecessor_row_on_undo_and_redo(store):
    cid = await _claim(store, "The user downloaded a tool")
    oid = await _obs(store, "tool page", 8)
    link = {"observation_id": oid, "relation": "led_to", "confidence": 0.8, "explanation": "first", "quote": "old"}
    await store.record_trail(cid, [link], reason="first")
    [before] = await store.links_into("claim", [cid])
    second = await store.record_trail(cid, [{**link, "explanation": "second", "quote": "new"}], reason="second")
    [after] = await store.links_into("claim", [cid])
    undo = await store.revert_event(second)
    assert [dict(r) for r in await store.links_into("claim", [cid])] == [dict(before)]
    await store.revert_event(undo)
    assert [dict(r) for r in await store.links_into("claim", [cid])] == [dict(after)]


async def test_forget_clears_entity_page_quoting_it_and_marks_it_for_rebuild(store):
    a = await _claim(store, "The user edits notes in Zed", n=1)
    b = await _claim(store, "The user reads docs in Zed", n=2)
    eid = await store.create_entity("Zed", "app", "zed")
    await store.record_mentions([(eid, "claim", a), (eid, "claim", b)])
    await store.update_entity_page(eid, "Zed: used for notes and docs.")
    assert not await store.entities_needing_pages(10)
    await store.forget_claim(a)
    page = await store.get_entity(eid)
    assert page["summary"] == ""
    assert [r["id"] for r in await store.entities_needing_pages(10)] == [eid]
    assert not await store.q("SELECT 1 FROM entities_fts WHERE entities_fts MATCH 'notes'")


async def test_erase_removes_content_everywhere_and_blocks_relearning(store):
    cid = await _claim(store, "The user keeps zebrafish secretly", n=3)
    other = await _claim(store, "The user likes tea", n=4)
    await store.save_profile({"written_at": 1, "sections": [{"heading": "h", "points": [
        {"text": "They keep zebrafish.", "claims": [cid]}, {"text": "They like tea.", "claims": [other]}]}]}, run_id="r")
    await store.forget_claim(cid)
    assert await store.erase_claim(cid) == 1
    blob = ""
    for table in ("claims", "evidence", "memory_events", "suppressions", "meta", "links"):
        blob += "".join(str(tuple(r)) for r in await store.q(f"SELECT * FROM {table}"))
    assert "zebrafish" not in blob and "tea" in blob
    assert not await store.q("SELECT 1 FROM claims_fts WHERE claims_fts MATCH 'zebrafish'")
    assert await store.suppressed("preference", "The user keeps zebrafish secretly")
    assert [r[0] for r in await store.q("SELECT vector FROM suppressions WHERE claim_id=?", (cid,))] == [None]
    assert not await store.q("SELECT 1 FROM embeddings WHERE ref_kind='claim' AND ref_id=?", (cid,))
    event = (await store.recent_events(op="forget"))[0]
    with pytest.raises(RevertConflict):
        await store.revert_event(event["id"])
    assert [p["text"] for s in (await store.get_profile())["sections"] for p in s["points"]] == ["They like tea."]
