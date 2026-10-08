"""User feedback for the jarvis bridge uses the ordinary reversible memory log."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from discrete import textkit as tk
from discrete.models import ClaimDraft, ObservationDraft, QueryPlan, RetrievalResult
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval import hybrid
from discrete.store.index import embedding_text
from tests.conftest import bow_vector

REASON = "via Comma personalization (request test-123)"


async def _claim(store, text="The user likes tea", *, context="", kind="preference"):
    return await store.create_claim(ClaimDraft(kind=kind, text=text, context=context))


async def _event(store, eid, op):
    event = await store.get_event(eid)
    assert (event["actor"], event["op"], event["reason"]) == ("user", op, REASON)
    return event


async def test_state_new_lineage_indexes_testimony_and_reverts(store, fake):
    cid, eid = await store.state_claim("preference", "  The user likes tea  ", context="at work", reason=REASON)
    row = await store.get_claim(cid)
    assert (row["text"], row["context"], row["modality"], row["status"], row["version"]) == (
        "The user likes tea", "at work", "stated", "live", 1)
    assert row["confidence"] > 0
    assert json.loads(row["derived_from"]) == []
    assert cid in await store.fts_ids("claim", "tea")
    assert await store.q1("SELECT 1 FROM index_outbox WHERE ref_kind='claim' AND ref_id=?", (cid,))
    assert not fake.embedded and not fake.calls
    event = await _event(store, eid, "state")
    assert json.loads(event["created"]) == [cid]
    undo = await store.revert_event(eid)
    assert (await store.get_claim(cid))["status"] == "reverted"
    await store.revert_event(undo)
    assert (await store.get_claim(cid))["status"] == "live"


async def test_state_copies_support_fields_and_mentions_from_all_sources(store):
    oid, _ = await store.add_observation(ObservationDraft(app="Arc", window_title="Tea",
        screen_text="Tea shop", started_at=1000, ended_at=1001, keys=1))
    a = await store.create_claim(ClaimDraft(kind="event", text="Bought tea", evidence=[
        {"observation_id": oid, "quote": "Tea shop", "field": "screen_text"}]))
    b = await _claim(store, "Tea is warm")
    # Even empty supporting quotes retain their supports relation (not cooccurs).
    await store.execute("INSERT INTO evidence(claim_id, observation_id, relation, quote, field, created_at) "
                        "VALUES (?,?,'supports','','caption',0)", (b, oid))
    entity = await store.create_entity("Tea", "thing", "tea")
    await store.record_mentions([(entity, "claim", a)])
    cid, eid = await store.state_claim("preference", "I like tea", derived_from=[b, a, a], reason=REASON)
    row = await store.get_claim(cid)
    assert json.loads(row["derived_from"]) == [a, b]
    assert row["lineage_id"] not in {r["lineage_id"] for r in await store.get_claims([a, b])}
    [ev] = await store.evidence_for_claim(cid)
    assert (ev["observation_id"], ev["quote"], ev["field"], ev["relation"]) == (
        oid, "Tea shop", "screen_text", "supports")
    assert await store.q1("SELECT 1 FROM mentions WHERE ref_kind='claim' AND ref_id=? AND entity_id=?", (cid, entity))
    assert json.loads((await _event(store, eid, "state"))["data"])["derived_from"] == [a, b]
    await store.revert_event(eid)
    assert all(r["status"] == "live" for r in await store.get_claims([a, b]))


async def test_state_preserves_empty_support_quote(store):
    oid, _ = await store.add_observation(ObservationDraft(screen_text="Tea", started_at=1, ended_at=2))
    cid = await _claim(store)
    await store.execute("INSERT INTO evidence(claim_id, observation_id, relation, quote, field, created_at) "
                        "VALUES (?,?,'supports','','caption',0)", (cid, oid))
    new, _ = await store.state_claim("preference", "I like tea", derived_from=[cid])
    [ev] = await store.evidence_for_claim(new)
    assert (ev["relation"], ev["quote"], ev["field"]) == ("supports", "", "caption")


@pytest.mark.parametrize("op", ["state", "correct"])
async def test_user_wording_wins_over_suppression(store, op):
    await store.reject_inference("I like tea")
    original = await _claim(store, "The user likes drinks")
    if op == "state":
        cid, _ = await store.state_claim("preference", "I like tea")
    else:
        cid, _ = await store.correct_claim(original, "I like tea")
    assert (await store.get_claim(cid))["status"] == "live"
    assert await store.suppressed("preference", "I like tea")


async def test_confirm_is_one_event_with_all_refs_and_no_belief_mutation(store):
    a, b = await _claim(store), await _claim(store, "The user likes coffee")
    before = [dict(r) for r in await store.get_claims([a, b])]
    eid = await store.confirm_claims([a, b, a], context="buying drinks", reason=REASON)
    event = await _event(store, eid, "confirm")
    assert json.loads(event["data"]) == {"claim_ids": [a, b], "context": "buying drinks"}
    assert json.loads(event["changes"]) == [] and json.loads(event["created"]) == []
    for cid in [a, b]:
        assert eid in [e["id"] for e in await store.events_for_claim(cid)]
    undo = await store.revert_event(eid)
    assert (await store.get_event(eid))["undone_by"] == undo
    assert [dict(r) for r in await store.get_claims([a, b])] == before
    await store.revert_event(undo)
    assert (await store.get_event(eid))["undone_by"] is None


@pytest.mark.parametrize("context", [None, "only at work", ""])
async def test_correct_context_and_revert(store, context):
    cid = await _claim(store, context="old scope")
    nid, eid = await store.correct_claim(cid, "The user prefers coffee", context=context, reason=REASON)
    assert (await store.get_claim(nid))["context"] == ("old scope" if context is None else context)
    assert (await store.get_claim(cid))["status"] == "superseded"
    await _event(store, eid, "correct")
    undo = await store.revert_event(eid)
    assert (await store.get_claim(cid))["context"] == "old scope"
    assert (await store.get_claim(cid))["status"] == "live"
    assert (await store.get_claim(nid))["status"] == "reverted"
    await store.revert_event(undo)
    assert (await store.get_claim(nid))["status"] == "live"


async def test_retire_all_live_versions_of_named_lineage_and_revert(store):
    cid = await _claim(store)
    nid, _ = await store.correct_claim(cid, "The user prefers coffee")
    # Exercise a lineage containing several live versions, including an older named version.
    await store.execute("UPDATE claims SET status='live' WHERE id=?", (cid,))
    eid = await store.retire_claim(cid, reason=REASON)
    assert all(r["status"] == "retired" and r["valid_to"] is not None for r in await store.lineage(cid))
    assert not await store.suppressed("preference", "The user likes tea")
    assert not await store.suppressed("preference", "The user prefers coffee")
    event = await _event(store, eid, "retire")
    assert {c["id"] for c in json.loads(event["changes"])} == {cid, nid}
    undo = await store.revert_event(eid)
    assert all(r["status"] == "live" for r in await store.lineage(cid))
    await store.revert_event(undo)
    assert all(r["status"] == "retired" for r in await store.lineage(cid))
    assert await _claim(store)  # Re-learning the wording remains possible.


@pytest.mark.parametrize("op", ["retire", "forget"])
async def test_withdrawal_excluded_from_retrieval_and_profile_then_restored(store, op):
    cid = await _claim(store, "The user prefers zebrafish")
    row = await store.get_claim(cid)
    await store.save_embeddings([("claim", cid, bow_vector(row["text"]), tk.content_hash(embedding_text("claim", row)))])
    await store.save_profile({"sections": [{"heading": "Preferences", "points": [
        {"text": row["text"], "claims": [cid]}]}]}, run_id="test")
    assert cid in [h.ref_id for h in (await hybrid.retrieve(store, "zebrafish", verify=False)).hits
                   if h.ref_kind == "claim"]
    eid = await getattr(store, f"{op}_claim")(cid, reason=REASON)
    await _event(store, eid, op)
    assert not (await store.get_profile())["sections"]
    assert not await store.q1("SELECT 1 FROM embeddings WHERE ref_kind='claim' AND ref_id=?", (cid,))
    assert cid not in [h.ref_id for h in (await hybrid.retrieve(store, "zebrafish", verify=False)).hits
                      if h.ref_kind == "claim"]
    assert bool(await store.suppressed("preference", row["text"])) == (op == "forget")
    await store.revert_event(eid)
    assert (await store.get_profile())["sections"]
    assert cid in [h.ref_id for h in (await hybrid.retrieve(store, "zebrafish", verify=False)).hits
                   if h.ref_kind == "claim"]


async def test_reject_suppresses_drafting_without_changing_premises_and_redoes(store, fake):
    cid = await _claim(store, "The user bought tea", kind="event")
    before = dict(await store.get_claim(cid))
    eid = await store.reject_inference("The user prefers tea", claim_ids=[cid, cid], reason=REASON)
    event = await _event(store, eid, "reject")
    assert json.loads(event["data"])["claim_ids"] == [cid]
    assert eid in [e["id"] for e in await store.events_for_claim(cid)]
    assert dict(await store.get_claim(cid)) == before
    assert await store.suppressed("preference", "the user prefers tea.")
    assert not await store.suppressed("event", "The user prefers tea")
    with pytest.raises(ValueError, match="suppressed"):
        await _claim(store, "The user prefers tea")  # Same transaction-time gate as drafting/dreaming.
    assert not fake.embedded and not fake.calls
    undo = await store.revert_event(eid)
    assert not await store.suppressed("preference", "The user prefers tea")
    await store.revert_event(undo)
    assert await store.suppressed("preference", "The user prefers tea")
    assert dict(await store.get_claim(cid)) == before


async def test_reject_reuses_exact_wording_vector_and_revert_restores_it(store):
    cid = await _claim(store)
    row = await store.get_claim(cid)
    await store.save_embeddings([("claim", cid, bow_vector(row["text"]), tk.content_hash(embedding_text("claim", row)))])
    eid = await store.reject_inference(row["text"])
    sup = await store.q1("SELECT * FROM suppressions WHERE event_id=?", (eid,))
    assert sup["vector"] is not None
    undo = await store.revert_event(eid)
    await store.revert_event(undo)
    assert dict(await store.q1("SELECT * FROM suppressions WHERE event_id=?", (eid,))) == dict(sup)


@pytest.mark.parametrize("op", ["reject", "retire"])
async def test_feedback_gates_real_drafting_relearning(store, fake, op):
    wording = "The user prefers dark mode in editors"
    cid = await _claim(store, wording)
    if op == "reject":
        await store.reject_inference(wording, claim_ids=[cid])
    else:
        await store.retire_claim(cid)
    draft = ObservationDraft(app="Code", window_title="Settings", screen_text=wording,
        started_at=1_789_000_000., ended_at=1_789_000_005., keys=1, session_id="feedback")
    oid, _ = await store.add_observation(draft)
    tracker = EpisodeTracker(store)
    await tracker.on_observation(oid, await store.get_observation(oid), session_id=draft.session_id)
    (ep,) = await tracker.close_current()
    fake.draft = {"title": "Settings", "summary": wording, "claims": [{
        "kind": "preference", "text": wording, "evidence": [{"observation_id": oid, "quote": wording}]}]}
    outcome = await ProcessPipeline(store).draft_episode(ep)
    assert outcome is not None
    if op == "reject":
        assert outcome.rejected == {"forgotten": 1} and outcome.added == 0
        assert (await store.get_claim(cid))["status"] == "live"
    else:
        assert outcome.added == 1 and not outcome.rejected
        assert (await store.get_claim(cid))["status"] == "retired"
        assert [r["text"] for r in await store.live_claims()] == [wording]


async def test_feedback_without_source_ids_is_reversible(store):
    confirm = await store.confirm_claims([], reason=REASON)
    reject = await store.reject_inference("I dislike tea", reason=REASON)
    await _event(store, confirm, "confirm")
    await _event(store, reject, "reject")
    await store.revert_event(confirm)
    await store.revert_event(reject)
    assert not await store.suppressed("preference", "I dislike tea")


@pytest.mark.parametrize("op", ["state", "confirm", "correct", "retire", "reject"])
async def test_unknown_ids_roll_back_feedback(store, op):
    before = await store.scalar("SELECT COUNT(*) FROM memory_events")
    with pytest.raises(KeyError):
        if op == "state":
            await store.state_claim("preference", "I like tea", derived_from=[999])
        elif op == "confirm":
            await store.confirm_claims([999])
        elif op == "correct":
            await store.correct_claim(999, "I like tea")
        elif op == "retire":
            await store.retire_claim(999)
        else:
            await store.reject_inference("I like tea", claim_ids=[999])
    assert await store.scalar("SELECT COUNT(*) FROM memory_events") == before
    assert not await store.live_claims()
    assert not await store.q("SELECT * FROM suppressions")


@pytest.mark.parametrize("op", ["confirm", "correct", "retire"])
async def test_nonlive_ids_fail_when_no_live_target(store, op):
    cid = await _claim(store)
    await store.retire_claim(cid)
    with pytest.raises(KeyError):
        if op == "confirm":
            await store.confirm_claims([cid])
        elif op == "correct":
            await store.correct_claim(cid, "I like coffee")
        else:
            await store.retire_claim(cid)


async def test_confirmation_requires_every_id_live_and_rolls_back(store):
    a, b = await _claim(store), await _claim(store, "Coffee")
    await store.forget_claim(b)
    with pytest.raises(KeyError):
        await store.confirm_claims([a, b])
    assert not await store.recent_events(op="confirm")


async def test_evidence_summary_ranks_support_and_returns_domain_scope_and_limits(store):
    cid = await _claim(store, context="at work")
    ids = []
    for n, keys in [(1, 1), (2, 0), (3, 1)]:
        oid, _ = await store.add_observation(ObservationDraft(app="Arc", window_title=f"Tea {n}",
            url=f"https://shop.example.com/tea/{n}", site="example.com", screen_text=f"Tea {n}",
            started_at=1000. + n, ended_at=1001. + n, keys=keys))
        ids.append(oid)
        await store.execute("INSERT INTO evidence(claim_id, observation_id, relation, quote, field, created_at) "
                            "VALUES (?,?,'supports',?,'screen_text',?)", (cid, oid, f"Tea {n}", n))
    summary = await store.evidence_summary([cid, 999])
    assert summary == {cid: [{"observation_id": ids[2], "app": "Arc", "window_title": "Tea 3",
        "site": "example.com", "started_at": 1003., "quote": "Tea 3", "context": "at work"}], 999: []}
    assert [r["observation_id"] for r in (await store.evidence_summary([cid], per_claim=2))[cid]] == [ids[2], ids[0]]
    assert len((await store.evidence_summary([cid], per_claim=10))[cid]) == 3
    assert await store.evidence_summary([]) == {}
    assert await store.evidence_summary([cid], per_claim=0) == {cid: []}
    await store.execute("UPDATE evidence SET relation='cooccurs' WHERE claim_id=?", (cid,))
    assert await store.evidence_summary([cid]) == {cid: []}


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("intent", ["recall", "enumerate"])
async def test_retrieve_verify_switch_including_enumeration_fallback(store, monkeypatch, verify, intent):
    result = RetrievalResult(query="tea", plan=QueryPlan(raw="tea", intent=intent))
    retrieve = AsyncMock(return_value=result)
    verifier = AsyncMock()
    monkeypatch.setattr(hybrid, "_retrieve", retrieve)
    monkeypatch.setattr(hybrid, "verify_support", verifier)
    kwargs = {} if verify else {"verify": False}  # Exercise the unchanged default, too.
    await hybrid.retrieve(store, "tea", **kwargs)
    assert verifier.await_count == ((2 if intent == "enumerate" else 1) if verify else 0)
    assert retrieve.await_count == (2 if intent == "enumerate" else 1)
