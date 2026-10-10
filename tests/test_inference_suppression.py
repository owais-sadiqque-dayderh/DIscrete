"""Rejected ideas cross kinds; outdated ideas require genuinely newer evidence."""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock

import numpy as np
import pytest

from discrete import llm
from discrete.models import ClaimDraft, ObservationDraft
from discrete.store import db


@pytest.mark.parametrize("kind", ["preference", "pattern", "fact"])
async def test_inference_cross_kind_insertion_revert_redo(store, kind):
    eid = await store.reject_inference("You prefer tactile keyboards with brown switches")
    wording = "For keyboards, you prefer brown switches that are tactile"
    assert await store.suppressed(kind, wording)
    with pytest.raises(ValueError, match="suppressed"):
        await store.create_claim(ClaimDraft(kind=kind, text=wording))
    event = await store.create_claim(ClaimDraft(kind="event", text=wording))
    assert (await store.get_claim(event))["status"] == "live"
    undo = await store.revert_event(eid)
    assert not await store.suppressed(kind, wording)
    await store.revert_event(undo)
    assert await store.suppressed(kind, wording)


@pytest.mark.parametrize(("rejected", "candidate", "matches"), [
    ("You prefer tactile keyboards with brown switches", "You prefer brown-switch tactile keyboards", True),
    ("You prefer dark mode in code editors", "In code editors you prefer the dark mode", True),
    ("You buy on Flipkart with the Axis card", "You prefer Flipkart for electronics", False),
    ("You buy on Flipkart with the Axis card", "You buy on Flipkart with the Visa card", False),
    ("You prefer tactile keyboards with brown switches", "You prefer silent keyboards with lavender switches", False),
    ("You prefer dark mode in editors", "You do not prefer dark mode in editors", False),
])
async def test_offline_paraphrases_and_near_misses(store, monkeypatch, rejected, candidate, matches):
    # Shared Flipkart is a different assertion: purchase/payment vs category preference.
    monkeypatch.setattr(llm, "try_embed", AsyncMock(return_value=None))
    await store.reject_inference(rejected)
    assert await store.suppressed("pattern", candidate) == matches


async def test_embedding_generated_and_semantic_threshold(store, monkeypatch):
    embed = AsyncMock(return_value=[1., 0.])
    monkeypatch.setattr(llm, "try_embed", embed)
    eid = await store.reject_inference("You enjoy reading science fiction")
    assert embed.await_args.args == ("You enjoy reading science fiction",)
    row = await store.q1("SELECT * FROM suppressions WHERE event_id=?", (eid,))
    assert np.frombuffer(row["vector"], dtype=np.float32).tolist() == [1., 0.]
    assert await store.suppressed("fact", "Space novels are your favorite books", vector=[0.95, 0.312])
    assert not await store.suppressed("fact", "You write essays about space", vector=[0.89, 0.456])
    assert await store.suppressed("fact", "Space novels are your favorite books")
    undo = await store.revert_event(eid)
    await store.revert_event(undo)
    assert dict(await store.q1("SELECT * FROM suppressions WHERE event_id=?", (eid,))) == dict(row)


async def test_no_endpoint_keeps_lexical_suppression(store, monkeypatch):
    monkeypatch.delenv("EMBED_LM_MODEL_NAME")
    eid = await store.reject_inference("You prefer dark mode in editors")
    assert (await store.q1("SELECT vector FROM suppressions WHERE event_id=?", (eid,)))[0] is None
    assert await store.suppressed("fact", "In editors you prefer dark mode")


async def test_profile_read_save_and_revert(store):
    source = await store.create_claim(ClaimDraft(kind="event", text="Bought a keyboard"))
    point = {"text": "You prefer tactile keyboards with brown switches", "claims": [source]}
    profile = {"sections": [{"heading": "Preferences", "points": [point]}]}
    await store.save_profile(profile, run_id="before")
    eid = await store.reject_inference(point["text"], claim_ids=[source])
    assert not (await store.get_profile())["sections"]
    assert (await store.get_profile(live_only=False))["sections"]
    undo = await store.revert_event(eid)
    assert (await store.get_profile())["sections"]
    await store.revert_event(undo)
    point["text"] = "For keyboards you prefer brown switches that are tactile"
    saved = await store.save_profile(profile, run_id="after")
    assert not (await store.get_profile(live_only=False))["sections"]
    await store.revert_event(saved)
    assert (await store.get_profile(live_only=False))["sections"]
    assert not (await store.get_profile())["sections"]
    with pytest.raises(ValueError, match="suppressed"):
        await store.add_pattern(point["text"], "preference", [source], reason="test", run_id="test")


async def _source(store, at):
    oid, _ = await store.add_observation(ObservationDraft(screen_text=f"Dark mode {at}", started_at=at, ended_at=at+1))
    cid = await store.create_claim(ClaimDraft(kind="event", text=f"Used dark mode {at}", evidence=[
        {"observation_id": oid, "quote": f"Dark mode {at}"}]))
    return oid, cid


async def test_retirement_revives_only_new_support_and_is_undoable(store, monkeypatch):
    monkeypatch.setattr("discrete.store.corrections.now", lambda: 100.)
    old_oid, old = await _source(store, 99.)
    text = "You prefer dark mode in editors"
    eid = await store.retire_inference(text, claim_ids=[old])
    before = dict(await store.get_claim(old))
    assert await store.suppressed("pattern", text, claim_ids=[old])
    assert await store.suppressed("fact", text)
    with pytest.raises(ValueError, match="suppressed"):
        await store.create_claim(ClaimDraft(kind="preference", text=text, occurred_to=999., evidence=[
            {"observation_id": old_oid, "quote": "Dark mode 99.0"}]))
    equal_oid, equal = await _source(store, 100.)
    assert await store.suppressed("pattern", text, claim_ids=[equal])
    new_oid, fresh = await _source(store, 101.)
    assert not await store.suppressed("pattern", text, claim_ids=[fresh])
    cid = await store.create_claim(ClaimDraft(kind="preference", text=text, evidence=[
        {"observation_id": new_oid, "quote": "Dark mode 101.0"}]))
    assert cid
    await store.save_profile({"sections": [{"heading": "Preferences", "points": [
        {"text": text, "claims": [cid]}]}]}, run_id="fresh")
    assert (await store.get_profile())["sections"]
    undo = await store.revert_event(eid)
    assert not await store.suppressed("fact", text, claim_ids=[old])
    await store.revert_event(undo)
    assert await store.suppressed("fact", text, claim_ids=[old])
    assert not await store.suppressed("fact", text, claim_ids=[fresh])
    assert dict(await store.get_claim(old)) == before
    # A permanent rejection wins even with newer evidence.
    await store.reject_inference(text)
    assert await store.suppressed("pattern", text, claim_ids=[fresh])


async def test_retirement_without_sources_and_invalid_input(store):
    eid = await store.retire_inference("You prefer tea")
    assert await store.suppressed("fact", "You prefer tea")
    undo = await store.revert_event(eid)
    assert not await store.suppressed("fact", "You prefer tea")
    await store.revert_event(undo)
    assert await store.suppressed("fact", "You prefer tea")
    for op in (store.reject_inference, store.retire_inference):
        with pytest.raises(ValueError):
            await op("Bought tea", kind="event")
        with pytest.raises(ValueError):
            await op(" ")
        with pytest.raises(KeyError):
            await op("You like tea", claim_ids=[999])


def test_m016_upgrades_rejections_and_legacy_undo_payloads(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    for migration in db.MIGRATIONS[:15]:
        migration(conn)
    conn.execute("PRAGMA user_version=15")
    conn.execute("INSERT INTO memory_events(id,at,op,actor) VALUES (1,0,'reject','user')")
    conn.execute("INSERT INTO memory_events(id,at,op,actor) VALUES (2,0,'forget','user')")
    for eid in (1, 2):
        conn.execute("INSERT INTO suppressions(kind,text,norm_key,event_id,created_at) VALUES "
                     "('preference','tea','tea',?,0)", (eid,))
    payload = {"suppressions": [{"event_id": 1}, {"event_id": 2}]}
    conn.execute("INSERT INTO memory_events(at,op,actor,data) VALUES (0,'revert','user',?)", (json.dumps(payload),))
    conn.commit()
    assert db.migrate(conn) == 1
    assert conn.execute("SELECT scope, retired_after FROM suppressions ORDER BY id").fetchall() == [
        ("inference", None), ("kind", None)]
    rows = json.loads(conn.execute("SELECT data FROM memory_events WHERE op='revert'").fetchone()[0])["suppressions"]
    assert rows == [{"event_id": 1, "scope": "inference", "retired_after": None},
                    {"event_id": 2, "scope": "kind", "retired_after": None}]
    assert db.migrate(conn) == 0
    conn.close()


async def test_actual_dream_cannot_relearn_and_revert_restores_learning(store, fake):
    from discrete.process.dream import run_dream
    from tests.test_dream import _football, _pattern

    sources = await _football(store)
    wording = "The user follows football scores on FotMob"
    eid = await store.reject_inference(wording, claim_ids=sources)
    fake.dream = {"patterns": [_pattern(sources, "The user follows scores for football on FotMob")], "audit": []}
    assert (await run_dream(store))["patterns"]["added"] == 0
    assert not await store.live_claims(kind="pattern")
    undo = await store.revert_event(eid)
    assert (await run_dream(store))["patterns"]["added"] == 1
    await store.revert_event(undo)
    assert not await store.confirm_pattern((await store.live_claims(kind="pattern"))[0]["id"], sources,
                                           reason="re-suggested", run_id="test")


@pytest.mark.parametrize("fresh", [False, True])
async def test_retirement_real_episode_pipeline(store, fake, monkeypatch, fresh):
    from discrete.process.episodes import EpisodeTracker
    from discrete.process.pipeline import ProcessPipeline

    monkeypatch.setattr("discrete.store.corrections.now", lambda: 1_789_000_000.)
    wording = "The user prefers dark mode in editors"
    await store.retire_inference(wording)
    at = 1_789_000_001. if fresh else 1_788_999_999.
    draft = ObservationDraft(app="Code", window_title="Settings", screen_text=wording,
                             started_at=at, ended_at=at+5., keys=1, session_id="retirement")
    oid, _ = await store.add_observation(draft)
    tracker = EpisodeTracker(store)
    await tracker.on_observation(oid, await store.get_observation(oid), session_id=draft.session_id)
    (ep,) = await tracker.close_current()
    fake.draft = {"title": "Settings", "summary": wording, "claims": [{
        "kind": "preference", "text": wording, "evidence": [{"observation_id": oid, "quote": wording}]}]}
    result = await ProcessPipeline(store).draft_episode(ep)
    assert result is not None
    assert result.added == int(fresh)
    assert result.rejected == ({} if fresh else {"forgotten": 1})


async def test_retirement_cooccurrence_does_not_revive(store, monkeypatch):
    monkeypatch.setattr("discrete.store.corrections.now", lambda: 100.)
    await store.retire_inference("You prefer tea")
    oid, _ = await _source(store, 101.)
    with pytest.raises(ValueError, match="suppressed"):
        await store.create_claim(ClaimDraft(kind="preference", text="You prefer tea", evidence=[
            {"observation_id": oid, "quote": ""}]))


async def test_semantic_profile_and_pattern_filter(store, monkeypatch):
    monkeypatch.setattr(llm, "try_embed", AsyncMock(return_value=[1., 0.]))
    source = await store.create_claim(ClaimDraft(kind="event", text="Finished a novel"))
    point = {"text": "Space novels are your favorite books", "claims": [source]}
    await store.save_profile({"sections": [{"heading": "Interests", "points": [point]}]}, run_id="old")
    await store.reject_inference("You enjoy reading science fiction")
    assert not (await store.get_profile())["sections"]
    assert (await store.get_profile(live_only=False))["sections"]
    with pytest.raises(ValueError, match="suppressed"):
        await store.add_pattern(point["text"], "interest", [source], reason="test", run_id="test")
    await store.save_profile({"sections": [{"heading": "Interests", "points": [point]}]}, run_id="new")
    assert not (await store.get_profile(live_only=False))["sections"]


@pytest.mark.parametrize("kind", ["preference", "pattern", "fact"])
@pytest.mark.parametrize("op", ["state", "correct"])
async def test_scoped_stated_wording_exempt_from_inference_only(store, monkeypatch, kind, op):
    monkeypatch.setattr(llm, "try_embed", AsyncMock(return_value=[1., 0.]))
    general = "You prefer Flipkart for laptops"
    scoped = "You prefer Flipkart for laptops only when helping someone else buy"
    source = await store.create_claim(ClaimDraft(kind="preference", text=general))
    await store.reject_inference(general, claim_ids=[source])
    # Both lexical and semantic rewordings remain suppressed without testimony.
    assert await store.suppressed(kind, scoped)
    if op == "state":
        cid, _ = await store.state_claim("preference", scoped, context="helping someone else buy")
    else:
        cid, _ = await store.correct_claim(source, scoped, context="helping someone else buy")
    assert not await store.suppressed(kind, "  " + scoped.upper() + "  ")
    assert not await store.suppressed(kind, scoped, vector=[1., 0.])
    reworded = "When assisting another buyer, choose Flipkart for a laptop"
    assert await store.suppressed(kind, reworded, vector=[1., 0.])
    assert not await store.suppressed(kind, reworded, vector=[1., 0.], claim_ids=[cid])
    assert await store.suppressed(kind, general)
    assert not await store.suppressed("event", general, vector=[1., 0.])
    # Even a live stated match cannot override an ordinary kind-scope suppression.
    duplicate, _ = await store.state_claim(kind, scoped)
    await store.forget_claim(duplicate)
    assert (await store.get_claim(cid))["status"] == "live"
    assert await store.suppressed(kind, scoped, claim_ids=[cid])


@pytest.mark.parametrize("op", ["forget", "retire", "erase", "revert", "correct"])
async def test_withdrawn_stated_claim_loses_exemption(store, op):
    scoped = "You prefer Flipkart for laptops only when helping someone else buy"
    await store.reject_inference("You prefer Flipkart for laptops")
    cid, eid = await store.state_claim("preference", scoped)
    assert not await store.suppressed("preference", scoped, claim_ids=[cid])
    if op == "revert":
        await store.revert_event(eid)
    elif op == "correct":
        await store.correct_claim(cid, "You enjoy hiking")
    else:
        await getattr(store, f"{op}_claim")(cid)
    assert await store.suppressed("preference", scoped)
    assert await store.suppressed("preference", scoped, claim_ids=[cid])


async def test_scoped_stated_profile_save_and_read(store):
    scoped = "You prefer Flipkart for laptops only when helping someone else buy"
    cid, _ = await store.state_claim("preference", scoped)
    point = {"text": "When helping another buyer, choose Flipkart for a laptop", "claims": [cid]}
    profile = {"sections": [{"heading": "Preferences", "points": [point]}]}
    await store.save_profile(profile, run_id="before")
    await store.reject_inference(point["text"])
    assert (await store.get_profile())["sections"] == profile["sections"]
    await store.save_profile(profile, run_id="after")
    assert (await store.get_profile(live_only=False))["sections"] == profile["sections"]
    await store.forget_claim(cid)
    assert not (await store.get_profile())["sections"]
    assert await store.suppressed("preference", scoped, claim_ids=[cid])


async def test_actual_dream_preserves_scoped_stated_correction(store, fake):
    from discrete.process.dream import run_dream
    from tests.test_dream import _football, _pattern

    sources = await _football(store)
    general = "The user follows football scores on FotMob"
    scoped = "The user follows football scores on FotMob only when helping someone else"
    reworded = "The user follows scores for football on FotMob only when helping someone else"
    cid, _ = await store.state_claim("preference", scoped, derived_from=sources)
    await store.reject_inference(general, claim_ids=sources)
    fake.dream = {"patterns": [_pattern([cid], reworded)], "audit": []}
    assert (await run_dream(store))["patterns"]["added"] == 1
    pattern = (await store.live_claims(kind="pattern"))[0]
    assert pattern["text"] == reworded
    assert await store.confirm_pattern(pattern["id"], [cid, *sources], reason="new support", run_id="test")
    # Ordinary rewordings backed only by observed events remain suppressed.
    assert not await store.confirm_pattern(pattern["id"], sources, reason="observed only", run_id="test")
