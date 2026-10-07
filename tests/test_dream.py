"""Dreaming: patterns across episodes (checked in code and by the verifier), audits,
conflict resolution, staleness and the profile. Every change is a revertible event."""

from __future__ import annotations

import json

import pytest

from discrete.config import T
from discrete.models import ClaimDraft, ObservationDraft
from discrete.process import dream as dr
from discrete.process.dream import run_dream
from discrete.process.profile import render_profile
from discrete.timeutil import now, to_local

DAY = 86400.0


async def _memory(store, text: str, *, day: int, episode: int, kind: str = "event", modality: str = "engaged",
                  hours_ago: float = 0.0) -> int:
    """A claim with evidence in its own episode, `day` days ago."""
    t = now() - day * DAY - hours_ago * 3600
    await store.execute("INSERT OR IGNORE INTO episodes(id, session_id, started_at, ended_at, status, created_at, "
                        "updated_at) VALUES (?, 's', ?, ?, 'closed', ?, ?)", (episode, t, t + 60, t, t))
    oid, _ = await store.add_observation(ObservationDraft(
        app="Arc", window_title=f"{text} {episode}", screen_text=f"{text} {episode} " * 3, caption=text,
        session_id="s", started_at=t, ended_at=t + 5, keys=1 if modality == "engaged" else 0))
    await store.assign_episode([oid], episode)
    return await store.create_claim(ClaimDraft(kind=kind, text=text, modality=modality, episode_id=episode,
                                               occurred_from=t, occurred_to=t + 5,
                                               evidence=[{"observation_id": oid, "quote": text}]), actor="draft")


async def _football(store) -> list[int]:
    return [await _memory(store, "The user checked football scores on FotMob", day=d, episode=10 + d)
            for d in (0, 1, 2)]


def _pattern(ids, text="The user follows football scores on FotMob", category="interest") -> dict:
    return {"text": text, "category": category, "support_ids": ids}


async def test_a_supported_pattern_is_added_with_its_support_and_evidence(store, fake):
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}

    report = await run_dream(store)

    assert report["patterns"]["added"] == 1
    [p] = await store.live_claims(kind="pattern")
    assert json.loads(p["derived_from"]) == sorted(ids) and p["context"] == "interest"
    assert len(await store.evidence_for_claim(p["id"])) == 3
    st = (await store.strengths([p["id"]]))[p["id"]]
    assert (st.episodes, st.days, st.tier) == (3, 3, "established")
    [ev] = await store.recent_events(op="dream_add")
    assert ev["actor"] == "dream" and ev["run_id"] == report["run_id"]


@pytest.mark.parametrize("proposal, why", [
    (lambda ids: _pattern(ids[:2]), "too_few_episodes"),
    (lambda ids: _pattern([*ids, 99999]), "unknown_support"),
    (lambda ids: _pattern(ids, category="personality"), "bad_category"),
    (lambda ids: _pattern(ids, text="Football fan"), "bad_text"),
    (lambda ids: _pattern(ids, text="The user checks FotMob every day", category="habit"), "frequency_overclaim"),
])
async def test_code_checks_reject_bad_patterns(store, fake, proposal, why):
    ids = await _football(store)
    fake.dream = {"patterns": [proposal(ids)], "audit": []}
    report = await run_dream(store)
    assert report["patterns"]["added"] == 0 and report["patterns"]["rejected"] == {why: 1}


async def test_habits_need_several_days_projects_do_not(store, fake, monkeypatch):
    # All three sittings are on one local calendar day, including when run before dawn.
    noon = to_local(now()).replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
    monkeypatch.setattr(__name__ + ".now", lambda: noon)
    ids = [await _memory(store, "The user worked on the AMO agent repository", day=0, episode=20 + i,
                         hours_ago=2 * i) for i in range(3)]
    fake.dream = {"patterns": [_pattern(ids, "The user works on AMO in the evenings", "habit"),
                               _pattern(ids, "The user is building the AMO agent", "project")], "audit": []}
    report = await run_dream(store)
    assert report["patterns"]["added"] == 1 and report["patterns"]["rejected"] == {"too_few_days": 1}
    [p] = await store.live_claims(kind="pattern")
    assert p["context"] == "project"


async def test_the_verifier_vetoes_and_patterns_never_rest_on_patterns(store, fake):
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}
    fake.dream_rejects = {0}
    assert (await run_dream(store))["patterns"]["rejected"] == {"not_verified": 1}
    fake.dream_rejects = set()
    await run_dream(store)
    [p] = await store.live_claims(kind="pattern")
    fake.dream = {"patterns": [_pattern([p["id"], *ids], "The user is a sports fan")], "audit": []}
    assert (await run_dream(store))["patterns"]["rejected"] == {"rests_on_pattern": 1}


async def test_a_restated_pattern_confirms_the_old_one(store, fake):
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}
    await run_dream(store)
    more = await _memory(store, "The user checked football scores on FotMob", day=3, episode=30)
    fake.dream = {"patterns": [_pattern([*ids, more])], "audit": []}
    report = await run_dream(store)
    assert report["patterns"] | {"rejected": {}} == {"proposed": 1, "added": 0, "confirmed": 1, "retired": 0,
                                                    "rejected": {}}
    [p] = await store.live_claims(kind="pattern")
    assert more in json.loads(p["derived_from"])


async def test_audit_retires_and_withdrawn_support_retires(store, fake):
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}
    await run_dream(store)
    [p] = await store.live_claims(kind="pattern")
    for cid in ids:
        await store.forget_claim(cid)
    fake.dream = {"patterns": [], "audit": []}
    assert (await run_dream(store))["patterns"]["retired"] == 1
    assert (await store.get_claim(p["id"]))["status"] == "retired"


async def test_conflicts_resolve_to_the_engaged_newer_claim_and_can_be_reverted(store, fake, monkeypatch):
    monkeypatch.setattr(T, "CONFLICT_SIM", -1.0)  # bag-of-words vectors in tests: pair everything
    old = await _memory(store, "The user uses Arc as their main browser", day=5, episode=40, kind="fact",
                        modality="passive")
    new = await _memory(store, "The user switched to Helium as their main browser", day=0, episode=41, kind="fact")
    from discrete.process.embed_worker import drain_outbox
    await drain_outbox(store)
    fake.conflicts = {"pairs": [{"index": 0, "verdict": "conflict", "reason": "one main browser"}]}

    report = await run_dream(store)

    assert report["conflicts"]["resolved"] == 1
    assert (await store.get_claim(old))["status"] == "invalidated"
    assert (await store.get_claim(new))["status"] == "live"
    [ev] = await store.recent_events(op="resolve")
    assert "acted on" in ev["reason"]
    # Asked once per pair.
    await run_dream(store)
    assert fake.calls.count("conflicts") == 1
    await store.revert_event(ev["id"])
    assert (await store.get_claim(old))["status"] == "live"


async def test_old_lasting_memories_go_stale_events_never(store, fake):
    fact = await _memory(store, "The user uses Figma for logo work", day=T.STALE_DAYS + 5, episode=50, kind="fact")
    event = await _memory(store, "The user exported a logo from Figma", day=T.STALE_DAYS + 5, episode=51)
    await run_dream(store)
    assert (await store.get_claim(fact))["flag"] == "stale"
    assert (await store.get_claim(event))["flag"] == ""


async def test_the_profile_keeps_only_points_backed_by_real_memories(store, fake):
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}
    await run_dream(store)
    [p] = await store.live_claims(kind="pattern")
    fake.profile = {"sections": [
        {"heading": "Interests", "points": [{"text": "They follow football on FotMob.", "claim_ids": [p["id"]]}]},
        {"heading": "Interests", "points": [{"text": "They love cricket.", "claim_ids": [12345]}]},
        {"heading": "Secrets", "points": [{"text": "x", "claim_ids": [p["id"]]}]}]}
    fake.dream = {"patterns": [], "audit": []}

    await run_dream(store)

    profile = await store.get_profile()
    assert [p["text"] for s in profile["sections"] for p in s["points"]] == ["They follow football on FotMob."]
    assert "Interests:" in render_profile(profile)
    [ev] = await store.recent_events(op="profile")
    await store.revert_event(ev["id"])
    assert await store.get_profile() == {}


async def test_dreaming_waits_for_time_and_activity(store, fake, monkeypatch):
    monkeypatch.setattr(T, "DREAM_INTERVAL_MIN", 60.0)
    monkeypatch.setattr(T, "DREAM_MIN_NEW_EPISODES", 2)
    await _memory(store, "a", day=0, episode=60)
    assert not await dr.due(store)  # one new episode
    await _memory(store, "b", day=0, episode=61)
    assert await dr.due(store)
    await store.set_meta("last_dream", str(now()))
    assert not await dr.due(store)  # too soon
    monkeypatch.setattr(T, "DREAM_INTERVAL_MIN", 0.0)
    assert not await dr.due(store)  # 0 = only by hand


async def test_episodes_of_one_sitting_are_one_occasion(store, fake):
    """Review #11: episodes close every 40 captures, so one 25-minute read is several episodes."""
    ids = [await _memory(store, "The user read about agent memory repositories", day=0, episode=70 + i,
                         hours_ago=0.1 * i) for i in range(4)]
    fake.dream = {"patterns": [_pattern(ids, "The user is researching agent memory", "project")], "audit": []}
    assert (await run_dream(store))["patterns"]["rejected"] == {"single_occasion": 1}


async def test_a_crashing_step_neither_stops_the_others_nor_reruns_every_minute(store, fake, monkeypatch):
    """Review #1/#2: a list reply or a crash used to abort the dream before last_dream."""
    ids = await _football(store)
    fake.dream = ["not", "an", "object"]
    fake.profile = {"sections": []}
    report = await run_dream(store)
    assert "error" in report["patterns"] and "stale" in report and "error" not in report["stale"]
    assert float(await store.get_meta("last_dream", "0")) > 0
    assert ids


async def test_quote_sample_does_not_erase_support_membership(store, fake, monkeypatch):
    monkeypatch.setattr(T, "EVIDENCE_MAX_SUPPORTS", 1)
    ids = await _football(store)
    fake.dream = {"patterns": [_pattern(ids)], "audit": []}
    await run_dream(store)
    [p] = await store.live_claims(kind="pattern")
    st = (await store.strengths([p["id"]]))[p["id"]]
    assert (st.episodes, st.days) == (3, 3)
    assert len(await store.evidence_for_claim(p["id"])) == 1


async def test_one_reinforced_claim_can_supply_separate_occasions(store, fake):
    ids = await _football(store)
    [merged] = await store.consolidate_cluster(await store.get_claims(ids), [{
        "text": "The user checks football scores on FotMob", "input_ids": ids}])
    fake.dream = {"patterns": [_pattern([merged], "The user regularly checks football scores", "habit")], "audit": []}
    # Three distinct evidence days, but frequency language requires still more days.
    report = await run_dream(store)
    assert report["patterns"]["rejected"] == {"frequency_overclaim": 1}
    fake.dream = {"patterns": [_pattern([merged], category="habit")], "audit": []}
    assert (await run_dream(store))["patterns"]["added"] == 1
