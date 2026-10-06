"""Causal trails: what led to a completed action, checked in code and by the verifier."""

from __future__ import annotations

import json

from discrete.models import ClaimDraft, Hit, ObservationDraft, QueryPlan, RetrievalResult
from discrete.process.trails import build_trails
from discrete.retrieval.hybrid import expand_why
from discrete.retrieval.query import fallback_plan

T0 = 1_790_000_000.0


async def _obs(store, n: int, title: str, text: str, caption: str = "", url: str = "", keys: int = 0) -> int:
    oid, _ = await store.add_observation(ObservationDraft(
        app="Arc", window_title=title, url=url, screen_text=text, caption=caption or title, session_id="s",
        started_at=T0 + n, ended_at=T0 + n + 2, keys=keys, clicks=1))
    return oid


async def _helium(store):
    theo = await _obs(store, 0, 'Theo - t3.gg on X: "I asked Opus 5.5"', "Helium File Edit View Tabs Profiles Window\n"
                      "feat: pin favorite projects", "The user views Theo's post with a demo video.")
    music = await _obs(store, 10, "Spotify", "Blue Hair TV Girl", "The user plays music.")
    search = await _obs(store, 20, "helium - Google Search", "helium\nHelium Browser privacy-first",
                        "Google results for helium.", url="https://www.google.com/search?q=helium", keys=6)
    dl = await _obs(store, 40, "Helium · Download", "Thank you for downloading Helium!", "Download confirmation.")
    cid = await store.create_claim(ClaimDraft(
        kind="event", text="The user downloaded the Helium browser for macOS", modality="engaged", action="downloaded",
        occurred_from=T0 + 40, occurred_to=T0 + 42,
        evidence=[{"observation_id": dl, "quote": "Thank you for downloading Helium!"}]))
    return theo, music, search, dl, cid


def test_why_questions_are_planned_as_why():
    for q in ("why did I download helium browser?", "What made me install Helium?", "how did I end up on that site"):
        assert fallback_plan(q).intent == "why", q
    assert fallback_plan("what did I download today?").intent != "why"


async def test_a_trail_keeps_only_real_earlier_moments_that_name_the_thing(store, fake):
    theo, music, search, dl, cid = await _helium(store)
    fake.trail = {"links": [
        {"observation_id": theo, "role": "trigger", "quote": "Helium File Edit View", "confidence": 0.7,
         "explanation": "Theo's post showed a demo recorded in the Helium browser."},
        {"observation_id": search, "role": "step", "quote": "helium", "confidence": 0.9,
         "explanation": "The user then searched Google for helium."},
        {"observation_id": music, "role": "trigger", "quote": "Blue Hair", "confidence": 0.5,
         "explanation": "Music was playing."},  # does not name the thing
        {"observation_id": theo, "role": "step", "quote": "Helium is great", "confidence": 0.5,
         "explanation": "made-up quote"},  # not on that screen
        {"observation_id": dl, "role": "step", "quote": "Helium", "confidence": 0.5,
         "explanation": "the action itself"},  # not before it
    ]}

    assert await build_trails(store) == 2

    links = await store.links_into("claim", [cid])
    assert {(lk["src_id"], lk["relation"]) for lk in links} == {(theo, "led_to"), (search, "step")}
    [ev] = await store.recent_events(op="trail")
    assert ev["claim_id"] == cid and json.loads(ev["data"])["terms"][0] == "helium"
    # Done once per action.
    assert await build_trails(store) == 0 and fake.calls.count("trail") == 1


async def test_the_verifier_can_veto_a_link(store, fake):
    theo, _, search, _, cid = await _helium(store)
    fake.trail = {"links": [
        {"observation_id": theo, "role": "trigger", "quote": "Helium File Edit", "confidence": 0.7,
         "explanation": "Theo's post showed Helium."},
        {"observation_id": search, "role": "step", "quote": "helium", "confidence": 0.9,
         "explanation": "searched for it"}]}
    fake.trail_rejects = {0}
    await build_trails(store)
    assert [lk["src_id"] for lk in await store.links_into("claim", [cid])] == [search]


async def test_reverting_a_trail_retires_its_links(store, fake):
    theo, _, _, _, cid = await _helium(store)
    fake.trail = {"links": [{"observation_id": theo, "role": "trigger", "quote": "Helium File", "confidence": 0.7,
                             "explanation": "Theo's post showed Helium."}]}
    await build_trails(store)
    [ev] = await store.recent_events(op="trail")
    await store.revert_event(ev["id"])
    assert await store.links_into("claim", [cid]) == []


async def test_why_retrieval_puts_the_trail_right_after_the_action(store, fake):
    theo, _, search, _, cid = await _helium(store)
    fake.trail = {"links": [
        {"observation_id": theo, "role": "trigger", "quote": "Helium File Edit", "confidence": 0.7,
         "explanation": "Theo's post showed a demo recorded in Helium."},
        {"observation_id": search, "role": "step", "quote": "helium", "confidence": 0.9,
         "explanation": "searched for it"}]}
    await build_trails(store)
    action = Hit("claim", cid, "The user downloaded the Helium browser for macOS", "", claim_kind="event",
                 action="downloaded", occurred_at=T0 + 40, strong=True)
    other = Hit("claim", 999, "The user played music", "", claim_kind="event", occurred_at=T0 + 10)
    result = RetrievalResult(query="why did I download helium", plan=QueryPlan(raw="", intent="why"),
                             hits=[action, other])

    await expand_why(store, result)

    assert [(h.ref_kind, h.ref_id) for h in result.hits] == [
        ("claim", cid), ("observation", theo), ("observation", search), ("claim", 999)]
    assert result.hits[1].via.startswith(f"led to claim {cid}") and result.hits[1].strong


async def test_without_a_trail_the_moments_before_the_action_are_offered(store, fake):
    theo, music, search, _, cid = await _helium(store)
    action = Hit("claim", cid, "The user downloaded the Helium browser for macOS", "", claim_kind="event",
                 action="downloaded", occurred_at=T0 + 40, strong=True)
    result = RetrievalResult(query="why did I download helium",
                             plan=QueryPlan(raw="", intent="why", terms=["download", "helium"]), hits=[action])
    await expand_why(store, result)
    via = {h.ref_id: h.via for h in result.hits if h.ref_kind == "observation"}
    assert via == {theo: f"before claim {cid}, about the same thing",
                   music: f"just before claim {cid}",
                   search: f"before claim {cid}, about the same thing"}
    theo_hit = next(h for h in result.hits if h.ref_id == theo and h.ref_kind == "observation")
    assert "Helium File Edit" in theo_hit.body  # the screen line naming it, not just the caption
    assert not any(h.strong for h in result.hits if h.ref_kind == "observation")  # the support check judges
