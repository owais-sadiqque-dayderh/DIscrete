"""Relative event retrieval uses real visit boundaries without model calls."""

import pytest

from discrete.models import ClaimDraft, ObservationDraft, RetrievalResult
from discrete.retrieval.grounding import evaluate
from discrete.retrieval.hybrid import retrieve
from discrete.retrieval.query import EventQueryPlan, fallback_plan
from discrete.retrieval.temporal import relative_timeline

T0 = 1_789_000_000.0


async def add(store, title, t, *, app="Browser", text="", keys=0, session="s1", end=None, url=""):
    oid, _ = await store.add_observation(ObservationDraft(
        app=app, window_title=title, screen_text=text or title, keys=keys,
        started_at=T0 + t, ended_at=T0 + (end if end is not None else t + 5), session_id=session, url=url))
    return oid


@pytest.mark.parametrize("cue,direction,last", [
    ("right after I stopped capture", "after", False),
    ("just after I ran a task", "after", False),
    ("after I installed a tool", "after", False),
    ("before I opened a file", "before", False),
    ("right before I launched an app", "before", False),
    ("when I finished the setup", "after", True),
])
def test_planner_event_precedes_enumeration(cue, direction, last):
    plan = fallback_plan(f"What did I open {cue}?")
    assert isinstance(plan, EventQueryPlan)
    assert plan.intent == "temporal"
    assert plan.anchor.direction == direction
    assert plan.anchor.last == last
    assert not plan.anchor.phrase.startswith("I ")


def test_clock_and_why_keep_existing_plans():
    assert not isinstance(fallback_plan("What did I open after 10:30?"), EventQueryPlan)
    assert fallback_plan("Why did I open a page after installing a browser?").intent == "why"


async def test_command_input_beats_help_and_plain_answer_starts_after(store, fake):
    anchor = await add(store, "Shell", 20, app="Terminal", text="$ capturectl halt --all", keys=24)
    await add(store, "Shell", 26, app="Terminal", text="capturectl halt stopped recording")
    first = await add(store, "Downloads", 30, app="Finder")
    await add(store, "Downloads", 36, app="Finder", text="Downloads scrolled")
    second = await add(store, "Project", 40, app="Editor")
    await add(store, "Shell", 80, app="Terminal", text="Help: available commands\ncapturectl halt", keys=20)
    await add(store, "Other session", 90, session="s2")
    result = await retrieve(store, "What did I open right after I stopped Capture with capturectl halt?")
    assert [h.ref_id for h in result.hits] == [anchor, first, second]
    assert "capturectl halt --all" in result.hits[0].body
    assert result.hits[1].via.endswith("(+5s)")
    assert result.page_answer.splitlines()[0].startswith("- Finder — Downloads")
    assert evaluate(result).mode == "pages"
    assert "plan" not in fake.calls and not fake.embedded


async def test_finished_activity_uses_last_visit_not_later_log_or_widened_row(store):
    early = await add(store, "Nimbus setup", 10, text="Nimbus setup welcome")
    last = await add(store, "Nimbus setup", 50, text="Nimbus setup ready to browse")
    first = await add(store, "New Tab", 54.9999, end=60)
    await add(store, "Log", 100, app="Terminal", text="Earlier Nimbus setup ready to browse")
    # An exact dedup return widens the observation's interval, but not its visit.
    await add(store, "Nimbus setup", 1000, text="Nimbus setup welcome", session="s2")
    query = "What did I do right after finishing the Nimbus setup?"
    result = RetrievalResult(query, fallback_plan(query))
    assert await relative_timeline(store, result)
    assert result.hits[0].ref_id == early  # latest actual visit of the activity
    assert result.hits[0].occurred_at == T0 + 1000
    # Restricting the session interval finds the final setup step in that session.
    result.plan.interval = (T0, T0 + 200)
    assert await relative_timeline(store, result)
    assert [h.ref_id for h in result.hits[:2]] == [last, first]
    assert result.hits[0].occurred_to == T0 + 55


async def test_before_is_nearest_first_and_distinct(store, fake):
    farther = await add(store, "Mail", 10)
    await add(store, "Project", 20, app="Editor")
    nearest = await add(store, "Project", 26, app="Editor", text="Project editing")
    anchor = await add(store, "Shell", 40, app="Terminal", text="$ deployctl publish", keys=20)
    await add(store, "After", 50)
    result = await retrieve(store, "What did I do right before I ran deployctl publish?")
    assert [h.ref_id for h in result.hits] == [anchor, nearest, farther]
    assert result.hits[1].via.endswith("(-9s)")
    assert result.page_answer.splitlines()[0].startswith("- Editor — Project")


async def test_no_anchor_keeps_old_route(store, monkeypatch):
    import discrete.retrieval.hybrid as hybrid

    seen = []

    async def old_route(store, plan, result, qvec):
        seen.append(plan.intent)

    monkeypatch.setattr(hybrid, "enumerate_pages", old_route)
    result = await retrieve(store, "What did I open right after I finished the missing setup?")
    assert seen == ["enumerate"]
    assert not isinstance(result.plan, EventQueryPlan)
    assert not result.hits and not result.page_answer


async def test_verifier_rejecting_anchor_prevents_plain_answer(store, fake):
    await add(store, "Shell", 10, app="Terminal", text="$ taskctl end", keys=5)
    await add(store, "Inbox", 20)
    fake.support_indices = [1]
    result = await retrieve(store, "What did I open after I ran taskctl end?")
    assert not result.page_answer
    assert evaluate(result).mode != "pages"


async def test_window_and_distinct_limit_are_tunable(store, monkeypatch):
    await add(store, "Shell", 10, app="Terminal", text="$ taskctl end", keys=5)
    first = await add(store, "Page", 20, url="https://example.com/one")
    await add(store, "Page", 25, url="https://example.com/two")
    await add(store, "Later", 50)
    monkeypatch.setenv("DISCRETE_TEMPORAL_AFTER_S", "20")
    monkeypatch.setenv("DISCRETE_TEMPORAL_MAX_MOMENTS", "1")
    result = await retrieve(store, "What did I do after I ran taskctl end?")
    assert [h.ref_id for h in result.hits[1:]] == [first]


async def test_help_alone_is_not_command_execution(store):
    await add(store, "Shell", 10, app="Terminal", text="Help: available commands\ncapturectl halt", keys=20)
    query = "What did I open after I stopped Capture with capturectl halt?"
    result = RetrievalResult(query, fallback_plan(query))
    assert not await relative_timeline(store, result)


@pytest.mark.parametrize("kind", ["episode", "claim"])
async def test_summary_and_event_claim_can_supply_anchor(store, kind):
    if kind == "episode":
        oid = await add(store, "Installer", 10, text="Installation complete")
        eid = await store.open_episode("s1", T0 + 10)
        await store.assign_episode([oid], eid)
        await store.mark_episode_ready(eid)
        await store.close_episode(eid, title="Nimbus setup", summary="Completed Nimbus setup",
                                  apps=["Browser"], sites=[], engaged_fraction=1)
        anchor = eid
    else:
        anchor = await store.create_claim(ClaimDraft(kind="event", text="Completed Nimbus setup",
                                                     occurred_from=T0 + 10, occurred_to=T0 + 15))
    next_id = await add(store, "Inbox", 20)
    query = "What did I do after finishing Nimbus setup?"
    result = RetrievalResult(query, fallback_plan(query))
    assert await relative_timeline(store, result)
    assert result.hits[0].ref_kind == kind
    assert result.hits[0].ref_id == anchor
    assert result.hits[1].ref_id == next_id


async def test_same_title_different_known_urls_are_distinct(store):
    await add(store, "Shell", 10, app="Terminal", text="$ taskctl end", keys=5)
    one = await add(store, "Page", 20, url="https://example.com/one")
    two = await add(store, "Page", 25, url="https://example.com/two")
    query = "What did I open after I ran taskctl end?"
    result = RetrievalResult(query, fallback_plan(query))
    assert await relative_timeline(store, result)
    assert [h.ref_id for h in result.hits[1:]] == [one, two]


@pytest.mark.parametrize("question", [
    "What did I do after midnight on October 1?",          # a clock/date is an interval
    "What did Yacine Mahdid reply after I sent the PDF?",  # someone else's content: keep refusal path
    "What did ChatGPT say after I asked about Mem0?",
])
def test_relative_event_only_for_users_own_next_steps(question):
    from discrete.retrieval.query import fallback_plan
    assert getattr(fallback_plan(question), "anchor", None) is None
