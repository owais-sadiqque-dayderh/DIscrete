"""Clock recall must survive a later evaluation date and detail egress."""
from datetime import datetime

from discrete.models import Hit, ObservationDraft, RetrievalResult
from discrete.retrieval.hybrid import retrieve
from discrete.retrieval.query import fallback_plan, parse_interval
from discrete.retrieval.support import _record
from discrete.timeutil import from_local, to_local


async def test_clock_recall_uses_captured_day_not_wall_clock(store, fake, monkeypatch):
    start = from_local(datetime(2026, 10, 6, 22, 20))
    oid, _ = await store.add_observation(ObservationDraft(
        app='Browser', window_title='Demo', caption='Watching a coding demo',
        screen_text='A coding demo', started_at=start, ended_at=start + 10))
    monkeypatch.setattr('discrete.retrieval.query.now', lambda: start + 10 * 86400)
    monkeypatch.setattr('discrete.retrieval.temporal.now', lambda: start + 10 * 86400)
    result = await retrieve(store, 'What was I doing around 22:20?')
    assert result.hits[0].ref_id == oid
    assert to_local(result.plan.interval[0]).date() == datetime(2026, 10, 6).date()
    assert result.hits[0].verification == 'verified'
    assert 'plan' not in fake.calls and not fake.embedded


async def test_clock_uses_latest_matching_day_and_ignores_agent_output(store, fake):
    start = from_local(datetime(2026, 10, 4, 22, 31))
    for delta, origin in [(0, 'user'), (86400, 'user'), (2 * 86400, 'agent')]:
        await store.add_observation(ObservationDraft(app='Browser', window_title=str(delta),
            caption='Reading embeddings', started_at=start + delta, ended_at=start + delta + 5, origin=origin))
    result = await retrieve(store, 'What was I looking at around 22:31?')
    assert to_local(result.plan.interval[0]).day == 5
    assert [h.title for h in result.hits] == ['Browser — 86400']


def test_explicit_date_does_not_use_latest_clock_day():
    ref = from_local(datetime(2026, 10, 8))
    start, end = parse_interval('What was I doing around midnight between September 30 and October 1?', ref)
    assert to_local(start).strftime('%m-%d %H:%M') == '09-30 23:45'
    assert to_local(end).strftime('%m-%d %H:%M') == '10-01 00:15'
    assert fallback_plan('What was I doing around 22:20?').interval is None
    assert fallback_plan('What was I doing yesterday around 22:20?').interval is not None


def test_habitual_browser_query_includes_profile():
    assert fallback_plan('What browser do I normally use?').about_user


def test_verifier_sees_local_visit_boundaries_not_just_epoch():
    start = from_local(datetime(2026, 10, 6, 22, 20))
    record = _record(0, Hit('observation', 1, 'Demo', 'A demo', occurred_at=start, occurred_to=start + 10))
    assert '2026-10-06T22:20:00' in record['occurred_at']
    assert '2026-10-06T22:20:10' in record['occurred_to']


async def test_timed_out_relative_timeline_keeps_evidence_for_hedged_writer(store, fake, monkeypatch):
    from discrete.retrieval.temporal import temporal_answer
    plan = fallback_plan('What did I open right after I ran capturectl halt?')
    result = RetrievalResult(plan.raw, plan, [
        Hit('observation', 1, 'Shell', '$ capturectl halt', via='relative event anchor', verification='timed_out'),
        Hit('observation', 2, 'Files', 'Files', via='right after Shell', verification='timed_out')])
    temporal_answer(result)
    assert len(result.hits) == 2
    assert not result.page_answer


async def test_clock_candidate_cap_keeps_nearest_visits_not_earliest_burst(store, fake):
    start = from_local(datetime(2026, 10, 6, 22, 31))
    for offset in range(-170, -140):
        await store.add_observation(ObservationDraft(app='Browser', window_title=str(offset),
            caption='Earlier browsing', started_at=start + offset, ended_at=start + offset + 1))
    oid, _ = await store.add_observation(ObservationDraft(app='Browser', window_title='Embedding rankings',
        caption='Reading embedding rankings', started_at=start, ended_at=start + 5))
    result = await retrieve(store, 'What was I looking at around 22:31?')
    assert any(h.ref_id == oid for h in result.hits)
    assert len(result.hits) <= 12


async def test_clock_recall_does_not_override_an_edit(store, fake):
    start = from_local(datetime(2026, 10, 6, 22, 20))
    await store.add_observation(ObservationDraft(app='Notes', caption='A report', started_at=start, ended_at=start + 5))
    result = await retrieve(store, 'Rewrite this: meet at 22:20')
    assert result.plan.intent == 'edit' and not result.hits
