"""Detail evidence is selected from the full stored source."""
from discrete.bubble.engine import render_records
from discrete.models import ObservationDraft, RetrievalResult
from discrete.retrieval.hybrid import _exhaustive, materialize
from discrete.retrieval.query import QueryPlan


async def test_deep_details_share_one_pack(store):
    text = 'Navigation filler. ' * 200 + '\nBudget: 740 units for helium.\n' + 'Footer filler. ' * 100
    oid, _ = await store.add_observation(ObservationDraft(app='Notes', window_title='Lab', screen_text=text))
    plan = QueryPlan('What was the helium budget?', terms=['helium', 'budget'])
    hits = await materialize(store, 'observation', [oid], terms=plan.terms)
    hit = hits[oid]
    hit.verification = "verified"
    assert '740' in hit.body
    assert hit.source_spans and hit.source_spans[0][0] > 1500
    assert hit.body in render_records(RetrievalResult(plan.raw, plan, [hit]))


async def test_short_interval_expands_summary_into_details(store):
    oid, _ = await store.add_observation(ObservationDraft(
        app='Notes', window_title='Lab', screen_text='Noise. ' * 400 + 'Helium budget: 740 units.',
        started_at=100, ended_at=110))
    eid = await store.open_episode('s', 100)
    await store.assign_episode([oid], eid)
    await store.mark_episode_ready(eid)
    await store.close_episode(eid, title='Lab', summary='Reviewed lab planning', apps=['Notes'], sites=[], engaged_fraction=1)
    plan = QueryPlan('What was the helium budget?', terms=['helium', 'budget'], interval=(90, 120), intent='temporal')
    result = RetrievalResult(plan.raw, plan)
    await _exhaustive(store, plan, result)
    assert any(h.ref_kind == 'observation' and '740' in h.body for h in result.hits)
