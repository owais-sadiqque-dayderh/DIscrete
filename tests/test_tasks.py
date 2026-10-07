"""Audit #8: task boundaries, conservative pre-draft merging and task recall."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from discrete import textkit as tk
from discrete.cli import build_parser, cmd_tasks
from discrete.config import T
from discrete.models import Hit, QueryPlan, RetrievalResult
from discrete.process.embed_worker import drain_outbox
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval.hybrid import expand_tasks, retrieve
from discrete.store import Store, db
from tests.test_pipeline import T0, obs


async def tracked(store, tracker, *, app='Arc', site='cognition.com', dt=0, vec=None):
    oid, new = await store.add_observation(obs(app=app, site=site, title=f'{dt}-{app}',
                                             text=f'Unique capture {dt} {app}', t=T0 + dt))
    assert new
    closed = await tracker.on_observation(oid, await store.get_observation(oid), session_id='s1',
                                          vec=np.array(vec, dtype=np.float32) if vec is not None else None)
    return oid, closed


async def ready(store, *, dt=0, session='s1', vec=(1., 0.), entities=('Agent Memory Repo',), count=1):
    eid = await store.open_episode(session, T0 + dt)
    ids = []
    for i in range(count):
        oid, _ = await store.add_observation(obs(title=f'{eid}-{i}', text=f'Unique screen {eid} {i}',
                                               t=T0 + dt + i * 10, session=session, entities=list(entities)))
        await store.assign_episode([oid], eid)
        if vec is not None:
            await store.save_embeddings([('observation', oid, vec, f'{eid}-{i}')])
        ids.append(oid)
    await store.mark_episode_ready(eid)
    return eid, ids


async def closed(store, *, dt=0, title='Agent Memory Repo research', summary='Investigated Agent Memory Repo design',
                 entities=('Agent Memory Repo',), vec=None):
    eid, _ = await ready(store, dt=dt, entities=entities)
    await store.close_episode(eid, title=title, summary=summary, apps=['Arc'], sites=[], engaged_fraction=1.)
    if vec is not None:
        await store.save_embeddings([('episode', eid, vec, tk.content_hash(f'{title}\n{summary}'))])
    return eid


async def test_detour_across_multiple_pairs_returns_without_split(store):
    tr = EpisodeTracker(store)
    for app, site, dt in [('Arc', 'cognition.com', 0), ('Slack', '', 10),
                          ('Terminal', '', 40), ('Arc', 'cognition.com', 80)]:
        _, ends = await tracked(store, tr, app=app, site=site, dt=dt, vec=(1., 0.))
        assert ends == []
    await tr.close_current()
    assert len(await store.q('SELECT * FROM episodes')) == 1


async def test_away_timer_does_not_reset_for_each_new_pair(store):
    tr = EpisodeTracker(store)
    ids = []
    for app, site, dt in [('Arc', 'cognition.com', 0), ('Helium', '', 10),
                          ('Helium', 'github.com', 60), ('Terminal', '', 110)]:
        oid, ends = await tracked(store, tr, app=app, site=site, dt=dt)
        ids.append(oid)
    assert len(ends) == 1
    rows = await store.get_observations(ids)
    assert rows[0]['episode_id'] != rows[1]['episode_id']
    assert len({r['episode_id'] for r in rows[1:]}) == 1
    assert (await store.get_episode(ends[0]))['ended_at'] == T0 + 5


async def test_topic_shift_splits_across_pairs_before_away_timeout(store):
    tr = EpisodeTracker(store)
    first, _ = await tracked(store, tr, vec=(1., 0.))
    second, _ = await tracked(store, tr, app='Helium', dt=10, vec=(0., 1.))
    third, ends = await tracked(store, tr, app='Terminal', dt=20, vec=(0., 1.))
    assert len(ends) == 1
    rows = await store.get_observations([first, second, third])
    assert rows[0]['episode_id'] != rows[1]['episode_id'] == rows[2]['episode_id']
    assert np.allclose(tr.state.centroid, [0., 1.])


async def test_missing_embedding_breaks_consecutive_shift(store):
    tr = EpisodeTracker(store)
    for dt, vec in [(0, (1., 0.)), (10, (0., 1.)), (20, None), (30, (0., 1.))]:
        _, ends = await tracked(store, tr, dt=dt, vec=vec)
        assert not ends


async def test_similar_three_way_over_split_merges_before_draft(store, fake):
    episodes, ids = [], []
    for dt in (0, 20, 40):
        eid, obs_ids = await ready(store, dt=dt)
        episodes.append(eid)
        ids.extend(obs_ids)
    fake.draft = {'title': 'Agent Memory Repo', 'summary': 'Researched its design.', 'claims': []}
    outcomes = await ProcessPipeline(store).process_ready()
    assert [o.episode_id for o in outcomes] == episodes[:1]
    assert fake.calls.count('draft') == 1
    assert {r['episode_id'] for r in await store.get_observations(ids)} == {episodes[0]}
    assert [(await store.get_episode(i))['status'] for i in episodes] == ['closed', 'merged', 'merged']
    assert len(await store.episodes_in_interval(T0, T0 + 100)) == 1
    assert (await store.get_episode(episodes[0]))['ended_at'] == T0 + 45


@pytest.mark.parametrize('barrier', ['drafted', 'attempted', 'checkpoint', 'gap', 'session', 'topic',
                                    'missing_vector', 'obs_cap', 'time_cap', 'intervening'])
async def test_merge_guards(store, monkeypatch, barrier):
    first, ids = await ready(store)
    if barrier == 'intervening':
        await ready(store, dt=10, vec=(0., 1.))
    second, _ = await ready(store, dt=100 if barrier == 'gap' else 20,
                            session='s2' if barrier == 'session' else 's1',
                            vec=None if barrier == 'missing_vector' else (0., 1.) if barrier == 'topic' else (1., 0.))
    if barrier == 'drafted':
        await store.close_episode(first, title='research', summary='done', apps=[], sites=[], engaged_fraction=1.)
    if barrier == 'attempted':
        await store.execute('UPDATE episodes SET draft_attempts=1 WHERE id=?', (first,))
    if barrier == 'checkpoint':
        await store.save_draft_chunk(first, 0, 'hash', {'title': 'paid result'})
    if barrier == 'obs_cap':
        monkeypatch.setattr(T, 'EPISODE_MAX_OBS', 1)
    if barrier == 'time_cap':
        monkeypatch.setattr(T, 'EPISODE_MAX_S', 15.)
    assert await store.begin_draft(second)
    assert (await store.get_observation(ids[0]))['episode_id'] == first
    assert (await store.get_episode(first))['status'] != 'merged'


async def test_drafted_three_episode_research_thread_shares_one_task(store, fake):
    episodes = []
    for i, site in enumerate(['cognition.com', 'github.com', 'chatgpt.com']):
        eid, _ = await ready(store, dt=i * 300)
        fake.draft = {'title': 'Agent Memory Repo research', 'summary': f'Researched Agent Memory Repo on {site}',
                      'claims': []}
        assert await ProcessPipeline(store).draft_episode(eid)
        # Uses the normal episode outbox, with the existing fake embedding call.
        await drain_outbox(store)
        episodes.append(eid)
    tasks = await store.list_tasks()
    assert len(tasks) == 1 and tasks[0]['n_episodes'] == 3
    assert tasks[0]['title'] == 'Agent Memory Repo research'
    assert tasks[0]['first_seen'] == T0 and tasks[0]['last_seen'] == T0 + 605
    assert {r['id'] for r in await store.task_siblings(episodes[:1], limit=5)} == set(episodes[1:])
    assert not any('task' in x.lower() for x in fake.embedded)


async def test_embedding_assignment_joins_different_titles_without_extra_call(store, fake):
    first = await closed(store, title='Agent Memory architecture', summary='Repo design', vec=(1., 0.))
    second = await closed(store, dt=300, title='Comparing implementations', summary='Technical analysis')
    assert len(await store.list_tasks()) == 2  # little lexical overlap
    before = len(fake.embedded)
    await store.save_embeddings([('episode', second, (0.99, 0.01),
                                  tk.content_hash('Comparing implementations\nTechnical analysis'))])
    assert len(fake.embedded) == before
    assert len(await store.list_tasks()) == 1
    assert (await store.task_siblings([first], limit=5))[0]['id'] == second


@pytest.mark.parametrize('difference', ['entities', 'window', 'embedding', 'model', 'dimensions'])
async def test_task_assignment_guards(store, difference, monkeypatch):
    await closed(store, vec=(1., 0.))
    if difference == 'model':
        monkeypatch.setenv('EMBED_LM_MODEL_NAME', 'other-model')
    await closed(store, dt=T.TASK_WINDOW_S + 100 if difference == 'window' else 100,
                 entities=('Helium',) if difference == 'entities' else ('Agent Memory Repo',),
                 # Different text keeps lexical fallback from hiding model mismatch.
                 title='Implementation comparison', summary='Technical evaluation',
                 vec=(0., 1.) if difference == 'embedding' else (1., 0., 0.) if difference == 'dimensions' else (1., 0.))
    assert len(await store.list_tasks()) == 2


async def test_retrieve_expands_task_siblings_after_strong_episode_match(store, fake, monkeypatch):
    first = await closed(store, title='Agent Memory Repo research', summary='Agent Memory Repo architecture', vec=(1., 0.))
    second = await closed(store, dt=300, title='Agent Memory Repo implementation', summary='Compared storage backends', vec=(1., 0.))
    third = await closed(store, dt=600, title='Agent Memory Repo conclusions',
                         summary='Preferred SQLite for the prototype', vec=(1., 0.))
    # Only the anchor is found by the initial search. Siblings must enter through tasks.
    async def fts(kind, text, **kwargs):
        return [first] if kind == 'episode' else []
    async def dense(*args, **kwargs):
        return []
    monkeypatch.setattr(store, 'fts_ids', fts)
    monkeypatch.setattr(store, 'dense_search', dense)
    monkeypatch.setattr(T, 'COLD_START_ROWS', 0)
    result = await retrieve(store, 'Agent Memory Repo research')
    assert {h.ref_id for h in result.hits if h.ref_kind == 'episode'} == {first, second, third}
    assert [h.via for h in result.hits if h.ref_id == second] == ['task 1 sibling']
    assert next(h for h in result.hits if h.ref_id == second).strong  # confirmed by the support verifier
    assert 'support' in fake.calls


async def test_task_expansion_is_bounded_and_obeys_interval_and_strength(store, monkeypatch):
    first = await closed(store, vec=(1., 0.))
    second = await closed(store, dt=100, vec=(1., 0.))
    await closed(store, dt=200, vec=(1., 0.))
    result = RetrievalResult('research', QueryPlan('research', intent='recall'),
                             [Hit('episode', first, 'research', 'research', strong=False)])
    await expand_tasks(store, result)
    assert len(result.hits) == 1
    result.hits[0].strong = True
    monkeypatch.setattr(T, 'TASK_RETRIEVAL_SIBLINGS', 1)
    await expand_tasks(store, result)
    assert [h.ref_id for h in result.hits] == [first, second]
    result.hits = result.hits[:1]
    result.plan.interval = (T0, T0 + 50)
    await expand_tasks(store, result)
    assert len(result.hits) == 1
    result.plan.interval = None
    monkeypatch.setattr(T, 'TASK_RETRIEVAL_SIBLINGS', 0)
    await expand_tasks(store, result)
    assert len(result.hits) == 1


async def test_m011_upgrades_populated_v10_store_and_backfills_without_calls(tmp_path, fake):
    path = str(tmp_path / 'v10.db')
    conn = sqlite3.connect(path, isolation_level=None)
    for n, migration in enumerate(db.MIGRATIONS[:10], start=1):
        conn.execute('BEGIN')
        migration(conn)
        conn.execute(f'PRAGMA user_version={n}')
        conn.execute('COMMIT')
    conn.execute("INSERT INTO episodes(id, session_id, started_at, ended_at, title, summary, status, created_at, updated_at) "
                 "VALUES (7, 'old', 1, 2, 'Research', 'Old summary', 'closed', 1, 2)")
    conn.close()
    s = await Store.open(path)
    try:
        assert await s.scalar('PRAGMA user_version') == db.SCHEMA_VERSION
        assert (await s.get_episode(7))['summary'] == 'Old summary'
        before = list(fake.calls), list(fake.embedded)
        assert await s.backfill_tasks() == 1
        assert await s.backfill_tasks() == 0
        assert (await s.list_tasks())[0]['title'] == 'Research'
        assert before == (fake.calls, fake.embedded)
        assert db.migrate(s.conn) == 0
    finally:
        await s.close()


async def test_tasks_cli_lists_threads_without_calls(store, fake, monkeypatch, capsys):
    from contextlib import asynccontextmanager

    from discrete import cli

    @asynccontextmanager
    async def session():
        yield store
    monkeypatch.setattr(cli, '_session', session)
    await closed(store, vec=(1., 0.))
    args = build_parser().parse_args(['tasks', '--json', '-n', '3'])
    before = list(fake.calls), list(fake.embedded)
    assert await cmd_tasks(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output[0]['title'] == 'Agent Memory Repo research'
    assert output[0]['n_episodes'] == 1
    assert before == (fake.calls, fake.embedded)


async def test_idle_close_and_time_cap_still_apply(store, monkeypatch):
    from discrete.process import episodes

    tr = EpisodeTracker(store)
    await tracked(store, tr, vec=(1., 0.))
    monkeypatch.setattr(episodes, 'now', lambda: T0 + T.EPISODE_IDLE_CLOSE_S + 6)
    assert len(await tr.close_idle()) == 1
    assert tr.state is None
    monkeypatch.setattr(T, 'EPISODE_MAX_S', 15.)
    await tracked(store, tr, dt=100, vec=(1., 0.))
    _, ends = await tracked(store, tr, dt=110, vec=(1., 0.))
    assert len(ends) == 1 and tr.state is None


async def test_claim_and_observation_matches_expand_their_task(store):
    from discrete.models import ClaimDraft

    first = await closed(store, vec=(1., 0.))
    second = await closed(store, dt=100, vec=(1., 0.))
    oid = (await store.observations_for_episode(first))[0]['id']
    cid = await store.create_claim(ClaimDraft(kind='event', text='The user researched Agent Memory Repo',
                                             episode_id=first, occurred_from=T0, occurred_to=T0 + 5))
    for kind, rid in [('observation', oid), ('claim', cid)]:
        result = RetrievalResult('research', QueryPlan('research', intent='recall'),
                                 [Hit(kind, rid, 'research', 'research', strong=True)])
        await expand_tasks(store, result)
        assert [h.ref_id for h in result.hits if h.ref_kind == 'episode'] == [second]
        assert not result.hits[-1].strong  # only verification can promote sibling context


async def test_siblings_filtered_before_limit_and_deduplicated(store, monkeypatch):
    first = await closed(store, vec=(1., 0.))
    await closed(store, dt=100, vec=(1., 0.))
    last = await closed(store, dt=200, vec=(1., 0.))
    monkeypatch.setattr(T, 'TASK_RETRIEVAL_SIBLINGS', 1)
    result = RetrievalResult('research', QueryPlan('research', intent='recall', interval=(T0 + 190, T0 + 210)),
                             [Hit('episode', first, 'research', 'research', strong=True)])
    await expand_tasks(store, result)
    assert [h.ref_id for h in result.hits] == [first, last]
    await expand_tasks(store, result)
    assert [h.ref_id for h in result.hits] == [first, last]


async def test_task_assignment_is_atomic_with_draft_commit(store, fake, monkeypatch):
    eid, _ = await ready(store)
    fake.draft = {'title': 'research', 'summary': 'Agent Memory Repo', 'claims': []}
    original = store._assign_episode_task
    def fail(conn, episode_id, **kwargs):
        original(conn, episode_id, **kwargs)
        raise RuntimeError('failure after task assignment')
    monkeypatch.setattr(store, '_assign_episode_task', fail)
    with pytest.raises(RuntimeError, match='failure after task'):
        await ProcessPipeline(store).draft_episode(eid)
    assert (await store.get_episode(eid))['status'] == 'ready'
    assert await store.list_tasks() == []
    assert await store.q('SELECT * FROM episode_tasks') == []
    monkeypatch.setattr(store, '_assign_episode_task', original)
    assert await ProcessPipeline(store).draft_episode(eid)
    assert len(await store.list_tasks()) == 1
    assert fake.calls.count('draft') == 1  # checkpoint survived the failed final transaction


async def test_task_assignment_out_of_order_indexing_and_stale_vectors(store):
    later = await closed(store, dt=100, title='Implementation comparison', summary='Technical evaluation')
    earlier = await closed(store, title='Repo architecture', summary='Design research', vec=(1., 0.))
    assert len(await store.list_tasks()) == 2
    await store.save_embeddings([('episode', later, (1., 0.), 'stale-hash')])
    assert len(await store.list_tasks()) == 2
    await store.save_embeddings([('episode', later, (1., 0.), tk.content_hash('Implementation comparison\nTechnical evaluation'))])
    tasks = await store.list_tasks()
    assert len(tasks) == 1 and tasks[0]['first_seen'] == T0 and tasks[0]['last_seen'] == T0 + 105
    await store.save_embeddings([('episode', later, (1., 0.), tk.content_hash('Implementation comparison\nTechnical evaluation'))])
    assert len(await store.list_tasks()) == 1
    assert (await store.task_siblings([earlier], limit=5))[0]['id'] == later


async def test_merge_and_claim_roll_back_together(store, monkeypatch):
    first, first_ids = await ready(store)
    second, second_ids = await ready(store, dt=20)
    original = store._metric
    def fail(conn, name, delta=1.):
        original(conn, name, delta)
        raise RuntimeError('merge transaction failed')
    monkeypatch.setattr(store, '_metric', fail)
    with pytest.raises(RuntimeError, match='merge transaction failed'):
        await store.begin_draft(first)
    assert (await store.get_episode(first))['status'] == 'ready'
    assert (await store.get_episode(second))['status'] == 'ready'
    assert (await store.get_observation(first_ids[0]))['episode_id'] == first
    assert (await store.get_observation(second_ids[0]))['episode_id'] == second
    monkeypatch.setattr(store, '_metric', original)
    assert await store.begin_draft(first)
    assert not await store.begin_draft(second)


async def test_concurrent_claims_draft_merged_input_only_once(store):
    import asyncio

    first, first_ids = await ready(store)
    second, second_ids = await ready(store, dt=20)
    claims = await asyncio.gather(store.begin_draft(first), store.begin_draft(second))
    assert sum(claims) == 1
    rows = await store.get_observations([*first_ids, *second_ids])
    assert len({r['episode_id'] for r in rows}) == 1
    assert sorted([(await store.get_episode(e))['status'] for e in [first, second]]) == ['drafting', 'merged']
