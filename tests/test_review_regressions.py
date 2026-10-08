"""Offline reproductions of the memory-evolution review findings."""

from contextlib import contextmanager
from pathlib import Path

import pytest

from bench.redraft import redraft
from discrete import textkit as tk
from discrete.config import T
from discrete.evaluation.invariants import check_store
from discrete.llm import LLMError
from discrete.models import ClaimDraft, Hit, QueryPlan, RetrievalResult
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval.hybrid import expand_tasks
from discrete.store import Store
from tests.test_pipeline import obs
from tests.test_tasks import closed, ready, tracked


async def test_redraft_preserves_forget_correct_and_drafting_suppression(store, tmp_db, tmp_path, fake):
    ep, ids = await ready(store)
    forgotten_text = 'The user likes blue widgets'
    await store.execute('UPDATE observations SET screen_text=? WHERE id=?', (forgotten_text, ids[0]))
    forgotten = await store.create_claim(ClaimDraft(kind='fact', text=forgotten_text, episode_id=ep))
    forget_event = await store.forget_claim(forgotten)
    old = await store.create_claim(ClaimDraft(kind='preference', text='The user prefers red widgets', episode_id=ep))
    corrected, correct_event = await store.correct_claim(old, 'The user prefers green widgets')
    events = [tuple(await store.get_event(e)) for e in (forget_event, correct_event)]
    output = tmp_path / 'redraft.db'
    redraft(Path(tmp_db), output)
    copied = await Store.open(str(output))
    try:
        assert await copied.suppressed('fact', forgotten_text)
        assert [tuple(await copied.get_event(e)) for e in (forget_event, correct_event)] == events
        claim = await copied.get_claim(corrected)
        assert claim['modality'] == 'stated' and claim['status'] == 'live'
        assert await copied.q('PRAGMA foreign_key_check') == []
        fake.draft = {'title': 'widgets', 'summary': 'widgets', 'claims': [
            {'kind': 'fact', 'text': forgotten_text,
             'evidence': [{'observation_id': ids[0], 'quote': forgotten_text}]}]}
        result = await ProcessPipeline(copied).draft_episode(ep)
        assert result.rejected['forgotten'] == 1
        assert not [c for c in await copied.live_claims() if c['text'] == forgotten_text]
        assert (await copied.get_claim(corrected))['status'] == 'live'
        # The preserved directives still have the inputs required for undo.
        await copied.revert_event(correct_event)
        assert (await copied.get_claim(old))['status'] == 'live'
        await copied.revert_event(forget_event)
        assert not await copied.suppressed('fact', forgotten_text)
    finally:
        await copied.close()


async def test_redraft_refuses_unsupported_user_directives(store, tmp_db, tmp_path):
    cid = await store.create_claim(ClaimDraft(kind='fact', text='The user likes blue widgets'))
    event = await store.forget_claim(cid)
    await store.revert_event(event)
    output = tmp_path / 'redraft.db'
    with pytest.raises(SystemExit, match='user.*revert'):
        redraft(Path(tmp_db), output)
    assert not output.exists()


async def test_live_split_waits_for_open_successor_then_merges(store, fake):
    tracker = EpisodeTracker(store)
    for dt, app in [(0, 'Arc'), (10, 'Terminal'), (110, 'Finder')]:
        oid, ends = await tracked(store, tracker, dt=dt, app=app, vec=(1., 0.))
        await store.save_embeddings([('observation', oid, [1., 0.], 'screen')])
    first = ends[0]
    successor = tracker.state.episode_id
    assert not await store.begin_draft(first)
    assert await store.episodes_ready_for_draft() == []
    await tracker.close_current()
    assert await store.begin_draft(first)
    assert (await store.get_episode(successor))['status'] == 'merged'


async def test_split_keeps_first_moved_context(store):
    tracker = EpisodeTracker(store)
    for dt, app in [(0, 'A'), (10, 'B'), (110, 'C')]:
        await tracked(store, tracker, dt=dt, app=app, vec=(1., 0.))
    successor = tracker.state.episode_id
    assert tracker.state.pair == ('B', 'cognition.com')
    for dt in (120, 220):
        _, ends = await tracked(store, tracker, dt=dt, app='B', vec=(1., 0.))
        assert not ends
        assert tracker.state.episode_id == successor


@pytest.mark.parametrize('arrival', [(0, 1, 2), (2, 0, 1)])
async def test_delayed_embeddings_reconcile_entire_established_task_groups(store, arrival):
    entries = [('Architecture survey', 'Compared design choices'),
               ('Implementation comparison', 'Technical evaluation'),
               ('Implementation comparison', 'Technical evaluation')]
    ids = [await closed(store, dt=i * 100, title=title, summary=summary)
           for i, (title, summary) in enumerate(entries)]
    assert sorted(t['n_episodes'] for t in await store.list_tasks()) == [1, 2]
    # Arrival order must not leave one member stranded in its old group.
    for i in arrival:
        title, summary = entries[i]
        await store.save_embeddings([('episode', ids[i], [1., 0.], tk.content_hash(f'{title}\n{summary}'))])
    tasks = await store.list_tasks()
    assert len(tasks) == 1 and tasks[0]['n_episodes'] == 3
    assert tasks[0]['first_seen'] == (await store.get_episode(ids[0]))['started_at']
    assert tasks[0]['last_seen'] == (await store.get_episode(ids[2]))['ended_at']


async def test_duplicate_metadata_preserves_draft_checkpoint(store, fake, monkeypatch):
    ep, ids = await ready(store)
    fake.draft = {'title': 'research', 'summary': 'research', 'claims': []}
    original = store.commit_episode_draft
    async def fail(*args, **kwargs):
        raise LLMError('temporary final failure')
    monkeypatch.setattr(store, 'commit_episode_draft', fail)
    pipe = ProcessPipeline(store)
    assert await pipe.draft_episode(ep) is None
    assert fake.calls.count('draft') == 1
    # Same capture with the same interval changes dup_count/updated_at only.
    row = await store.get_observation(ids[0])
    duplicate = obs(title=row['window_title'], text=row['screen_text'], t=row['started_at'],
                    entities=tk.parse_json_list(row['entities']), clicks=0)
    assert await store.add_observation(duplicate) == (ids[0], False)
    monkeypatch.setattr(store, 'commit_episode_draft', original)
    assert await pipe.draft_episode(ep)
    assert fake.calls.count('draft') == 1


async def test_grounded_evidence_checks_every_quote(store, tmp_db):
    ids = []
    for dt, text in [(0, 'authentic text'), (10, 'another screen')]:
        oid, _ = await store.add_observation(obs(text=text, t=1000 + dt))
        ids.append(oid)
    cid = await store.create_claim(ClaimDraft(kind='fact', text='authentic text', evidence=[
        {'observation_id': ids[0], 'quote': 'authentic text'},
        {'observation_id': ids[1], 'quote': 'fabricated quote'}]))
    result = next(c for c in check_store(tmp_db).checks if c.name == 'grounded_evidence')
    assert not result.passed
    assert result.details['claim_ids'] == [cid]
    assert result.details['violations'] == [{'claim_id': cid, 'observation_id': ids[1]}]


async def test_redraft_rebuilds_candidates_from_captures(store, tmp_db, tmp_path):
    ep, ids = await ready(store, entities=('Capture Entity',))
    await store.begin_draft(ep)
    await store.commit_episode_draft(ep, title='old', summary='old', apps=[], sites=[], engaged_fraction=1,
                                     writes=[], candidates=['Old Draft Entity'], rejected={})
    output = tmp_path / 'redraft.db'
    redraft(Path(tmp_db), output)
    copied = await Store.open(str(output))
    try:
        candidates = {r['name']: tk.parse_json_list(r['episodes']) for r in await copied.entity_candidates()}
        assert 'Old Draft Entity' not in candidates
        assert candidates['Capture Entity'] == [ep]
        assert candidates['Code'] == [ep]  # engaged capture app
        assert (await copied.get_observation(ids[0]))['entities'] == '["Capture Entity"]'
    finally:
        await copied.close()


async def test_sibling_limit_excludes_existing_hits_and_empty_summaries(store, monkeypatch):
    first = await closed(store)
    occupied = await closed(store, dt=100)
    empty = await closed(store, dt=200)
    last = await closed(store, dt=300)
    await store.execute("UPDATE episodes SET summary='' WHERE id=?", (empty,))
    monkeypatch.setattr(T, 'TASK_RETRIEVAL_SIBLINGS', 1)
    result = RetrievalResult('research', QueryPlan('research', intent='recall'), [
        Hit('episode', first, 'research', 'research', strong=True),
        Hit('episode', occupied, 'research', 'research', strong=False)])
    await expand_tasks(store, result)
    assert [h.ref_id for h in result.hits] == [first, occupied, last]


async def test_backfill_uses_bounded_transactions_with_identical_groups(store, monkeypatch):
    ids = [await closed(store, dt=i * 100) for i in range(7)]
    expected = [[r['episode_id'] for r in await store.q(
        'SELECT episode_id FROM episode_tasks WHERE task_id=? ORDER BY episode_id', (t['id'],))]
                for t in await store.list_tasks()]
    await store.execute('DELETE FROM episode_tasks')
    await store.execute('DELETE FROM tasks')
    sizes = []
    original = store._tx
    original_assign = store._assign_episode_task
    assigned = 0
    @contextmanager
    def tx():
        nonlocal assigned
        assigned = 0
        with original() as conn:
            yield conn
        sizes.append(assigned)
    def assign(conn, eid, **kwargs):
        nonlocal assigned
        assigned += 1
        return original_assign(conn, eid, **kwargs)
    monkeypatch.setattr(store, '_tx', tx)
    monkeypatch.setattr(store, '_assign_episode_task', assign)
    from discrete.store import tasks
    monkeypatch.setattr(tasks, 'TASK_BACKFILL_BATCH_SIZE', 3, raising=False)
    assert await store.backfill_tasks() == len(ids)
    assert max(sizes) <= 3
    assert len([n for n in sizes if n]) == 3
    actual = [[r['episode_id'] for r in await store.q(
        'SELECT episode_id FROM episode_tasks WHERE task_id=? ORDER BY episode_id', (t['id'],))]
              for t in await store.list_tasks()]
    assert actual == expected


async def test_open_successor_hold_expires_without_blocking_other_ready_episodes(store, monkeypatch):
    from discrete.store import episodes

    first, _ = await ready(store)
    successor = await store.open_episode('s1', (await store.get_episode(first))['ended_at'] + 5)
    other, _ = await ready(store, dt=1000, session='other')
    since = (await store.get_episode(first))['ready_since']
    monkeypatch.setattr(episodes, 'now', lambda: since + episodes.EPISODE_DRAFT_GRACE_S - 0.1)
    assert [r['id'] for r in await store.episodes_ready_for_draft(limit=1)] == [other]
    assert not await store.begin_draft(first)
    monkeypatch.setattr(episodes, 'now', lambda: since + episodes.EPISODE_DRAFT_GRACE_S)
    assert [r['id'] for r in await store.episodes_ready_for_draft(limit=1)] == [first]
    assert await store.begin_draft(first)
    assert (await store.get_episode(successor))['status'] == 'open'


async def test_merging_ready_neighbors_does_not_restart_grace(store, monkeypatch):
    from discrete.store import episodes

    first, _ = await ready(store)
    await ready(store, dt=20)
    await store.open_episode('s1', (await store.get_episode(first))['ended_at'] + 40)
    since = (await store.get_episode(first))['ready_since']
    monkeypatch.setattr(episodes, 'now', lambda: since + episodes.EPISODE_DRAFT_GRACE_S)
    assert await store.begin_draft(first)


@pytest.mark.parametrize(('first_since', 'other_since', 'expected'), [
    (100., 200., 100.), (None, 200., 200.), (100., None, 100.), (None, None, 70.),
])
async def test_merge_preserves_ready_timer_and_legacy_fallback(store, first_since, other_since, expected):
    first, _ = await ready(store)
    other, _ = await ready(store, dt=20)
    await store.execute("UPDATE episodes SET ready_since=?,updated_at=70 WHERE id=?", (first_since, first))
    await store.execute("UPDATE episodes SET ready_since=?,updated_at=80 WHERE id=?", (other_since, other))
    await store.mark_episode_ready(first)  # Already ready: don't restart the timer.
    assert (await store.get_episode(first))['ready_since'] == first_since
    assert await store.begin_draft(first)
    assert (await store.get_episode(first))['ready_since'] == expected
    assert (await store.get_episode(other))['status'] == 'merged'
    assert await store.q("SELECT key FROM meta WHERE key GLOB 'episode_ready_since:*' "
                         "OR key GLOB 'episode_boundary_locked:*'") == []


@pytest.mark.parametrize('difference', ['entities', 'window'])
async def test_established_task_reconciliation_keeps_entity_and_time_guards(store, difference):
    first = await closed(store, title='Architecture survey', summary='Compared design choices', vec=(1., 0.))
    dt = T.TASK_WINDOW_S + 100 if difference == 'window' else 100
    entities = ('Other Project',) if difference == 'entities' else ('Agent Memory Repo',)
    entries = [await closed(store, dt=dt + i * 100, entities=entities,
                            title='Implementation comparison', summary='Technical evaluation') for i in range(2)]
    for eid in entries:
        await store.save_embeddings([('episode', eid, [1., 0.],
                                      tk.content_hash('Implementation comparison\nTechnical evaluation'))])
    assert sorted(t['n_episodes'] for t in await store.list_tasks()) == [1, 2]
    assert await store.task_siblings([first], limit=5) == []


async def test_engaged_screen_cannot_invalidate_stated_user_correction(store, fake):
    old = await store.create_claim(ClaimDraft(kind='preference', text='The user prefers blue widgets'))
    corrected, _ = await store.correct_claim(old, 'The user prefers green widgets')
    ep, ids = await ready(store)
    text = 'The user prefers blue widgets'
    await store.execute('UPDATE observations SET screen_text=? WHERE id=?', (text, ids[0]))
    fake.draft = {'title': 'widgets', 'summary': 'widgets', 'claims': [
        {'kind': 'preference', 'text': text, 'evidence': [{'observation_id': ids[0], 'quote': text}]}]}
    fake.relations[corrected] = 'contradicts'
    assert await ProcessPipeline(store).draft_episode(ep)
    assert (await store.get_claim(corrected))['status'] == 'live'
    assert (await store.get_claim(corrected))['modality'] == 'stated'


async def test_task_candidate_lookup_filters_unshared_entities_before_materializing(store, monkeypatch):
    first = await closed(store, entities=('Agent Memory Repo',))
    await closed(store, dt=100, entities=('Other Project',))
    await closed(store, dt=200, entities=('Another Project',))
    seen = []
    original = store._assign_episode_task
    class CandidateConnection:
        def __init__(self, conn):
            self.conn = conn
        def execute(self, sql, args=()):
            cursor = self.conn.execute(sql, args)
            if 'AS task_entities' not in sql:
                return cursor
            rows = cursor.fetchall()
            seen.append([r['id'] for r in rows])
            class Rows:
                def fetchall(self):
                    return rows
            return Rows()
    def assign(conn, eid, **kwargs):
        return original(CandidateConnection(conn), eid, **kwargs)
    monkeypatch.setattr(store, '_assign_episode_task', assign)
    await closed(store, dt=300, entities=('Agent Memory Repo',))
    assert seen == [[first]]
