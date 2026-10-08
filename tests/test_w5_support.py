"""Support completion is separate from search relevance."""
import asyncio

import pytest

from discrete.models import Hit, QueryPlan, RetrievalResult
from discrete.retrieval.grounding import evaluate
from discrete.retrieval.support import verify_support


@pytest.mark.parametrize('intent', ['recall', 'enumerate', 'temporal'])
async def test_timeout_never_authorizes_grounded_output(fake, monkeypatch, intent):
    from discrete.config import T
    from discrete.retrieval import support

    async def slow(*args):
        await asyncio.sleep(10)

    monkeypatch.setattr(T, 'SUPPORT_TIMEOUT_S', 0.001)
    monkeypatch.setattr(support, '_supported', slow)
    hit = Hit('observation', 1, 'Report', 'Relevant but unchecked', strong=True)
    result = RetrievalResult('q', QueryPlan('q', intent=intent), [hit], exhaustive=True, digest='Unchecked digest')
    await verify_support(result)
    assert evaluate(result).mode == 'unverified'
    assert not hit.eligible
    from discrete.bubble.engine import render_records
    assert 'not verified' in render_records(result)
    assert 'Relevant but unchecked' in render_records(result)
    assert hit.verification == 'timed_out'
    assert result.digest == ''


async def test_verified_state_is_independent_of_relevance(fake):
    result = RetrievalResult('q', QueryPlan('q'), [Hit('observation', 1, 'Report', 'Details')])
    await verify_support(result)
    assert result.hits[0].verification == 'verified'
    assert evaluate(result).mode == 'grounded'


async def test_why_timeout_is_explicitly_hedged(fake, monkeypatch):
    from discrete.bubble.engine import render_records
    from discrete.config import T
    from discrete.retrieval import support

    async def slow(*args):
        await asyncio.sleep(10)

    monkeypatch.setattr(T, 'SUPPORT_TIMEOUT_WHY_S', 0.001)
    monkeypatch.setattr(support, '_supported', slow)
    result = RetrievalResult('why', QueryPlan('why', intent='why'),
                             [Hit('observation', 1, 'Page', 'Context', strong=True, via='before claim 2')])
    await verify_support(result)
    assert evaluate(result).mode == 'why'
    assert 'timed_out' in render_records(result)


@pytest.mark.parametrize('intent', ['continuation', 'world'])
async def test_unchecked_intents_keep_their_records(fake, intent):
    from discrete.bubble.engine import render_records

    result = RetrievalResult('q', QueryPlan('q', intent=intent), [Hit('observation', 1, 'Report', 'Context line')])
    await verify_support(result)
    assert result.hits[0].verification == 'unchecked'
    assert 'Context line' in render_records(result)


async def test_disabled_check_is_unchecked_not_empty(fake, monkeypatch):
    from discrete.config import T

    monkeypatch.setattr(T, 'SUPPORT_CHECK', False)
    result = RetrievalResult('q', QueryPlan('q'), [Hit('observation', 1, 'Report', 'Details', strong=True)],
                             digest='Day digest')
    await verify_support(result)
    assert evaluate(result).mode == 'grounded'
    assert result.digest_verification == 'unchecked'


async def test_one_failed_batch_keeps_the_others(fake, monkeypatch):
    from discrete.retrieval import support

    async def flaky(result, records):
        if records[0]['text'].startswith('bad'):
            raise ValueError('synthetic malformed batch')
        return set(range(len(records)))

    monkeypatch.setattr(support, '_supported', flaky)
    hits = [Hit('observation', i, 'good' if i < 16 else 'bad', 'x') for i in range(20)]
    result = RetrievalResult('q', QueryPlan('q'), hits)
    await verify_support(result)
    assert [h.ref_id for h in result.hits] == list(range(16))
    assert all(h.verification == 'unverified' for h in hits[16:])


async def test_timeout_writer_discloses_and_hedges_without_rejected_evidence(store, fake):
    from discrete.bubble.engine import BubbleEngine, Prepared, render_records

    timed_out = Hit('observation', 1, 'Report', 'A visible report', strong=True, verification='timed_out')
    rejected = Hit('observation', 2, 'Rejected', 'Unsupported ownership', verification='rejected')
    result = RetrievalResult('q', QueryPlan('q'), [timed_out, rejected])
    p = Prepared(field_text='q?', query='q', app='Notes', field_desc='body', result=result, decision=evaluate(result))
    engine = BubbleEngine(store)
    assert p.mode == 'unverified'
    prompt = engine.messages(p)[-1]['content']
    assert 'NOT verified' in prompt and "'it looks like'" in prompt
    assert 'Unsupported ownership' not in render_records(result)
    fake.writer_reply = 'It looks like you viewed a report.'
    chunks = []
    answer = await engine.write(p, lambda s: chunks.append(s) or True)
    assert answer.lstrip().startswith('Support check timed out; this answer is not verified.')
    assert 'viewed a report' in answer
    assert ''.join(chunks) == answer
