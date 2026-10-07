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
    assert evaluate(result).mode == 'empty'
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
