"""Malformed model output cannot change query intent or abort retrieval."""
import pytest

from discrete.config import T
from discrete.models import Hit
from discrete.retrieval.hybrid import _rerank
from discrete.retrieval.query import build_plan
from discrete.retrieval.rerank import llm_rerank


@pytest.mark.parametrize('data', [[], {'order': ['0']}, {'order': [True]}, {'order': [99]},
                                      {'order': [0, 0]}, {'order': [0], 'unusable': '1'}])
async def test_invalid_llm_reranker_degrades_safely(fake, monkeypatch, data):
    async def reply(*args, **kwargs):
        return data
    monkeypatch.setattr(T, 'LLM_RERANK', True)
    monkeypatch.setattr('discrete.retrieval.rerank.complete_json', reply)
    assert await llm_rerank('q', ['a', 'b']) is None


@pytest.mark.parametrize('data', [ {'queries': 'abcd'}, {'queries': [42]}, {'entities': {'x': 1}},
                                        {'intent': ['aggregate']}, {'entity_kind_filter': ['person']}])
async def test_invalid_planner_payload_preserves_structural_plan(fake, monkeypatch, data):
    async def reply(*args, **kwargs):
        return data
    monkeypatch.setattr(T, 'QUERY_LLM_PLAN', True)
    monkeypatch.setattr('discrete.retrieval.query.complete_json', reply)
    plan = await build_plan('What happened with helium?')
    assert plan.intent == 'recall'
    assert plan.queries == ['happened helium']
    assert not plan.entities


@pytest.mark.parametrize('order', [[(True, 1.0)], [(0, float('nan'))], [(99, 1)], [(0, 1), (0, 0.5)], ['bad']])
async def test_native_reranker_invalid_indices_and_scores_keep_pool(fake, monkeypatch, order):
    async def reply(*args):
        return order
    monkeypatch.setattr('discrete.retrieval.hybrid.rerank', reply)
    pool = [Hit('observation', i, str(i), 'text') for i in range(4)]
    assert await _rerank('q', pool, False) == pool
