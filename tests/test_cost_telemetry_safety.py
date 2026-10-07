import asyncio

import pytest

from discrete import llm
from discrete.evaluation import bench
from discrete.perf import PERF, Call


async def test_failed_flush_is_retryable_and_atomic(store, monkeypatch):
    monkeypatch.setattr(llm, '_PENDING_BATCH', None, raising=False)
    llm.METER.drain()
    PERF.drain()
    llm.METER.add('synthetic', calls=1, cost=.25)
    PERF.call(Call('synthetic', 'test', cost=.25))
    original = store.record_telemetry
    async def fail(*args):
        raise RuntimeError('write failed')
    monkeypatch.setattr(store, 'record_telemetry', fail)
    with pytest.raises(RuntimeError):
        await llm.flush_usage(store)
    monkeypatch.setattr(store, 'record_telemetry', original)
    await llm.flush_usage(store)
    assert (await store.usage_by_component())['synthetic']['cost'] == .25
    assert len(await store.perf_calls()) == 1


async def test_cancel_after_commit_does_not_double_count_retry(store, monkeypatch):
    monkeypatch.setattr(llm, '_PENDING_BATCH', None, raising=False)
    llm.METER.drain()
    PERF.drain()
    llm.METER.add('synthetic', calls=1, cost=.25)
    PERF.call(Call('synthetic', 'test', cost=.25))
    original = store.record_telemetry
    async def commit_then_cancel(*args):
        await original(*args)
        raise asyncio.CancelledError
    monkeypatch.setattr(store, 'record_telemetry', commit_then_cancel)
    with pytest.raises(asyncio.CancelledError):
        await llm.flush_usage(store)
    monkeypatch.setattr(store, 'record_telemetry', original)
    await llm.flush_usage(store)
    assert (await store.usage_by_component())['synthetic']['calls'] == 1
    assert len(await store.perf_calls()) == 1


async def test_missing_cost_is_unknown_lower_bound_and_empty_sources_exclude_calls(store):
    await store.record_perf([Call('test', 'test', at=100, cost=None, source='live')], [])
    snap = await bench.collect(store, bench.Window(0, 200, 'synthetic'))
    assert snap['cost']['unknown_calls'] == 1
    assert snap['cost']['is_lower_bound']
    assert 'lower bound' in bench.render(snap)
    assert await store.perf_calls(sources=[]) == []


def test_perf_buffer_eviction_is_reported(monkeypatch):
    from discrete import perf
    monkeypatch.setattr(perf, '_MAX_BUFFERED', 1)
    log = perf.PerfLog()
    log.call(Call('test', 'a'))
    log.call(Call('test', 'b'))
    calls, samples = log.drain()
    assert len(calls) == 1
    assert any(s.stage == 'telemetry.dropped_calls' and s.n == 1 for s in samples)
