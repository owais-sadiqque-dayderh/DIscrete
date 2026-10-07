"""Transient provider failures: fake HTTP only, bounded retries and Bubble isolation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from email.utils import formatdate

import httpx
import openai
import pytest

from discrete import llm
from discrete.config import Endpoint, T
from discrete.perf import PERF

EP = Endpoint("dream", "http://fake/v1", "test", "fake-model")
MESSAGES = [{"role": "user", "content": "x"}]


@dataclass
class FakeClock:
    now: float
    yield_control: Callable[[float], Awaitable[None]]
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        # Let due asyncio timeout callbacks fire, so old budget timeouts fail fast.
        await self.yield_control(0)
        await self.yield_control(0)


@pytest.fixture
async def clock(monkeypatch):
    loop = asyncio.get_running_loop()
    clock = FakeClock(loop.time(), asyncio.sleep)
    monkeypatch.setattr(loop, "time", clock.monotonic)
    monkeypatch.setattr(llm.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm.time, "perf_counter", clock.monotonic)
    monkeypatch.setattr(llm.asyncio, "sleep", clock.sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda low, high: 0.0)
    return clock


@dataclass
class FakeEndpoint:
    statuses: list[int] = field(default_factory=lambda: [503, 200])
    headers: dict[str, str] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)
    message: str = "service_overloaded: The backend is temporarily overloaded. Please retry"

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status != 200:
            return httpx.Response(status, headers=self.headers,
                                  json={"error": {"message": self.message, "type": "provider_error"}})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}],
                                            "model": "fake", "object": "list"})
        if request.url.path.endswith("/rerank"):
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})
        if json.loads(request.content).get("stream"):
            chunk = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                     "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]}
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                                  text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"id": "fake", "object": "chat.completion", "created": 0,
                                        "model": "fake", "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": '{"ok":true}'}}]})

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


@pytest.fixture
async def endpoint(fake, monkeypatch):
    """Exercise the real SDK on a mock transport; no socket or paid provider."""
    server = FakeEndpoint()
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.respond))
    client = openai.AsyncOpenAI(base_url=EP.base_url, api_key="test", http_client=http, max_retries=0)
    backend = llm.OpenAIBackend()
    monkeypatch.setattr(backend, "_client", lambda ep: client)
    backend._http = http
    llm.set_backend(backend)
    monkeypatch.setattr(llm.asyncio, "sleep", server.sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda low, high: high)
    PERF.drain()
    llm.METER.drain()
    try:
        yield server
    finally:
        await client.close()
        PERF.drain()
        llm.METER.drain()


@pytest.mark.parametrize("operation", ["complete", "complete_json", "complete_obj", "embed", "rerank"])
async def test_every_model_api_recovers_from_503(endpoint, operation):
    if operation == "embed":
        assert await llm.embed(["x"]) == [[1.0, 0.0]]
    elif operation == "rerank":
        assert await llm.rerank("x", ["x"], 1) == [(0, 0.9)]
    else:
        result = await getattr(llm, operation)(EP, MESSAGES)
        assert result == ('{"ok":true}' if operation == "complete" else {"ok": True})
    assert len(endpoint.requests) == 2
    assert endpoint.sleeps == [T.LLM_PROVIDER_RETRY_S * 1.25]
    calls, samples = PERF.drain()
    assert [c.status for c in calls] == ["error", "ok"]
    assert sum(s.n for s in samples if s.stage == "health.llm_provider_retry") == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
async def test_retryable_http_statuses(endpoint, status):
    endpoint.statuses = [status, 200]
    assert await llm.complete(EP, MESSAGES)
    assert len(endpoint.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
async def test_validation_auth_and_wrong_model_are_never_retried(endpoint, status):
    # Even an overload word in a 4xx error must not override validation/auth status.
    endpoint.statuses = [status, 200]
    with pytest.raises(openai.APIStatusError):
        await llm.complete_json(EP, MESSAGES)
    assert len(endpoint.requests) == 1
    assert not endpoint.sleeps


async def test_json_mode_validation_error_is_terminal(endpoint):
    endpoint.statuses = [400, 200]
    endpoint.message = "response_format json_object is not supported"
    with pytest.raises(openai.BadRequestError):
        await llm.complete_json(EP, MESSAGES)
    assert len(endpoint.requests) == 1 and not endpoint.sleeps


@pytest.mark.parametrize("header", ["5", "http-date"])
async def test_retry_after_is_a_minimum_wait(endpoint, monkeypatch, header):
    if header == "http-date":
        monkeypatch.setattr(llm.time, "time", lambda: 1_800_000_000.0)
        header = formatdate(1_800_000_005, usegmt=True)
    endpoint.headers = {"Retry-After": header}
    assert await llm.complete(EP, MESSAGES)
    assert endpoint.sleeps == [5.0]


@pytest.mark.parametrize("header", ["bad date", "NaN", "Infinity", "-1"])
async def test_invalid_retry_after_uses_backoff(endpoint, header):
    endpoint.headers = {"Retry-After": header}
    assert await llm.complete(EP, MESSAGES)
    assert endpoint.sleeps == [T.LLM_PROVIDER_RETRY_S * 1.25]


async def test_exhaustion_is_bounded_and_durable_in_metrics_and_health(endpoint, store):
    endpoint.statuses = [503]
    with pytest.raises(openai.APIStatusError):
        await llm.complete(EP, MESSAGES)
    assert len(endpoint.requests) == T.LLM_PROVIDER_RETRIES + 1
    assert endpoint.sleeps == [T.LLM_PROVIDER_RETRY_S * 1.25 * 2 ** i for i in range(T.LLM_PROVIDER_RETRIES)]
    await llm.flush_usage(store)
    stats = await store.stats()
    assert stats["metrics"]["llm.dream.calls"] == T.LLM_PROVIDER_RETRIES + 1
    assert stats["silent_loss"]["llm_provider_retry"] == T.LLM_PROVIDER_RETRIES
    assert stats["silent_loss"]["llm_provider_giveup"] == 1
    samples = await store.perf_samples()
    assert sum(s["n"] for s in samples if s["stage"] == "llm.dream.provider_giveup") == 1


async def test_long_retry_after_gives_up_without_sleep_or_retry(endpoint, monkeypatch):
    monkeypatch.setattr(T, "LLM_RETRY_MAX_S", 2.0)
    endpoint.headers = {"Retry-After": "3"}
    with pytest.raises(openai.APIStatusError):
        await llm.complete(EP, MESSAGES)
    assert len(endpoint.requests) == 1
    assert not endpoint.sleeps


@pytest.mark.parametrize("failure", ["sdk_timeout", "sdk_connection", "http_timeout", "reset", "http_reset", "overload"])
async def test_transient_transport_and_overload_errors_are_retried(fake, monkeypatch, failure):
    request = httpx.Request("POST", "http://fake")
    errors = {
        "sdk_timeout": openai.APITimeoutError(request=request),
        "sdk_connection": openai.APIConnectionError(request=request),
        "http_timeout": httpx.ReadTimeout("timeout", request=request),
        "reset": ConnectionResetError("connection reset by peer"),
        "http_reset": httpx.ReadError("connection reset", request=request),
        "overload": llm.LLMError("service_overloaded"),
    }
    calls = 0

    async def chat(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise errors[failure]
        return "ok"

    monkeypatch.setattr(fake, "chat", chat)
    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 0.0)
    assert await llm.complete(EP, MESSAGES) == "ok"
    assert calls == 2


@pytest.mark.parametrize("interactive", [False, True])
async def test_retry_budget_does_not_cancel_a_healthy_attempt(fake, clock, monkeypatch, interactive):
    monkeypatch.setattr(T, "LLM_RETRY_MAX_S", 90.0)
    monkeypatch.setattr(T, "LLM_INTERACTIVE_RETRY_MAX_S", 8.0)
    duration = 9.0 if interactive else 91.0
    started = 0

    async def chat(ep, messages, **kwargs):
        nonlocal started
        started += 1
        await asyncio.sleep(duration)
        return "ok"

    monkeypatch.setattr(fake, "chat", chat)
    PERF.drain()
    with llm.interactive() if interactive else nullcontext():
        assert await llm.complete(EP, MESSAGES, max_tokens=2500) == "ok"
    assert started == 1 and clock.sleeps == [duration]
    calls, samples = PERF.drain()
    assert calls[-1].status == "ok"
    assert not any(s.stage == "health.llm_provider_giveup" for s in samples)


async def test_bubble_prepare_inherits_short_budget_without_changing_background(endpoint, store, monkeypatch):
    from discrete.bubble import engine

    endpoint.headers = {"Retry-After": "10"}  # Between the default interactive and background budgets.

    async def retrieve(*args, **kwargs):
        # Context must reach child tasks too, as real retrieval plans/embeds concurrently.
        return await asyncio.create_task(llm.complete(EP, MESSAGES))

    monkeypatch.setattr(engine, "retrieve", retrieve)
    with pytest.raises(openai.APIStatusError):
        await engine.BubbleEngine(store).prepare("Which browser did I use?")
    assert len(endpoint.requests) == 1 and not endpoint.sleeps
    # Leaving prepare must restore background context, even on failure.
    endpoint.statuses = [503, 200]
    assert await llm.complete(EP, MESSAGES)
    assert len(endpoint.requests) == 3 and endpoint.sleeps == [10.0]


@pytest.mark.parametrize("streaming", [False, True])
async def test_transient_failure_near_deadline_does_not_sleep_or_retry(fake, clock, monkeypatch, streaming):
    monkeypatch.setattr(T, "LLM_INTERACTIVE_RETRY_MAX_S", 8.0)
    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 1.5)
    calls = 0

    async def chat(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(7.0)
        raise ConnectionResetError("connection reset")

    async def stream(ep, messages, **kwargs):
        yield await chat(ep, messages, **kwargs)

    monkeypatch.setattr(fake, "chat", chat)
    monkeypatch.setattr(fake, "stream", stream, raising=False)
    PERF.drain()
    with llm.interactive(), pytest.raises(ConnectionResetError):
        if streaming:
            _ = [s async for s in llm.stream(EP, MESSAGES)]
        else:
            await llm.complete(EP, MESSAGES)
    assert calls == 1 and clock.sleeps == [7.0]
    _, samples = PERF.drain()
    assert sum(s.n for s in samples if s.stage == "health.llm_provider_giveup") == 1
    assert not any(s.stage == "health.llm_provider_retry" for s in samples)


async def test_sdk_stream_retries_503_before_first_text(endpoint):
    assert [s async for s in llm.stream(EP, MESSAGES)] == ["ok"]
    assert len(endpoint.requests) == 2
    assert endpoint.sleeps == [T.LLM_PROVIDER_RETRY_S * 1.25]


async def test_sdk_stream_uses_short_budget_and_honors_retry_after(endpoint):
    endpoint.headers = {"Retry-After": "10"}
    with pytest.raises(openai.APIStatusError):
        _ = [s async for s in llm.stream(EP, MESSAGES)]
    assert len(endpoint.requests) == 1 and not endpoint.sleeps


@pytest.mark.parametrize("streaming", [True, False])
async def test_stream_and_chat_fallback_retry_before_first_delta(fake, monkeypatch, streaming):
    calls = 0
    request = httpx.Request("POST", "http://fake")
    error = openai.APIStatusError("overloaded", response=httpx.Response(503, request=request), body=None)

    async def chat(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "ok"

    async def stream(ep, messages, **kwargs):
        yield await chat(ep, messages, **kwargs)

    monkeypatch.setattr(fake, "chat", chat)
    if streaming:
        monkeypatch.setattr(fake, "stream", stream, raising=False)
    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 0.0)
    assert [s async for s in llm.stream(EP, MESSAGES)] == ["ok"]
    assert calls == 2


async def test_stream_never_replays_after_text_is_emitted(fake, monkeypatch):
    calls = 0

    async def stream(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        yield "already typed "
        raise ConnectionResetError("connection reset")

    monkeypatch.setattr(fake, "stream", stream, raising=False)
    pieces = []
    with pytest.raises(ConnectionResetError):
        async for piece in llm.stream(EP, MESSAGES):
            pieces.append(piece)
    assert pieces == ["already typed "] and calls == 1


@pytest.mark.parametrize("streaming", [True, False])
async def test_interactive_stream_first_delta_can_arrive_after_retry_budget(fake, clock, monkeypatch, streaming):
    monkeypatch.setattr(T, "LLM_INTERACTIVE_RETRY_MAX_S", 8.0)
    calls = 0
    closed = False

    async def chat(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(9.0)
        return "ok"

    async def stream(ep, messages, **kwargs):
        nonlocal closed
        try:
            yield await chat(ep, messages, **kwargs)
        finally:
            closed = True

    monkeypatch.setattr(fake, "chat", chat)
    if streaming:
        monkeypatch.setattr(fake, "stream", stream, raising=False)
    with llm.interactive():
        assert [s async for s in llm.stream(EP, MESSAGES)] == ["ok"]
    assert calls == 1 and clock.sleeps == [9.0]
    assert closed == streaming


async def test_json_fallback_shares_deadline_without_capping_attempts(fake, clock, monkeypatch):
    monkeypatch.setattr(T, "LLM_RETRY_MAX_S", 90.0)
    calls = 0
    second_budget = 0.0
    policies = []
    original = llm._with_retry

    async def retry(fn, policy, **kwargs):
        policies.append(policy)
        return await original(fn, policy, **kwargs)

    async def chat(ep, messages, **kwargs):
        nonlocal calls, second_budget
        calls += 1
        if calls == 1:
            await asyncio.sleep(89.0)
            return "not JSON"
        second_budget = policies[-1].remaining()
        await asyncio.sleep(2.0)
        return '{"ok":true}'

    monkeypatch.setattr(llm, "_with_retry", retry)
    monkeypatch.setattr(fake, "chat", chat)
    assert await llm.complete_json(EP, MESSAGES) == {"ok": True}
    assert calls == 2 and policies[0] is policies[1]
    assert second_budget == 1.0 and policies[1].remaining() == 0.0


async def test_real_profile_recovers_and_exhausted_dream_step_does_not_block_others(store, fake, monkeypatch):
    from discrete.process import dream
    from tests.test_dream import _football, _memory

    await _football(store)
    fact = await _memory(store, "The user uses Helium as their browser", day=0, episode=50, kind="fact")
    original = fake.chat
    profile_calls = 0
    pattern_calls = 0
    request = httpx.Request("POST", "http://fake")
    error = openai.APIStatusError("service_overloaded", response=httpx.Response(503, request=request), body=None)

    async def chat(ep, messages, **kwargs):
        nonlocal profile_calls, pattern_calls
        prompt = json.dumps(messages)
        if "Write a short profile of the user" in prompt:
            profile_calls += 1
            if profile_calls == 1:
                raise error
        if "Find PATTERNS" in prompt:
            pattern_calls += 1
            raise error
        return await original(ep, messages, **kwargs)

    monkeypatch.setattr(fake, "chat", chat)
    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 0.0)
    fake.profile = {"sections": [{"heading": "Tools", "points": [{"text": "They use Helium as their browser.",
                                                                  "claim_ids": [fact]}]}]}
    report = await dream.run_dream(store)
    assert "error" in report["patterns"]
    assert all("error" not in report[step] for step in ("conflicts", "stale", "profile"))
    assert profile_calls == 2 and pattern_calls == T.LLM_PROVIDER_RETRIES + 1
    assert report["profile"]["points"] == 1
    assert float(await store.get_meta("last_dream", "0")) == 0
    assert float(await store.get_meta("last_dream_attempt", "0")) > 0


async def test_sdk_hidden_retries_are_disabled():
    backend = llm.OpenAIBackend()
    client = backend._client(EP)
    try:
        assert client.max_retries == 0
    finally:
        await client.close()


@pytest.mark.parametrize("failure", ["error", "eof", "partial_error"])
async def test_sdk_stream_protocol_failure_is_transient(endpoint, failure):
    # EOF mid-answer (no finish_reason) is truncation; a finished choice without [DONE] is accepted below.
    chunk = {"choices": [{"index": 0, "delta": {"content": "partial"},
                          "finish_reason": None if failure == "eof" else "stop"}]}
    events = ([chunk] if failure != "error" else [])
    if failure != "eof":
        events.append({"error": {"code": 400, "message": "upstream generation failed"}})
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def respond(request):
        endpoint.requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)

    # Real SSE parsing on mock HTTP; even an error carrying a 400 is a failed stream.
    llm.get_backend()._http._transport = httpx.MockTransport(respond)
    pieces = []
    with pytest.raises(llm.StreamError) as caught:
        async for piece in llm.stream(EP, MESSAGES):
            pieces.append(piece)
    assert llm.provider_glitch(caught.value)
    assert pieces == ([] if failure == "error" else ["partial"])
    expected = T.LLM_PROVIDER_RETRIES + 1 if failure == "error" else 1
    assert len(endpoint.requests) == expected


async def test_sdk_finished_stream_without_done_is_complete(endpoint):
    chunk = {"choices": [{"index": 0, "delta": {"content": "whole answer"}, "finish_reason": "stop"}]}
    body = f"data: {json.dumps(chunk)}\n\n"

    def respond(request):
        endpoint.requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)

    llm.get_backend()._http._transport = httpx.MockTransport(respond)
    assert [p async for p in llm.stream(EP, MESSAGES)] == ["whole answer"]
    assert len(endpoint.requests) == 1


async def test_sdk_reasoning_only_empty_stream_retries_once_with_more_tokens(endpoint, caplog):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            chunks = [{"choices": [{"index": 0, "delta": {"content": None, "reasoning": "thinking"},
                                    "finish_reason": "length"}]}]
        else:
            chunks = [{"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}]
        text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=text)

    llm.get_backend()._http._transport = httpx.MockTransport(respond)
    assert [s async for s in llm.stream(EP, MESSAGES, max_tokens=100)] == ["answer"]
    assert len(payloads) == 2 and payloads[1]["max_tokens"] > payloads[0]["max_tokens"]
    assert "finish_reason=length reasoning_present=True" in caplog.text


async def test_empty_stream_retry_obeys_shared_deadline(fake, clock, monkeypatch):
    monkeypatch.setattr(T, "LLM_INTERACTIVE_RETRY_MAX_S", 8.0)
    calls = 0

    async def stream(ep, messages, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(9.0)
        yield "   "

    monkeypatch.setattr(fake, "stream", stream, raising=False)
    with pytest.raises(llm.EmptyCompletion):
        _ = [s async for s in llm.stream(EP, MESSAGES)]
    assert calls == 1 and clock.sleeps == [9.0]
