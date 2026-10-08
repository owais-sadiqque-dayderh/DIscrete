"""API-only model access (plans Rev 2): chat, vision, embeddings, rerank.

A pluggable backend lets tests run without a network. Callers catch
EXPECTED_ERRORS only; anything else propagates (insights.md §7.2).

Every call is metered per component (the endpoint name: caption, insight,
embedding, …): running totals in `METER` (→ the metrics table, `discrete stats`)
and one `perf.Call` per call with its latency, tokens and provider-reported cost
(→ perf_calls, `discrete bench`). `flush_usage` writes both.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
import logging
import math
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx
import openai
from openai import AsyncOpenAI

from .config import Endpoint, T, embedding_endpoint, rerank_endpoint
from .perf import PERF, Call

log = logging.getLogger("discrete.llm")


class LLMError(RuntimeError):
    """A model call returned nothing usable."""


class StreamError(LLMError):
    """An interrupted or explicitly failed provider stream; safe to retry before text."""


class EmptyCompletion(LLMError):
    """A finished writer attempt with no visible answer."""

    def __init__(self, finish_reason: str | None = None, reasoning_present: bool = False):
        self.finish_reason = finish_reason
        self.reasoning_present = reasoning_present
        super().__init__(f"empty completion (finish_reason={finish_reason}, reasoning_present={reasoning_present})")


class NotConfigured(LLMError):
    pass


class TokenLimit(LLMError):
    """The model spent its whole max_tokens before writing an answer: typically a
    reasoning model thinking. Retrying the same request cannot help."""

    def __init__(self, ep: Endpoint, max_tokens: int | None):
        p = ep.prefix or "CORE_LM"
        super().__init__(
            f"{ep.name}: {ep.model} used all {max_tokens} tokens before answering (a reasoning model thinking?). "
            f"In .env set {p}_REASONING_EFFORT=minimal and/or raise {p}_EXTRA_TOKENS (now {ep.extra_tokens}).")


EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    LLMError,
    openai.APIError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    ConnectionError,
)


# The call in flight in this task, so usage a backend reports lands on its record.
_CURRENT: contextvars.ContextVar[Call | None] = contextvars.ContextVar("discrete_llm_call", default=None)
_INTERACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("discrete_llm_interactive", default=False)
_Result = TypeVar("_Result")


@contextlib.contextmanager
def interactive() -> Iterator[None]:
    """Use Bubble's short retry budget, inherited by its concurrent retrieval tasks."""
    token = _INTERACTIVE.set(True)
    try:
        yield
    finally:
        _INTERACTIVE.reset(token)


class UsageMeter:
    """In-memory per-component running totals, drained into the store."""

    FIELDS = ("calls", "prompt_tokens", "completion_tokens", "cost", "unknown_cost_calls")

    def __init__(self) -> None:
        self._data: dict[str, dict[str, float]] = {}

    def add(self, component: str, *, calls: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0,
            cost: float = 0.0) -> None:
        d = self._data.setdefault(component, dict.fromkeys(self.FIELDS, 0))
        d["calls"] += calls
        d["prompt_tokens"] += prompt_tokens or 0
        d["completion_tokens"] += completion_tokens or 0
        d["cost"] += cost or 0.0

    def add_usage(self, component: str, usage: Any) -> None:
        """Record an OpenAI-style `usage` object or dict, if the provider sent one.
        OpenRouter adds `cost` (USD) to it, for chat, embeddings and rerank alike."""
        if usage is None:
            return
        get = usage.get if isinstance(usage, dict) else (lambda k, d=None: getattr(usage, k, d))
        prompt, completion = int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)
        cost = get("cost", None)
        cost = float(cost) if isinstance(cost, int | float) else None
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            cost = None
        self.add(component, prompt_tokens=prompt, completion_tokens=completion, cost=cost or 0.0)
        rec = _CURRENT.get()
        if rec is not None and rec.component == component:
            rec.prompt_tokens += prompt
            rec.completion_tokens += completion
            if cost is not None:
                rec.cost = (rec.cost or 0.0) + cost

    def drain(self) -> dict[str, dict[str, float]]:
        out, self._data = self._data, {}
        return out

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._data.items()}


METER = UsageMeter()


_PENDING_BATCH = None
_FLUSH_LOCK = asyncio.Lock()


async def flush_usage(store: Any) -> None:
    """Atomically persist usage/perf; retain one idempotent batch until acknowledged.

    A cancellation can arrive after SQL committed. Retrying the same batch id
    avoids both losing that batch and counting it twice.
    """
    import uuid

    global _PENDING_BATCH
    async with _FLUSH_LOCK:
        for _ in range(2):  # a retained batch, then newly buffered measurements
            if _PENDING_BATCH is None:
                usage = METER.drain()
                calls, samples = PERF.drain()
                if not (usage or calls or samples):
                    return
                _PENDING_BATCH = (uuid.uuid4().hex, usage, calls, samples)
            await store.record_telemetry(*_PENDING_BATCH)
            _PENDING_BATCH = None


def _status(exc: BaseException) -> str:
    return "cancelled" if isinstance(exc, asyncio.CancelledError | GeneratorExit) else "error"


@contextlib.contextmanager
def _metered(ep: Endpoint) -> Iterator[Call]:
    """Count one call to `ep` and time it; usage reported inside lands on its record."""
    METER.add(ep.name, calls=1)
    rec = Call(ep.name, ep.model or "", at=time.time())
    token = _CURRENT.set(rec)
    t0 = time.perf_counter()
    try:
        yield rec
    except BaseException as exc:
        rec.status = _status(exc)
        raise
    finally:
        rec.ms = (time.perf_counter() - t0) * 1000
        _CURRENT.reset(token)
        if rec.cost is None:
            d = METER._data.setdefault(ep.name, dict.fromkeys(METER.FIELDS, 0))
            d["unknown_cost_calls"] += 1
        PERF.call(rec)


class OpenAIBackend:
    """Talks to any OpenAI-compatible API (OpenRouter by default)."""

    def __init__(self) -> None:
        self._clients: dict[tuple[str | None, str | None], AsyncOpenAI] = {}
        self._http: httpx.AsyncClient | None = None  # kept alive: one TLS handshake, not one per rerank

    def _client(self, ep: Endpoint) -> AsyncOpenAI:
        key = (ep.base_url, ep.api_key)
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                base_url=ep.base_url, api_key=ep.api_key, timeout=T.LLM_TIMEOUT_S, max_retries=0
            )
        return self._clients[key]

    @staticmethod
    def _extras(ep: Endpoint) -> dict[str, Any]:
        """Per-model settings from .env: reasoning effort, EXTRA_BODY, timeout."""
        out: dict[str, Any] = {}
        if body := ep.body_extras():
            out["extra_body"] = body
        if ep.timeout_s is not None:
            out["timeout"] = ep.timeout_s
        return out

    async def chat(self, ep: Endpoint, messages: list[dict], *, temperature: float | None,
                   max_tokens: int | None, json_mode: bool) -> str:
        payload: dict[str, Any] = {"model": ep.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        rsp = await self._client(ep).chat.completions.create(**payload, **self._extras(ep))
        METER.add_usage(ep.name, getattr(rsp, "usage", None))
        if not rsp.choices:
            raise LLMError(f"{ep.name}: empty choices")
        choice = rsp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and choice.finish_reason == "length":
            raise TokenLimit(ep, max_tokens)
        return text

    async def stream(self, ep: Endpoint, messages: list[dict], *, temperature: float | None,
                     max_tokens: int | None):
        payload: dict[str, Any] = {"model": ep.model, "messages": messages, "stream": True,
                                   "stream_options": {"include_usage": True}}  # else a stream reports no tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Read SSE explicitly: the SDK accepts EOF without [DONE], hiding truncation.
        wrote, reasoning_present, finish_reason = False, False, None
        async with self._client(ep).chat.completions.with_streaming_response.create(
                **payload, **self._extras(ep)) as rsp:
            data_lines: list[str] = []
            async for line in rsp.iter_lines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                    continue
                if line or not data_lines:
                    continue
                data = "\n".join(data_lines)
                data_lines.clear()
                if data.strip() == "[DONE]":
                    if not wrote:
                        raise EmptyCompletion(finish_reason, reasoning_present)
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise StreamError(f"{ep.name}: invalid stream chunk") from exc
                if not isinstance(chunk, dict):
                    raise StreamError(f"{ep.name}: invalid stream chunk")
                if chunk.get("error"):
                    raise StreamError(f"{ep.name}: provider stream error: {chunk['error']}")
                usage = chunk.get("usage")
                METER.add_usage(ep.name, usage)
                if isinstance(usage, dict):
                    details = usage.get("completion_tokens_details") or {}
                    reasoning_present |= bool(details.get("reasoning_tokens"))
                for choice in chunk.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        continue
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    reasoning_present |= any(bool(delta.get(k)) for k in
                                             ("reasoning", "reasoning_content", "reasoning_details"))
                    content = delta.get("content")
                    if content:
                        wrote |= bool(content.strip())
                        yield content
            # Some compatible servers omit [DONE]; a finished choice with text is complete.
            if wrote and finish_reason in ("stop", "length"):
                return
            raise StreamError(f"{ep.name}: stream ended without [DONE]")

    async def embed(self, ep: Endpoint, texts: list[str]) -> list[list[float]]:
        rsp = await self._client(ep).embeddings.create(model=ep.model or "", input=texts, **self._extras(ep))
        METER.add_usage(ep.name, getattr(rsp, "usage", None))
        return [d.embedding for d in rsp.data]

    async def rerank(self, ep: Endpoint, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        url = (ep.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/rerank"
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=T.LLM_TIMEOUT_S)
        r = await self._http.post(
            url,
            headers={"Authorization": f"Bearer {ep.api_key}"},
            json={"model": ep.model, "query": query, "documents": docs, "top_n": top_n, **ep.body_extras()},
            timeout=ep.timeout_s or T.LLM_TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
        METER.add_usage(ep.name, data.get("usage"))
        return [(int(x["index"]), float(x["relevance_score"])) for x in data.get("results", [])]


_backend: Any = OpenAIBackend()


def set_backend(backend: Any) -> None:
    global _backend
    _backend = backend


def get_backend() -> Any:
    return _backend


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(raw: str) -> Any:
    """Tolerant JSON extraction; None when nothing parses."""
    if not raw:
        return None
    txt = raw.strip()
    m = _JSON_BLOCK.search(txt)
    if m:
        txt = m.group(1).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = txt.find(opener), txt.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(txt[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _tuned(ep: Endpoint, temperature: float | None, max_tokens: int | None) -> tuple[float | None, int | None]:
    """The caller's temperature and max_tokens, adjusted by the model's .env settings."""
    if ep.no_temperature:
        temperature = None
    elif ep.temperature is not None:
        temperature = ep.temperature
    if max_tokens is not None:
        max_tokens += ep.extra_tokens
    return temperature, max_tokens


def provider_glitch(exc: BaseException) -> bool:
    """Retry provider/transport failures, never validation or authentication errors.

    Preserve the router-specific upstream 404 exception from 5e7d73a; a wrong
    model name is still terminal. Status takes precedence over overload wording.
    """
    if isinstance(exc, StreamError | EmptyCompletion):
        return True
    if isinstance(exc, openai.APIStatusError | httpx.HTTPStatusError):
        status = exc.status_code if isinstance(exc, openai.APIStatusError) else exc.response.status_code
        if status in (429, 500, 502, 503, 504, 529):
            return True
        text = str(exc)
        return status == 404 and (
            "Provider returned error" in text or ("model_not_found" in text and "provider_name" in text))
    if isinstance(exc, openai.APIConnectionError | httpx.TimeoutException | httpx.NetworkError
                  | httpx.RemoteProtocolError | TimeoutError | ConnectionError):
        return True
    return isinstance(exc, openai.APIError | LLMError) and "overload" in str(exc).lower()


def _retry_after(exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    value = response.headers.get("Retry-After") if response is not None else None
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


@dataclass
class _RetryPolicy:
    """Gate additional attempts; in-flight requests keep their transport timeout."""

    ep: Endpoint
    interactive: bool = field(default_factory=_INTERACTIVE.get)
    attempts: int = 0
    deadline: float = field(init=False)

    def __post_init__(self) -> None:
        budget = T.LLM_INTERACTIVE_RETRY_MAX_S if self.interactive else T.LLM_RETRY_MAX_S
        self.deadline = time.monotonic() + max(0.0, budget)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def giveup(self) -> None:
        PERF.count(f"llm.{self.ep.name}.provider_giveup")
        PERF.count("health.llm_provider_giveup")

    async def retry(self, exc: BaseException, *, replayable: bool = True) -> bool:
        if not provider_glitch(exc):
            return False
        base = max(0.0, T.LLM_PROVIDER_RETRY_S) * (2 ** self.attempts)
        delay = max(_retry_after(exc), base + random.uniform(0.0, base * 0.25))
        if not replayable or self.attempts >= T.LLM_PROVIDER_RETRIES or delay >= self.remaining():
            self.giveup()
            return False
        log.info("%s: transient provider failure, retry %d in %.2fs", self.ep.name, self.attempts + 1, delay)
        await asyncio.sleep(delay)
        # A suspended process may wake after the deadline: never issue another request then.
        if self.remaining() <= 0:
            self.giveup()
            return False
        self.attempts += 1
        PERF.count(f"llm.{self.ep.name}.provider_retry")
        PERF.count("health.llm_provider_retry")
        return True


async def _with_retry(fn: Callable[[], Awaitable[_Result]], policy: _RetryPolicy, *, metered: bool = True) -> _Result:
    while True:
        try:
            with _metered(policy.ep) if metered else contextlib.nullcontext():
                # The budget limits retry starts, not healthy (possibly slow) attempts.
                return await fn()
        except EXPECTED_ERRORS as exc:
            if not await policy.retry(exc):
                raise


async def complete(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.0,
                   max_tokens: int | None = None, json_mode: bool = False) -> str:
    return await _complete(ep, messages, policy=_RetryPolicy(ep), temperature=temperature,
                           max_tokens=max_tokens, json_mode=json_mode)


async def _complete(ep: Endpoint, messages: list[dict], *, policy: _RetryPolicy,
                    temperature: float | None, max_tokens: int | None, json_mode: bool = False) -> str:
    if not ep.configured:
        raise NotConfigured(f"{ep.name} endpoint not configured")
    temperature, max_tokens = _tuned(ep, temperature, max_tokens)
    json_mode = json_mode and ep.json_mode

    async def _once() -> str:
        return await _backend.chat(ep, messages, temperature=temperature, max_tokens=max_tokens,
                                   json_mode=json_mode)

    return await _with_retry(_once, policy)


async def complete_json(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.0,
                        max_tokens: int | None = None) -> Any:
    """JSON-producing call. Tries json mode (unless the model's JSON_MODE is off), then
    a plain call. Raises LLMError if unparseable, TokenLimit if the model ran out."""
    # JSON-mode fallback shares the original deadline and retry count.
    policy = _RetryPolicy(ep)
    if ep.json_mode:
        # A 400 (including response_format validation) is terminal. Providers that
        # do not support JSON mode use <COMPONENT>_JSON_MODE=false instead.
        parsed = parse_json(await _complete(ep, messages, policy=policy, temperature=temperature,
                                            max_tokens=max_tokens, json_mode=True))
        if parsed is not None:
            return parsed
    parsed = parse_json(await _complete(ep, messages, policy=policy, temperature=temperature, max_tokens=max_tokens))
    if parsed is None:
        raise LLMError(f"{ep.name}: unparseable JSON")
    return parsed


async def complete_obj(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.0,
                       max_tokens: int | None = None) -> dict:
    """A JSON *object* reply, or LLMError: a model that answers with a list or a bare
    value is a failed call, never an AttributeError deep in the caller."""
    data = await complete_json(ep, messages, temperature=temperature, max_tokens=max_tokens)
    if not isinstance(data, dict):
        raise LLMError(f"{ep.name}: expected a JSON object, got {type(data).__name__}")
    return data


async def stream(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.2,
                 max_tokens: int | None = None):
    """Yield text deltas. Backends without streaming yield one whole reply.

    The call record is made current only while the backend runs (each `anext`),
    never across a `yield`: the consumer's own calls must not report into it."""
    if not ep.configured:
        raise NotConfigured(f"{ep.name} endpoint not configured")
    temperature, max_tokens = _tuned(ep, temperature, max_tokens)
    METER.add(ep.name, calls=1)
    rec = Call(ep.name, ep.model or "", at=time.time())
    t0 = time.perf_counter()
    fn = getattr(_backend, "stream", None)
    # Streaming is Bubble's interactive writer. Gate retries with its short budget;
    # even the first delta keeps the transport timeout (reasoning can take longer).
    policy = _RetryPolicy(ep, interactive=True)
    try:
        empty_attempts = 0
        while True:
            pending = ""
            try:
                if fn is None:
                    token = _CURRENT.set(rec)
                    try:
                        reply = await _backend.chat(ep, messages, temperature=temperature,
                                                    max_tokens=max_tokens, json_mode=False)
                    finally:
                        _CURRENT.reset(token)
                    if not reply.strip():
                        raise EmptyCompletion()
                    rec.first_ms = (time.perf_counter() - t0) * 1000
                    yield reply
                    return
                async with contextlib.aclosing(fn(ep, messages, temperature=temperature,
                                                  max_tokens=max_tokens)) as inner:
                    while True:
                        token = _CURRENT.set(rec)
                        try:
                            piece = await anext(inner)
                        except StopAsyncIteration:
                            break
                        finally:
                            _CURRENT.reset(token)
                        if rec.first_ms is None:
                            pending += piece or ""
                            if not pending.strip():
                                continue
                            piece, pending = pending, ""
                            rec.first_ms = (time.perf_counter() - t0) * 1000
                        yield piece
                if rec.first_ms is None:
                    raise EmptyCompletion()
                return
            except EXPECTED_ERRORS as exc:
                if isinstance(exc, TokenLimit) and rec.first_ms is None:
                    exc = EmptyCompletion("length", reasoning_present=True)
                if isinstance(exc, EmptyCompletion):
                    log.warning("%s: empty writer completion; finish_reason=%s reasoning_present=%s",
                                ep.name, exc.finish_reason, exc.reasoning_present)
                    empty_attempts += 1
                    if empty_attempts > 1:
                        raise exc
                    if exc.finish_reason == "length" and max_tokens is not None:
                        max_tokens += max(256, max_tokens // 2)
                # Share the transport retry deadline/count; never replay visible text.
                if not await policy.retry(exc, replayable=rec.first_ms is None):
                    raise exc
    except BaseException as exc:
        rec.status = _status(exc)
        raise
    finally:
        rec.ms = (time.perf_counter() - t0) * 1000
        PERF.call(rec)


def image_part(jpeg: bytes, *, detail: str = "auto") -> dict:
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


async def embed(texts: list[str]) -> list[list[float]]:
    """Batch embeddings through the embedding endpoint. Raises on failure."""
    ep = embedding_endpoint()
    if not ep.configured:
        raise NotConfigured("embedding endpoint not configured")
    out: list[list[float]] = []
    for i in range(0, len(texts), T.EMBED_BATCH):
        batch = [t if t.strip() else " " for t in texts[i : i + T.EMBED_BATCH]]
        got = await _with_retry(lambda batch=batch: _backend.embed(ep, batch), _RetryPolicy(ep))
        if len(got) != len(batch):
            raise LLMError(f"embedding: {len(got)} vectors for {len(batch)} texts")
        out.extend(got)
    return out


async def try_embed(text: str) -> list[float] | None:
    """One embedding, or None when the endpoint is missing or failing."""
    try:
        return (await embed([text]))[0]
    except (NotConfigured, *EXPECTED_ERRORS):
        return None


def embedding_model() -> str:
    return embedding_endpoint().model or ""


async def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    ep = rerank_endpoint()
    if not ep.configured:
        raise NotConfigured("rerank endpoint not configured")
    return await _with_retry(lambda: _backend.rerank(ep, query, docs, top_n), _RetryPolicy(ep))
