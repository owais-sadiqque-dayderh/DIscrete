"""API-only model access (plans Rev 2): chat, vision, embeddings, rerank.

A pluggable backend lets tests run without a network. Callers catch
EXPECTED_ERRORS only; anything else propagates (insights.md §7.2).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from .config import Endpoint, T, embedding_endpoint, rerank_endpoint

log = logging.getLogger("discrete.llm")


class LLMError(RuntimeError):
    """A model call returned nothing usable."""


class NotConfigured(LLMError):
    pass


EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    LLMError,
    openai.APIError,
    httpx.HTTPError,
    asyncio.TimeoutError,
)


class OpenAIBackend:
    """Talks to any OpenAI-compatible API (OpenRouter by default)."""

    def __init__(self) -> None:
        self._clients: dict[tuple[str | None, str | None], AsyncOpenAI] = {}
        self._http: httpx.AsyncClient | None = None  # kept alive: one TLS handshake, not one per rerank

    def _client(self, ep: Endpoint) -> AsyncOpenAI:
        key = (ep.base_url, ep.api_key)
        if key not in self._clients:
            self._clients[key] = AsyncOpenAI(
                base_url=ep.base_url, api_key=ep.api_key, timeout=T.LLM_TIMEOUT_S, max_retries=2
            )
        return self._clients[key]

    async def chat(self, ep: Endpoint, messages: list[dict], *, temperature: float | None,
                   max_tokens: int | None, json_mode: bool) -> str:
        payload: dict[str, Any] = {"model": ep.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        rsp = await self._client(ep).chat.completions.create(**payload)
        if not rsp.choices:
            raise LLMError(f"{ep.name}: empty choices")
        return (rsp.choices[0].message.content or "").strip()

    async def stream(self, ep: Endpoint, messages: list[dict], *, temperature: float | None,
                     max_tokens: int | None):
        payload: dict[str, Any] = {"model": ep.model, "messages": messages, "stream": True}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        rsp = await self._client(ep).chat.completions.create(**payload)
        try:
            async for chunk in rsp:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await rsp.close()  # an abandoned stream otherwise errors when the loop shuts down

    async def embed(self, ep: Endpoint, texts: list[str]) -> list[list[float]]:
        rsp = await self._client(ep).embeddings.create(model=ep.model, input=texts)
        return [d.embedding for d in rsp.data]

    async def rerank(self, ep: Endpoint, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        url = (ep.base_url or "https://openrouter.ai/api/v1").rstrip("/") + "/rerank"
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=T.LLM_TIMEOUT_S)
        r = await self._http.post(
            url,
            headers={"Authorization": f"Bearer {ep.api_key}"},
            json={"model": ep.model, "query": query, "documents": docs, "top_n": top_n},
        )
        r.raise_for_status()
        data = r.json()
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


async def complete(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.0,
                   max_tokens: int | None = None, json_mode: bool = False) -> str:
    if not ep.configured:
        raise NotConfigured(f"{ep.name} endpoint not configured")
    return await _backend.chat(ep, messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


async def complete_json(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.0,
                        max_tokens: int | None = None) -> Any:
    """JSON-producing call. Tries json mode, then a plain call. Raises LLMError if unparseable."""
    try:
        parsed = parse_json(await complete(ep, messages, temperature=temperature, max_tokens=max_tokens, json_mode=True))
        if parsed is not None:
            return parsed
    except openai.BadRequestError as exc:  # proxy rejected response_format
        log.debug("json mode rejected by %s: %s", ep.model, exc)
    parsed = parse_json(await complete(ep, messages, temperature=temperature, max_tokens=max_tokens))
    if parsed is None:
        raise LLMError(f"{ep.name}: unparseable JSON")
    return parsed


async def stream(ep: Endpoint, messages: list[dict], *, temperature: float | None = 0.2,
                 max_tokens: int | None = None):
    """Yield text deltas. Backends without streaming yield one whole reply."""
    if not ep.configured:
        raise NotConfigured(f"{ep.name} endpoint not configured")
    fn = getattr(_backend, "stream", None)
    if fn is None:
        yield await _backend.chat(ep, messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
        return
    async for piece in fn(ep, messages, temperature=temperature, max_tokens=max_tokens):
        yield piece


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
        out.extend(await _backend.embed(ep, batch))
    return out


def embedding_model() -> str:
    return embedding_endpoint().model or ""


async def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    ep = rerank_endpoint()
    if not ep.configured:
        raise NotConfigured("rerank endpoint not configured")
    return await _backend.rerank(ep, query, docs, top_n)
