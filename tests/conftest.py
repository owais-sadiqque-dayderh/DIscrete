"""Test fixtures: a temp store and a deterministic fake model backend (no network)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from discrete import llm
from discrete import textkit as tk


def _prompt(messages) -> str:
    content = messages[-1]["content"]
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


class FakeBackend:
    """Routes chat calls by prompt text. Tests set the response hooks they need."""

    def __init__(self) -> None:
        self.draft: dict = {"title": "t", "summary": "s", "claims": []}
        self.relations: dict[int, str] = {}
        self.consolidation: dict = {"claims": []}
        self.writer_reply = ""
        self.calls: list[str] = []

    async def chat(self, ep, messages, *, temperature, max_tokens, json_mode) -> str:
        p = _prompt(messages)
        if "Memory records:" in p:
            self.calls.append("writer")
            return self.writer_reply
        if "one episode of a person's computer activity" in p:
            self.calls.append("draft")
            return json.dumps(self.draft)
        if "Compare a NEW claim" in p:
            self.calls.append("relate")
            ids = [int(x) for x in __import__("re").findall(r"\[id=(\d+)\]", p)]
            return json.dumps({"relations": [{"id": i, "relation": self.relations.get(i, "unrelated")} for i in ids]})
        if "Rewrite them as a clear" in p:
            self.calls.append("consolidate")
            return json.dumps(self.consolidation)
        if "Plan a memory search" in p:
            self.calls.append("plan")
            return json.dumps({"intent": "retrieve_reading_history", "queries": [], "needs_store": False})
        if "Write a short page" in p:
            return json.dumps({"kind": "software", "summary": "A tool the user uses."})
        if "digest" in p:
            return "The user worked."
        if "same thing" in p:
            return json.dumps({"same": False})
        return "{}"

    async def embed(self, ep, texts):
        return [bow_vector(t) for t in texts]

    async def rerank(self, ep, query, docs, top_n):
        q = set(tk.content_tokens(query))
        scored = [(i, len(q & set(tk.content_tokens(d))) / max(1, len(q))) for i, d in enumerate(docs)]
        return sorted(scored, key=lambda x: -x[1])[:top_n]


def bow_vector(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    for t in tk.content_tokens(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1.0
    if not v.any():
        v[0] = 1.0
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


@pytest.fixture
def fake(monkeypatch):
    for k, v in {
        "CORE_LM_API_KEY": "test", "CORE_LM_MODEL_NAME": "fake-chat", "CORE_LM_API_BASE": "http://fake",
        "EMBED_LM_API_KEY": "test", "EMBED_LM_MODEL_NAME": "fake-embed",
        "RERANK_LM_API_KEY": "test", "RERANK_LM_MODEL_NAME": "fake-rerank",
    }.items():
        monkeypatch.setenv(k, v)
    for k in ("CAPTION_LM_MODEL_NAME", "INSIGHT_LM_MODEL_NAME", "UTILITY_LM_MODEL_NAME", "CONSOLIDATE_LM_MODEL_NAME",
              "CAPTION_LM_API_KEY", "INSIGHT_LM_API_KEY", "UTILITY_LM_API_KEY", "EMBED_LM_API_BASE"):
        monkeypatch.delenv(k, raising=False)
    backend = FakeBackend()
    prev = llm.get_backend()
    llm.set_backend(backend)
    yield backend
    llm.set_backend(prev)


@pytest.fixture
async def store(tmp_db, fake):
    from discrete.store import Store

    s = await Store.open(tmp_db)
    yield s
    await s.close()
