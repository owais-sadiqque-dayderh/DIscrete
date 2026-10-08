"""Per-model request settings from .env (docs/C7.md §4): any model without code changes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from discrete import config, llm
from discrete.config import ConfigError, caption_endpoint, embedding_endpoint, writer_endpoint


@pytest.fixture
def env(monkeypatch):
    for k in ("CORE_LM", "CAPTION_LM", "INSIGHT_LM", "BUBBLE_LM", "EMBED_LM"):
        for f in ("EXTRA_TOKENS", "REASONING_EFFORT", "TEMPERATURE", "JSON_MODE", "TIMEOUT_S", "EXTRA_BODY"):
            monkeypatch.delenv(f"{k}_{f}", raising=False)
    monkeypatch.setenv("CORE_LM_API_KEY", "k")
    monkeypatch.setenv("CORE_LM_MODEL_NAME", "meta/thinker")
    monkeypatch.setenv("CORE_LM_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("EMBED_LM_MODEL_NAME", "openai/text-embedding-3-small")
    return monkeypatch


def test_settings_cascade_from_core_and_components_override(env):
    env.setenv("CORE_LM_EXTRA_TOKENS", "1000")
    env.setenv("CORE_LM_REASONING_EFFORT", "minimal")
    env.setenv("CAPTION_LM_REASONING_EFFORT", "LOW")
    env.setenv("INSIGHT_LM_TEMPERATURE", "off")
    cap, w = caption_endpoint(), writer_endpoint()
    assert cap.extra_tokens == 1000 and cap.reasoning_effort == "low" and cap.prefix == "CAPTION_LM"
    assert w.reasoning_effort == "minimal" and w.no_temperature  # writer falls back to INSIGHT_LM, then CORE_LM
    assert cap.body_extras() == {"reasoning": {"effort": "low"}}
    emb = embedding_endpoint()  # never inherits chat settings
    assert emb.extra_tokens == 0 and emb.reasoning_effort is None and emb.body_extras() == {}


def test_reasoning_effort_uses_the_providers_field(env):
    env.setenv("CORE_LM_REASONING_EFFORT", "low")
    env.setenv("CORE_LM_API_BASE", "https://api.openai.com/v1")
    env.setenv("CORE_LM_EXTRA_BODY", '{"service_tier": "flex"}')
    assert caption_endpoint().body_extras() == {"reasoning_effort": "low", "service_tier": "flex"}


@pytest.mark.parametrize("name,value", [("CORE_LM_EXTRA_TOKENS", "lots"), ("CAPTION_LM_EXTRA_BODY", "{bad"),
                                        ("CAPTION_LM_EXTRA_BODY", "[1]"), ("CORE_LM_TEMPERATURE", "warm")])
def test_malformed_setting_names_the_variable(env, name, value):
    env.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        caption_endpoint()


class Recorder:
    def __init__(self, reply="{}"):
        self.reply, self.seen = reply, []

    async def chat(self, ep, messages, *, temperature, max_tokens, json_mode):
        self.seen.append({"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode})
        return self.reply


async def test_requests_carry_the_models_settings(env):
    env.setenv("CORE_LM_EXTRA_TOKENS", "1000")
    env.setenv("CORE_LM_TEMPERATURE", "0.7")
    env.setenv("CORE_LM_JSON_MODE", "false")
    prev, rec = llm.get_backend(), Recorder('{"ok": true}')
    llm.set_backend(rec)
    try:
        assert await llm.complete_json(caption_endpoint(), [{"role": "user", "content": "x"}], max_tokens=30) == {"ok": True}
    finally:
        llm.set_backend(prev)
    assert rec.seen == [{"temperature": 0.7, "max_tokens": 1030, "json_mode": False}]  # one call: no JSON-mode try


async def test_a_model_that_thinks_past_its_limit_raises_a_fix_it_error(env):
    backend = llm.OpenAIBackend()
    sent: dict = {}

    async def create(**kw):
        sent.update(kw)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="length",
                                                                    message=SimpleNamespace(content=""))])

    backend._client = lambda ep: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    env.setenv("CAPTION_LM_REASONING_EFFORT", "minimal")
    env.setenv("CAPTION_LM_TIMEOUT_S", "20")
    ep = caption_endpoint()
    with pytest.raises(llm.TokenLimit, match="CAPTION_LM_EXTRA_TOKENS"):
        await backend.chat(ep, [], temperature=0.0, max_tokens=500, json_mode=True)
    assert sent["extra_body"] == {"reasoning": {"effort": "minimal"}} and sent["timeout"] == 20.0


def test_every_component_is_listed_for_doctor_and_bench(env):
    names = [ep.name for ep in config.all_endpoints()]
    assert "bubble_vision" in names and len(names) == len(set(names))


def test_component_endpoint_cascades_like_own_components(monkeypatch):
    from discrete.config import component_endpoint

    monkeypatch.setenv("CORE_LM_MODEL_NAME", "core-model")
    monkeypatch.setenv("CORE_LM_API_KEY", "k")
    monkeypatch.setenv("VERIFY_LM_MODEL_NAME", "verify-model")
    monkeypatch.delenv("X_LM_MODEL_NAME", raising=False)
    assert component_endpoint("x", "X_LM", "VERIFY_LM").model == "verify-model"
    monkeypatch.setenv("X_LM_MODEL_NAME", "own-model")
    monkeypatch.setenv("X_LM_REASONING_EFFORT", "none")
    monkeypatch.setenv("CORE_LM_REASONING_EFFORT", "minimal")
    ep = component_endpoint("x", "X_LM", "VERIFY_LM")
    assert (ep.name, ep.model, ep.reasoning_effort) == ("x", "own-model", None)
