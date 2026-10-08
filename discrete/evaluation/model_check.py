"""`discrete doctor --models`: one real request per configured model, with the same
prompts, token limits and .env settings the pipeline uses, so a model swap in
.env is verified in seconds instead of discovered in a failed run.

Each check uses the component's *tightest* real call: a reasoning model that
passes here has room to think everywhere.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from PIL import Image, ImageDraw

from ..bubble.prompts import FIELD_READ_PROMPT
from ..capture.enrich import VLM_MAX_TOKENS, EnrichJob, jpeg, vlm_messages
from ..config import (
    Endpoint,
    T,
    bubble_vision_endpoint,
    caption_endpoint,
    consolidate_endpoint,
    embedding_endpoint,
    insight_endpoint,
    rerank_endpoint,
    utility_endpoint,
    verifier_endpoint,
    writer_endpoint,
)
from ..llm import EXPECTED_ERRORS, NotConfigured, TokenLimit, complete_json, embed, image_part, rerank, stream
from ..perf import PERF
from ..process.prompts import ADJUDICATE_PROMPT, VERIFY_CLAIMS_PROMPT
from ..retrieval.query import _PLAN_PROMPT


@dataclass
class Check:
    component: str
    model: str
    ok: bool
    seconds: float
    detail: str
    limit: int | None = None
    out_tokens: int = 0
    cost: float | None = None


async def check_models() -> list[Check]:
    """Run checks concurrently using only a synthetic, nonpersonal fixture."""
    frame = Image.new("RGB", (800, 500), (250, 250, 250))
    ImageDraw.Draw(frame).text((40, 40), "Synthetic test page: quarterly report", fill="black")

    async def caption() -> str:
        job = EnrichJob(frames=[frame], app="Test", window_title="Test window", url="",
                        screen_text="A synthetic test page.", counts=dict.fromkeys(("keys", "clicks", "scrolls",
                                                                                        "clipboard"), 0))
        data = await complete_json(caption_endpoint(), vlm_messages(job), max_tokens=VLM_MAX_TOKENS)
        return _need(data, "caption")

    async def bubble_vision() -> str:  # bubble/service.py _read_field
        data = await complete_json(bubble_vision_endpoint(), [{"role": "user", "content": [
            image_part(jpeg(frame), detail="high"), {"type": "text", "text": FIELD_READ_PROMPT}]}], max_tokens=800)
        return _need(data, "field_description")

    async def insight() -> str:  # entity adjudication: the smallest insight budget
        data = await complete_json(insight_endpoint(), [{"role": "user", "content": ADJUDICATE_PROMPT.format(
            a="GitHub", b="Github Inc")}], max_tokens=30)
        return _need(data, "same")

    async def utility() -> str:  # query planning, which Bubble abandons after PLAN_TIMEOUT_S
        t0 = time.perf_counter()
        data = await complete_json(utility_endpoint(), [{"role": "user", "content": _PLAN_PROMPT.format(
            q="which articles did I read about rust last week")}], max_tokens=300)
        took = time.perf_counter() - t0
        out = _need(data, "intent")
        if took > T.PLAN_TIMEOUT_S:
            raise _Slow(f"answered in {took:.1f}s but Bubble waits only {T.PLAN_TIMEOUT_S:.0f}s: raise "
                        f"DISCRETE_PLAN_TIMEOUT_S or use a faster UTILITY_LM model")
        return out

    async def verifier() -> str:
        candidates = [{"index": 0, "kind": "event", "text": "The user exported the report.",
                       "observations": [{"screen_text": "I did not export the report."}]}]
        data = await complete_json(verifier_endpoint(), [{"role": "user", "content": VERIFY_CLAIMS_PROMPT.format(
            candidates=json.dumps(candidates))}], max_tokens=600)
        decisions = data.get("decisions", []) if isinstance(data, dict) else []
        if len(decisions) != 1 or decisions[0].get("index") != 0 or decisions[0].get("supported") is not False:
            raise ValueError("verifier accepted a negated action or returned malformed decisions")
        return "negated action rejected"

    async def consolidate() -> str:  # entity pages / clusters: 400-token JSON
        data = await complete_json(consolidate_endpoint(), [{"role": "user", "content":
            'Summarise in one sentence what "Linear" is to a software engineer. '
            'Return strict JSON: {"kind": "<software|person|org|topic>", "summary": "<one sentence>"}'}],
            max_tokens=400)
        return _need(data, "summary")

    async def writer() -> str:  # Bubble's streamed answer
        out = ""
        async for piece in stream(writer_endpoint(), [{"role": "user", "content":
                                  "Reply with one short sentence confirming you can write."}],
                                  temperature=0.2, max_tokens=T.BUBBLE_WRITER_MAX_TOKENS):
            out += piece
        if not out.strip():
            raise ValueError("empty reply")
        return out.strip()

    async def embedding() -> str:
        vec = (await embed(["a test sentence"]))[0]
        return f"{len(vec)} dimensions"

    async def reranker() -> str:
        order = await rerank("cats", ["a dog", "a cat"], 2)
        if not order or order[0][0] != 1:
            raise ValueError(f"unexpected order {order}")
        return "ranked correctly"

    checks: list[tuple[Endpoint, int | None, Callable[[], Awaitable[str]]]] = [
        (caption_endpoint(), VLM_MAX_TOKENS, caption),
        (bubble_vision_endpoint(), 800, bubble_vision),
        (insight_endpoint(), 30, insight),
        (verifier_endpoint(), 600, verifier),
        (utility_endpoint(), 300, utility),
        (consolidate_endpoint(), 400, consolidate),
        (writer_endpoint(), T.BUBBLE_WRITER_MAX_TOKENS, writer),
        (embedding_endpoint(), None, embedding),
        (rerank_endpoint(), None, reranker),
    ]
    PERF.drain()
    results = await asyncio.gather(*(_run(ep, limit, fn) for ep, limit, fn in checks))
    calls, _ = PERF.drain()
    for r in results:
        mine = [c for c in calls if c.component == r.component]
        r.out_tokens = sum(c.completion_tokens for c in mine)
        costs = [c.cost for c in mine if c.cost is not None]
        r.cost = sum(costs) if costs else None
    return results


class _Slow(Exception):
    pass


def _need(data, key: str) -> str:
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"reply has no {key!r}: {json.dumps(data)[:120]}")
    return f"{key}={json.dumps(data[key])[:70]}"


async def _run(ep: Endpoint, limit: int | None, fn: Callable[[], Awaitable[str]]) -> Check:
    total = None if limit is None else limit + ep.extra_tokens
    t0 = time.perf_counter()
    try:
        detail, ok = await fn(), True
    except NotConfigured:
        detail, ok = "not configured", True
    except TokenLimit as exc:
        detail, ok = str(exc).split(": ", 1)[-1], False
    except _Slow as exc:
        detail, ok = str(exc), False
    except (*EXPECTED_ERRORS, ValueError) as exc:
        detail, ok = f"{type(exc).__name__}: {exc}"[:300], False
    return Check(ep.name, ep.model or "-", ok, time.perf_counter() - t0, detail, total)


def render(results: list[Check]) -> str:
    rows = [("component", "model", "", "time", "out tok / limit", "cost $", "")]
    for r in results:
        rows.append((r.component, r.model, "OK" if r.ok else "FAIL", f"{r.seconds:.1f}s",
                     f"{r.out_tokens} / {r.limit}" if r.limit else f"{r.out_tokens}",
                     f"{r.cost:.5f}" if r.cost is not None else "-", r.detail))
    widths = [max(len(str(row[i])) for row in rows) for i in range(6)]
    return "\n".join("  " + "  ".join(str(row[i]).ljust(widths[i]) for i in range(6)) + "  " + str(row[6])
                     for row in rows)
