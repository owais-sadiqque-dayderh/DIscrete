"""Step-0 measurement: per-call records, stage samples, and `discrete bench` (docs/C7.md)."""

from __future__ import annotations

import asyncio
import json

import pytest

from discrete import llm
from discrete.config import Endpoint
from discrete.evaluation import bench
from discrete.perf import PERF, Call, Sample

EP = Endpoint("insight", "http://fake", "k", "fake/model")
WRITER = Endpoint("writer", "http://fake", "k", "fake/writer")


class UsageBackend:
    """Reports usage (with cost) the way OpenRouter does, after a delay."""

    def __init__(self) -> None:
        self.fail = False

    async def chat(self, ep, messages, *, temperature, max_tokens, json_mode) -> str:
        n = len(messages[-1]["content"])
        await asyncio.sleep(0.01 * (n % 3))
        if self.fail:
            raise llm.LLMError("down")
        llm.METER.add_usage(ep.name, {"prompt_tokens": n, "completion_tokens": 2, "cost": n / 1e6})
        return "ok"

    async def stream(self, ep, messages, *, temperature, max_tokens):
        for piece in ("one ", "two ", "three"):
            await asyncio.sleep(0.005)
            yield piece
        llm.METER.add_usage(ep.name, {"prompt_tokens": 50, "completion_tokens": 3, "cost": 0.00005})


@pytest.fixture
def usage_backend():
    prev = llm.get_backend()
    backend = UsageBackend()
    llm.set_backend(backend)
    PERF.drain()
    llm.METER.drain()
    yield backend
    llm.set_backend(prev)
    PERF.drain()


def _calls():
    calls, _ = PERF.drain()
    return calls


async def test_each_call_gets_its_own_tokens_and_cost_even_when_concurrent(usage_backend):
    prompts = ["a" * 10, "b" * 11, "c" * 12, "d" * 13]
    await asyncio.gather(*(llm.complete(EP, [{"role": "user", "content": p}]) for p in prompts))
    calls = _calls()
    assert sorted(c.prompt_tokens for c in calls) == [10, 11, 12, 13]
    for c in calls:
        assert c.component == "insight" and c.model == "fake/model" and c.status == "ok"
        assert c.cost == pytest.approx(c.prompt_tokens / 1e6)
        assert c.ms >= 0 and c.source == "live"


async def test_failed_call_is_recorded_as_error(usage_backend):
    usage_backend.fail = True
    with pytest.raises(llm.LLMError):
        await llm.complete(EP, [{"role": "user", "content": "x"}])
    [c] = _calls()
    assert c.status == "error" and c.prompt_tokens == 0 and c.cost is None


async def test_stream_records_first_token_and_keeps_the_consumers_calls_apart(usage_backend):
    out = []
    async for piece in llm.stream(WRITER, [{"role": "user", "content": "q"}]):
        out.append(piece)
        # A call made by the consumer mid-stream reports into its own record only.
        await llm.complete(EP, [{"role": "user", "content": "x" * 20}])
    assert "".join(out) == "one two three"
    calls = {c.component: c for c in _calls() if c.component == "writer"}
    w = calls["writer"]
    assert w.prompt_tokens == 50 and w.completion_tokens == 3 and w.cost == pytest.approx(0.00005)
    assert w.first_ms is not None and 0 < w.first_ms <= w.ms and w.status == "ok"


async def test_abandoned_stream_is_recorded_as_cancelled(usage_backend):
    gen = llm.stream(WRITER, [{"role": "user", "content": "q"}])
    async for _ in gen:
        break
    await gen.aclose()
    [c] = _calls()
    assert c.status == "cancelled" and c.first_ms is not None


def test_timer_records_only_blocks_that_finish():
    PERF.drain()
    with PERF.timer("draft.episode", ref_id=3) as t:
        t.n = 4
    with pytest.raises(RuntimeError), PERF.timer("draft.episode"):
        raise RuntimeError
    PERF.count("capture.frame_dup", 0)  # zero counts are not stored
    PERF.count("capture.frame_dup", 2)
    _, samples = PERF.drain()
    assert [(s.stage, s.n, s.ref_id) for s in samples] == [("draft.episode", 4, 3), ("capture.frame_dup", 2, None)]
    assert samples[0].ms is not None and samples[1].ms is None


def test_active_time_bridges_short_pauses_only():
    spans = [(0, 10), (100, 110), (1000, 1010)]  # a 90 s pause, then a 890 s one
    assert bench.active_seconds(spans, 300) == 110 + 10
    assert bench.active_seconds([], 300) == 0


# ------------------------------------------------------------------ store + bench

async def _seed(store, t0: float) -> None:
    from discrete.models import ObservationDraft

    for i in range(4):
        await store.add_observation(ObservationDraft(
            app="Safari", window_title=f"Page {i}", url=f"https://ex.org/{i}", screen_text=f"distinct text {i} " * 20,
            caption=f"Safari: Page {i}", started_at=t0 + i * 60, ended_at=t0 + i * 60 + 30, session_id="s",
            enrich_state="done"))
    calls = [Call("caption", "m/vision", at=t0 + i * 60, ms=900 + i * 100, prompt_tokens=3900, completion_tokens=90,
                  cost=0.0006, source="live") for i in range(4)]
    calls += [Call("embedding", "m/embed", at=t0 + 5, ms=120, prompt_tokens=400, cost=0.00001, source="live"),
              Call("utility", "m/chat", at=t0 + 6, ms=700, prompt_tokens=300, cost=0.0002, source="eval")]
    samples = [Sample("enrich.total", 1000.0 + i * 500, at=t0 + i * 60, source="live") for i in range(4)]
    samples += [Sample("capture.frame_dup", None, at=t0 + 10, n=7, source="live"),
                Sample("retrieve.total", 5000.0, at=t0 + 7, source="eval")]
    await store.record_perf(calls, samples)


async def test_bench_reports_cost_latency_and_captures_for_the_pipeline_only(store):
    t0 = 1_789_000_000.0
    await _seed(store, t0)
    window = await bench.parse_window(store, "all", None)
    snap = await bench.collect(store, window)
    assert "eval" not in snap["meta"]["sources"]  # eval runs are not pipeline cost
    assert snap["cost"]["usd_total"] == pytest.approx(4 * 0.0006 + 0.00001)
    assert snap["activity"]["active_hours"] == pytest.approx((3 * 60 + 30) / 3600, rel=1e-3)
    cap = snap["models"]["caption"]
    assert cap["calls"] == 4 and cap["prompt_tokens_per_call"] == 3900 and cap["cost_share"] > 0.99
    assert snap["latency_ms"]["enrich.total"]["n"] == 4 and "retrieve.total" not in snap["latency_ms"]
    assert snap["captures"]["rows"] == 4 and snap["captures"]["capture.frame_dup"] == 7
    text = bench.render(snap)
    assert "caption" in text and "enrich.total" in text and "COST" in text
    json.dumps(snap, default=str)  # a snapshot is plain data


async def test_bench_window_and_source_filters(store):
    t0 = 1_789_000_000.0
    await _seed(store, t0)
    snap = await bench.collect(store, bench.Window(t0 + 100, None, "late"))
    assert snap["models"]["caption"]["calls"] == 2  # the two calls at t0+120, t0+180
    snap = await bench.collect(store, bench.Window(None, None, "all"), sources=["eval"])
    assert set(snap["models"]) == {"utility"}


def test_compare_marks_direction_and_config_changes():
    def snap(cost, p50, width):
        return {"meta": {"name": "", "window": {"label": "w"}, "git": {"rev": "abc", "dirty": False},
                         "models": {"caption": "m"}, "tunables": {"VLM_MAX_WIDTH": width}},
                "cost": {"usd_per_active_hour": cost, "usd_total": 1.0}, "activity": {"active_hours": 1.0},
                "captures": {}, "models": {}, "storage": {}, "quality": None,
                "latency_ms": {"bubble.total": {"n": 3, "p50": p50, "p90": p50, "max": p50}}}

    text = bench.compare(snap(0.20, 3000, 1024), snap(0.10, 3600, 768))
    lines = {ln.split()[0]: ln for ln in text.splitlines() if ln.strip()}
    assert "-50%" in lines["cost.usd_per_active_hour"] and "better" in lines["cost.usd_per_active_hour"]
    assert "WORSE" in lines["latency.bubble.total.p50"]
    assert "VLM_MAX_WIDTH: 1024 → 768" in text


async def test_runtime_records_stage_timings(store, fake):
    from discrete.app import Runtime
    from discrete.capture.enrich import Capture
    from tests.test_runtime import TEXT, _draft, _job

    PERF.drain()
    rt = Runtime(store, bubble=False)
    tasks = [asyncio.create_task(rt._ingest_loop()), asyncio.create_task(rt._segment_loop())]
    try:
        await rt.queue.put(Capture(_draft("a", enrich_state="pending"), _job(TEXT)))
        for _ in range(200):
            if await store.q("SELECT 1 FROM observations WHERE episode_id IS NOT NULL"):
                break
            await asyncio.sleep(0.01)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    calls, samples = PERF.drain()
    stages = {s.stage for s in samples}
    assert {"ingest.stored", "enrich.wait", "enrich.vision", "enrich.total", "segment.embed", "segment.lag"} <= stages
    assert {c.component for c in calls} >= {"caption", "embedding"}


def test_cli_bench_saves_and_compares(tmp_db, fake, monkeypatch, tmp_path, capsys):
    from discrete import cli

    monkeypatch.setenv("DISCRETE_BENCH_DIR", str(tmp_path / "bench"))
    for argv in (["bench", "--save", "base"], ["bench", "--compare", "base"], ["bench", "--compare", "base", "base"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--db", tmp_db, *argv])
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert (tmp_path / "bench" / "base.json").exists()
    assert "COST" in out and "Config: identical" in out
