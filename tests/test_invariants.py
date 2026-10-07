"""Read-only regression gates and durable silent-loss telemetry (audit #12)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from discrete import textkit as tk
from discrete.cli import build_parser
from discrete.config import T
from discrete.evaluation import bench
from discrete.evaluation.check_cli import cmd_check
from discrete.evaluation.invariants import check_store, load_expectations
from discrete.evaluation.slos import evaluate_slos
from discrete.llm import LLMError, flush_usage
from discrete.models import ObservationDraft
from discrete.perf import PERF, Sample
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval.store_card import store_card


@pytest.fixture
def gate_db(tmp_path):
    """Small legacy-compatible store, deliberately independent of current migrations."""
    path = tmp_path / "golden.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE observations(id INTEGER PRIMARY KEY, origin TEXT DEFAULT 'user', app TEXT DEFAULT 'Browser',
            site TEXT DEFAULT 'example.com', screen_text TEXT DEFAULT '', caption TEXT DEFAULT '',
            window_title TEXT DEFAULT '', page_title TEXT DEFAULT '', episode_id INTEGER);
        CREATE TABLE claims(id INTEGER PRIMARY KEY, text TEXT, kind TEXT DEFAULT 'event', action TEXT DEFAULT '',
            status TEXT DEFAULT 'live');
        CREATE TABLE evidence(claim_id INTEGER, observation_id INTEGER, quote TEXT);
        CREATE TABLE episodes(id INTEGER PRIMARY KEY, status TEXT, draft_attempts INTEGER);
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE perf_samples(stage TEXT, ms REAL, src TEXT);
        INSERT INTO observations(id,screen_text) VALUES (1,'Downloaded ChatGPT.dmg successfully');
        INSERT INTO claims(id,text,action) VALUES (1,'The user downloaded ChatGPT.dmg','downloaded');
        INSERT INTO evidence VALUES (1,1,'Downloaded ChatGPT.dmg');
    """)
    conn.commit()
    yield path, conn
    conn.close()


def checks(path, **kwargs):
    return {result.name: result for result in check_store(path, **kwargs).checks}


def execute(conn, sql, params=()):
    conn.execute(sql, params)
    conn.commit()


def test_gate_passes_and_never_changes_store_or_schema(gate_db):
    path, _ = gate_db
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert check_store(path).passed
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not Path(str(path) + "-journal").exists()


@pytest.mark.parametrize("origin", ["agent", "self", "discarded"])
def test_no_live_claim_may_cite_non_user_observations(gate_db, origin):
    path, conn = gate_db
    execute(conn, "UPDATE observations SET origin=?", (origin,))
    result = checks(path)["no_self_evidence"]
    assert not result.passed
    assert result.details["violations"] == [{"claim_id": 1, "observation_id": 1, "origin": origin}]
    execute(conn, "UPDATE claims SET status='forgotten'")
    assert check_store(path).passed


@pytest.mark.parametrize("mutation", ["DELETE FROM evidence", "UPDATE evidence SET quote='invented quote'",
                                     "UPDATE evidence SET quote=''", "UPDATE evidence SET observation_id=99"])
def test_every_live_claim_needs_an_authentic_quote(gate_db, mutation):
    path, conn = gate_db
    execute(conn, mutation)
    result = checks(path)["grounded_evidence"]
    assert not result.passed and result.details["claim_ids"] == [1]


@pytest.mark.parametrize("field", ["screen_text", "caption", "window_title", "page_title"])
def test_quote_matches_normalized_text_in_any_single_field(gate_db, field):
    path, conn = gate_db
    execute(conn, "UPDATE observations SET screen_text=''")
    execute(conn, f"UPDATE observations SET {field}='Downloaded\n  CHATGPT.dmg'")
    assert checks(path)["grounded_evidence"].passed
    execute(conn, "INSERT INTO evidence VALUES (1,99,'invented')")
    result = checks(path)["grounded_evidence"]
    assert not result.passed
    assert result.details["violations"] == [{"claim_id": 1, "observation_id": 99}]


def test_quote_cannot_span_separate_fields(gate_db):
    path, conn = gate_db
    execute(conn, "UPDATE observations SET screen_text='Downloaded',caption='ChatGPT.dmg'")
    assert not checks(path)["grounded_evidence"].passed


def test_drafted_episode_requires_complete_persistent_coverage(gate_db):
    path, conn = gate_db
    execute(conn, "INSERT INTO episodes VALUES (7,'closed',1)")
    execute(conn, "UPDATE observations SET episode_id=7")
    result = checks(path)["draft_coverage"]
    assert result.status == "n/a" and result.passed is None
    assert result.details["unverifiable_episodes"] == [7] and result.details["reason"]
    assert check_store(path).passed
    execute(conn, "INSERT INTO meta VALUES ('draft_coverage:7',?)",
            (json.dumps({"input_hash": "hash", "chunks": {"0": []}}),))
    result = checks(path)["draft_coverage"]
    assert not result.passed and result.details["missing_observations"] == {7: [1]}
    execute(conn, "UPDATE meta SET value=?", (json.dumps({"input_hash": "hash", "chunks": {"0": [1]}}),))
    assert checks(path)["draft_coverage"].passed
    execute(conn, "INSERT INTO observations(id,origin,episode_id) VALUES (2,'agent',7)")
    assert checks(path)["draft_coverage"].passed  # audit-only rows are intentionally not drafted
    execute(conn, "INSERT INTO observations(id,screen_text,episode_id) VALUES (3,'new user row',7)")
    assert not checks(path)["draft_coverage"].passed


@pytest.mark.parametrize("length", [11999, 12000, 15999, 16000, 16001])
def test_cap_hits_need_individual_flags(gate_db, length):
    path, conn = gate_db
    execute(conn, "UPDATE observations SET screen_text=?", ("x" * length,))
    execute(conn, "INSERT INTO meta VALUES ('screen_text_flags_from','1')")
    result = checks(path)["truncation_flagged"]
    assert not result.passed and result.details["unflagged"] == [1]
    execute(conn, "INSERT INTO meta VALUES ('screen_text_truncated:1','16000')")
    assert checks(path)["truncation_flagged"].passed


def test_legacy_truncation_and_missing_columns_are_not_measured(gate_db, tmp_path):
    path, conn = gate_db
    execute(conn, "UPDATE observations SET screen_text=?", ("x" * 12000,))
    result = checks(path)["truncation_flagged"]
    assert result.status == "n/a" and result.details["unverifiable_observations"] == [1]
    execute(conn, "ALTER TABLE episodes DROP COLUMN draft_attempts")
    execute(conn, "ALTER TABLE observations DROP COLUMN page_title")
    execute(conn, "ALTER TABLE claims DROP COLUMN action")
    execute(conn, "ALTER TABLE perf_samples DROP COLUMN src")
    expected = tmp_path / "expect.yaml"
    expected.write_text("claims: [{name: download, text_regex: downloaded, action: downloaded}]")
    results = checks(path, expect=expected, slo=True)
    for name in ("draft_coverage", "grounded_evidence", "expected:download", "slo:capture.text"):
        assert results[name].status == "n/a" and results[name].details["reason"]
    assert results["no_self_evidence"].status == "pass"
    assert check_store(path, expect=expected, slo=True).passed


def test_legacy_store_with_no_perf_table_has_na_slos(gate_db):
    path, conn = gate_db
    execute(conn, "DROP TABLE perf_samples")
    assert check_store(path, slo=True).passed
    assert all(r.status == "n/a" for name, r in checks(path, slo=True).items() if name.startswith("slo:"))


def test_partial_legacy_telemetry_does_not_hide_real_violations(gate_db):
    path, conn = gate_db
    execute(conn, "INSERT INTO episodes VALUES (7,'closed',1),(8,'closed',1)")
    execute(conn, "UPDATE observations SET episode_id=7,screen_text=?", ("x" * 12000,))
    execute(conn, "INSERT INTO observations(id,episode_id,screen_text) VALUES (2,8,?)", ("x" * 12000,))
    execute(conn, "INSERT INTO meta VALUES ('screen_text_flags_from','2')")
    execute(conn, "INSERT INTO meta VALUES ('draft_coverage:8',?)",
            (json.dumps({"input_hash": "h", "chunks": {"0": []}}),))
    results = checks(path)
    assert results["truncation_flagged"].status == "fail"
    assert results["truncation_flagged"].details["unflagged"] == [2]
    assert results["truncation_flagged"].details["unverifiable_observations"] == [1]
    assert results["draft_coverage"].status == "fail"
    assert results["draft_coverage"].details["unverifiable_episodes"] == [7]
    execute(conn, "UPDATE meta SET value='invalid' WHERE key='draft_coverage:8'")
    assert checks(path)["draft_coverage"].details["invalid_records"] == [8]


def test_chrome_threshold_is_per_app_and_site(gate_db):
    path, conn = gate_db
    for oid in range(2, 7):
        execute(conn, "INSERT INTO observations(id,screen_text,site) VALUES (?,?,?)",
                (oid, "Repeated chrome labels\n" + str(oid), "chrome.example"))
    result = checks(path)["chrome_share"]
    assert not result.passed and result.details["share"] > .3
    execute(conn, "UPDATE observations SET site=CAST(id AS TEXT)")
    assert checks(path)["chrome_share"].status == "n/a"


def test_yaml_assertions_require_completed_typed_actions(gate_db, tmp_path):
    path, conn = gate_db
    expected = tmp_path / "expect.yaml"
    expected.write_text("screen_text_cap: 12000\nclaims:\n"
                        "  - name: downloaded\n    text_regex: 'downloaded.*ChatGPT\\.dmg'\n"
                        "    action: downloaded\n    kind: event\n")
    assert checks(path, expect=expected)["expected:downloaded"].details["claim_ids"] == [1]
    execute(conn, "UPDATE claims SET action=''")
    assert not checks(path, expect=expected)["expected:downloaded"].passed
    execute(conn, "UPDATE claims SET action='downloaded',status='dead'")
    assert not checks(path, expect=expected)["expected:downloaded"].passed


@pytest.mark.parametrize("content", ["[]", "claims: no", "claims: [{}]", "chrome_max_share: 2",
                                    "screen_text_cap: true", "claims: [{name: x, text_regex: x, min_count: 0}]",
                                    "unknown: true"])
def test_invalid_yaml_is_rejected(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_expectations(path)


async def test_cli_registration_exit_codes_and_missing_db_is_not_created(gate_db, tmp_path, capsys):
    path, conn = gate_db
    args = build_parser().parse_args(["--db", str(path), "check"])
    assert await args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["passed"]
    execute(conn, "DELETE FROM evidence")
    assert await args.func(args) == 1
    capsys.readouterr()
    missing = tmp_path / "does-not-exist.db"
    assert await cmd_check(argparse.Namespace(db=str(missing), expect=None, slo=False)) == 2
    assert not missing.exists()
    bad = tmp_path / "bad.yaml"
    bad.write_text("[")
    assert await cmd_check(argparse.Namespace(db=str(path), expect=bad, slo=False)) == 2
    args = build_parser().parse_args(["--db", str(path), "check", "--slo"])
    assert await args.func(args) == 1
    capsys.readouterr()
    execute(conn, "INSERT INTO evidence VALUES (1,1,'Downloaded ChatGPT.dmg')")
    assert await args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    unmeasured = [r for r in report["checks"] if r["name"].startswith("slo:")]
    assert len(unmeasured) == 3
    assert all(r["status"] == "n/a" and r["passed"] is None and r["details"]["reason"] for r in unmeasured)


def test_slos_strict_limits_missing_samples_and_non_pipeline_sources():
    samples = [{"stage": "capture.text", "ms": 1199, "src": "live"},
               {"stage": "capture.text", "ms": 50_000, "src": "eval"},
               {"stage": "enrich.total", "ms": 30_000, "src": "replay"},
               {"stage": "segment.lag", "ms": None, "src": "live"}]
    results = {r.stage: r for r in evaluate_slos(samples)}
    assert results["capture.text"].passed and results["capture.text"].n == 1
    assert not results["enrich.total"].passed and results["enrich.total"].status == "fail"
    assert results["segment.lag"].p90_ms is None and results["segment.lag"].status == "n/a"
    assert results["segment.lag"].passed is None and results["segment.lag"].reason
    assert not evaluate_slos(samples, pipeline_only=False)[0].passed


def test_check_slo_adds_real_stage_gates(gate_db):
    path, conn = gate_db
    for stage, duration in (("capture.text", 500), ("enrich.total", 10_000), ("segment.lag", 59_999)):
        execute(conn, "INSERT INTO perf_samples VALUES (?,?,?)", (stage, duration, "live"))
    assert check_store(path, slo=True).passed
    execute(conn, "UPDATE perf_samples SET ms=60000 WHERE stage='segment.lag'")
    assert not checks(path, slo=True)["slo:segment.lag"].passed


async def test_bench_shows_same_p90_slos_as_checker(store):
    samples = [Sample(stage, duration, at=10 + i, source="live")
               for stage in ("capture.text", "enrich.total", "segment.lag")
               for i, duration in enumerate([100, 200, 300, 1200])]
    await store.record_perf([], samples)
    snapshot = await bench.collect(store, bench.Window(0, 100, "test"))
    assert len(snapshot["slos"]) == 3
    for result in snapshot["slos"]:
        assert result["p90_ms"] == pytest.approx(snapshot["latency_ms"][result["stage"]]["p90"])
    rendered = bench.render(snapshot)
    assert "SLOs" in rendered and "capture.text: PASS" in rendered


async def test_empty_bench_slos_are_rendered_as_na(store):
    snapshot = await bench.collect(store, bench.Window(0, 100, "test"))
    assert all(r["status"] == "n/a" and r["reason"] for r in snapshot["slos"])
    assert "capture.text: N/A" in bench.render(snapshot)


async def test_perf_health_events_flush_into_metrics_and_store_card(store):
    PERF.drain()
    PERF.count("health.capture_ax_empty.viewport_mismatch", 2)
    PERF.count("health.capture_redaction_frame_drops")
    tk.truncate("a" * 4001, 4000)
    await flush_usage(store)
    stats = await store.stats()
    assert stats["silent_loss"]["capture_ax_empty.viewport_mismatch"] == 2
    assert stats["silent_loss"]["capture_redaction_frame_drops"] == 1
    assert stats["silent_loss"]["text_truncated.4000"] == 1
    assert stats["silent_loss"]["draft_window_dropped"] == 0
    assert "capture_redaction_frame_drops=1" in await store_card(store)
    await flush_usage(store)
    assert (await store.stats())["silent_loss"]["capture_redaction_frame_drops"] == 1


async def test_capped_observation_is_flagged_in_same_insert_transaction(store, tmp_db, monkeypatch):
    monkeypatch.setattr(T, "SCREEN_TEXT_MAX_CHARS", 12000)
    oid, new = await store.add_observation(ObservationDraft(screen_text="z" * 12000))
    assert new
    assert await store.get_meta(f"screen_text_truncated:{oid}") == "12000"
    assert (await store.stats())["silent_loss"]["capture_screen_text_truncated"] == 1
    assert checks(tmp_db)["truncation_flagged"].passed


async def test_draft_coverage_is_persistent_idempotent_and_resets_with_input(store):
    await store.record_draft_coverage(7, "first", 0, [1, 2])
    await store.record_draft_coverage(7, "first", 0, [1, 2])
    await store.record_draft_coverage(7, "first", 1, [3])
    stats = await store.stats()
    assert stats["silent_loss"]["draft_chunks"] == 2
    assert stats["silent_loss"]["draft_observations"] == 3
    await store.record_draft_coverage(7, "new", 0, [4])
    record = json.loads(await store.get_meta("draft_coverage:7"))
    assert record == {"input_hash": "new", "chunks": {"0": [4]}}


async def seed_episode(store):
    eid = await store.open_episode("gate", 100)
    ids = []
    for i in range(3):
        oid, _ = await store.add_observation(ObservationDraft(
            screen_text=f"Distinct research topic {i} " + "v" * 2500, window_title=f"Page {i}",
            started_at=100 + i * 10, ended_at=101 + i * 10))
        ids.append(oid)
    await store.assign_episode(ids, eid)
    await store.mark_episode_ready(eid)
    return eid, ids


async def test_successful_drafting_covers_every_chunk_after_checkpoint_cleanup(store, tmp_db, monkeypatch):
    monkeypatch.setattr(T, "DRAFT_MAX_OBS", 2)
    eid, ids = await seed_episode(store)
    pipe = ProcessPipeline(store)
    delivered = []

    async def draft(rows, chrome):
        delivered.extend(r["id"] for r in rows)
        pipe._draft_prompt(rows, chrome, record_losses=True)
        return {"title": "Research", "summary": "", "claims": []}

    monkeypatch.setattr(pipe, "_call_draft", draft)
    PERF.drain()
    assert await pipe.draft_episode(eid) is not None
    assert delivered == ids
    assert await store.q("SELECT * FROM episode_draft_chunks") == []
    assert checks(tmp_db)["draft_coverage"].passed
    await flush_usage(store)
    assert (await store.stats())["silent_loss"]["draft_text_truncated"] == 3


@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_and_cancelled_drafts_increment_durable_counters(store, monkeypatch, cancelled):
    eid, _ = await seed_episode(store)
    monkeypatch.setattr(T, "DRAFT_MAX_ATTEMPTS", 1)
    pipe = ProcessPipeline(store)

    async def fail(_):
        raise asyncio.CancelledError if cancelled else LLMError("offline test failure")

    monkeypatch.setattr(pipe, "_draft", fail)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await pipe.draft_episode(eid)
    else:
        assert await pipe.draft_episode(eid) is None
    counters = (await store.stats())["silent_loss"]
    assert counters["draft_cancelled" if cancelled else "draft_failed"] == 1
    assert counters["draft_abandoned"] == 1


@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_and_abandoned_trails_are_counted(store, monkeypatch, cancelled):
    from discrete.process import trails

    async def actions(*args, **kwargs):
        return [{"id": 7}]

    async def fail(*args, **kwargs):
        raise asyncio.CancelledError if cancelled else LLMError("offline test failure")

    async def record(*args, **kwargs):
        return None

    monkeypatch.setattr(T, "TRAILS", True)
    monkeypatch.setattr(T, "TRAIL_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(store, "actions_without_trail", actions)
    monkeypatch.setattr(store, "record_trail", record)
    monkeypatch.setattr(trails, "trail_for", fail)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await trails.build_trails(store)
    else:
        assert await trails.build_trails(store) == 0
    counters = (await store.stats())["silent_loss"]
    assert counters["trail_cancelled" if cancelled else "trail_failed"] == 1
    assert counters["trail_abandoned"] == (0 if cancelled else 1)


@pytest.mark.parametrize("reason", ["unavailable", "no_pid", "untrusted", "no_window", "viewport_mismatch", "error"])
def test_ax_empty_reason_is_recorded_without_real_capture(monkeypatch, reason):
    from discrete.capture import accessibility as ax

    monkeypatch.setattr(ax, "_HAS_QUARTZ", reason != "unavailable")
    monkeypatch.setattr(ax, "accessibility_granted", lambda: reason != "untrusted")
    monkeypatch.setattr(ax, "AXUIElementCreateApplication", lambda pid: object(), raising=False)
    monkeypatch.setattr(ax, "_enable_full_tree", lambda *args: None)

    def get(element, attribute):
        if reason == "error":
            raise RuntimeError("test AX failure")
        return None if reason == "no_window" else object()

    monkeypatch.setattr(ax, "_ax_get", get)
    monkeypatch.setattr(ax, "_ax_value", lambda *args: object())
    monkeypatch.setattr(ax, "_point_xy", lambda *args: (0, 0))
    monkeypatch.setattr(ax, "_size_wh", lambda *args: (100, 100))
    PERF.drain()
    assert ax.read_screen_text(0 if reason == "no_pid" else 42, max_nodes=10, timeout_ms=10, max_chars=100,
                               viewport=(20, 20, 100, 100)) == ""
    _, samples = PERF.drain()
    assert [s.stage for s in samples] == [f"health.capture_ax_empty.{reason}"]


@pytest.mark.parametrize("reason", ["no_text", "deadline", "node_limit"])
def test_ax_empty_walk_budget_reason(monkeypatch, reason):
    from discrete.capture import accessibility as ax

    monkeypatch.setattr(ax, "_HAS_QUARTZ", True)
    monkeypatch.setattr(ax, "accessibility_granted", lambda: True)
    monkeypatch.setattr(ax, "AXUIElementCreateApplication", lambda pid: object(), raising=False)
    monkeypatch.setattr(ax, "_enable_full_tree", lambda *args: None)
    monkeypatch.setattr(ax, "_ax_get", lambda *args: object())
    monkeypatch.setattr(ax, "_walk_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(T, "AX_TEXT_MARKERS", False)
    PERF.drain()
    assert ax.read_screen_text(42, max_nodes=0 if reason == "node_limit" else 100,
                               timeout_ms=-1 if reason == "deadline" else 1000, max_chars=100) == ""
    _, samples = PERF.drain()
    assert f"health.capture_ax_empty.{reason}" in [s.stage for s in samples]


async def test_redaction_drop_counts_and_saves_no_frame(monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    from discrete.capture import accessibility as ax
    from discrete.capture.redact import SecretMemory
    from discrete.capture.screen import ScreenObserver
    from discrete.config import Endpoint

    capture = ScreenObserver.__new__(ScreenObserver)
    capture.stats = {"redacted": 0, "degraded": 0}
    capture.queue = asyncio.Queue()
    capture.session = SimpleNamespace(session_id="test")
    capture.store = None
    capture._secrets = SecretMemory()
    capture._last_full_text = ""
    capture._last_full_capture = 0
    capture.ep = Endpoint("caption", "", "", "")
    secret_text = "OPENROUTER_API_KEY=sk-or-v1-" + "a" * 64
    monkeypatch.setattr(capture, "_screen_text", lambda *args: (secret_text, "ocr"))
    monkeypatch.setattr(T, "REDACT_SECRETS", True)
    context = ax.WindowContext(app="Browser", url="https://example.com", window_title="Page")
    image = Image.new("RGB", (32, 32), "white")
    PERF.drain()
    await capture._capture({"counts": {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0},
                            "started": 100, "kind": "click"}, context, image, image)
    result = capture.queue.get_nowait()
    assert result.draft.frame_path == "" and secret_text not in result.draft.screen_text
    _, samples = PERF.drain()
    assert [s.n for s in samples if s.stage == "health.capture_redaction_frame_drops"] == [1]


def test_worker_thread_health_events_are_not_lost_during_flush():
    from concurrent.futures import ThreadPoolExecutor

    from discrete.perf import PerfLog

    log = PerfLog()

    def emit():
        for _ in range(5000):
            log.count("health.capture_ax_empty.no_text")

    total = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(emit) for _ in range(2)]
        while not all(worker.done() for worker in workers):
            _, samples = log.drain()
            total += sum(s.n for s in samples)
        for worker in workers:
            worker.result()
    _, samples = log.drain()
    assert total + sum(s.n for s in samples) == 10_000


@pytest.mark.parametrize("source", ["eval", "query", "ask", "bench", "check"])
async def test_bench_and_gate_exclude_the_same_non_pipeline_sources(store, source):
    await store.record_perf([], [Sample("capture.text", 2000, at=10, source=source),
                                Sample("capture.text", 100, at=11, source="live")])
    snapshot = await bench.collect(store, bench.Window(0, 100, "test"))
    assert snapshot["slos"][0]["passed"] and snapshot["slos"][0]["n"] == 1


async def test_bad_regex_returns_cli_configuration_error(gate_db, tmp_path):
    path, _ = gate_db
    bad = tmp_path / "bad-pattern.yaml"
    bad.write_text("claims: [{name: bad, text_regex: '['}]")
    assert await cmd_check(argparse.Namespace(db=str(path), expect=bad, slo=False)) == 2


async def test_legacy_cap_is_flagged_on_new_replay_ingest(store, monkeypatch):
    monkeypatch.setattr(T, "SCREEN_TEXT_MAX_CHARS", 16000)
    oid, _ = await store.add_observation(ObservationDraft(screen_text="a" * 12000))
    assert await store.get_meta(f"screen_text_truncated:{oid}") == "12000"
