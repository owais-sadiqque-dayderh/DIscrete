"""Golden scoring failures are infrastructure, not accuracy; never call a model."""

from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from bench import run_g46 as run


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must not call a provider")

    monkeypatch.setattr(run.urllib.request, "urlopen", forbidden)
    resolver = run._endpoint
    resolver.cache_clear()
    yield
    resolver.cache_clear()


class FakeProcess:
    pid = 123456

    def __init__(self, stdout="", stderr="", returncode=0, timeout=False):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timeout = timeout
        self.communications = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def communicate(self, timeout=None):
        self.communications.append(timeout)
        if self.timeout and len(self.communications) == 1:
            raise subprocess.TimeoutExpired("fake ask", timeout, output=b"partial answer")
        return self.stdout, self.stderr


def processes(monkeypatch, *items):
    queue = iter(items)
    calls = []

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return next(queue)

    monkeypatch.setattr(run.subprocess, "Popen", popen)
    return calls


def verdict(result, model):
    result.supported = result.answers_question = True
    result.judge_reason = "supported"


def test_timeout_kills_group_discards_partial_answer_and_continues(monkeypatch, tmp_path):
    timed_out = FakeProcess("partial", timeout=True)
    calls = processes(monkeypatch, timed_out, FakeProcess("answer"))
    kills = []
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    first = run.ask(run.Case("slow", "recall"), tmp_path / "copy.db", tmp_path, 0)
    second = run.ask(run.Case("next", "recall"), tmp_path / "copy.db", tmp_path, 0)
    assert first.error == "timeout" and first.infra_error and first.answer == ""
    assert first.answer_ms is not None
    assert second.answer == "answer" and not second.infra_error
    assert kills == [(timed_out.pid, signal.SIGKILL)]
    assert timed_out.communications == [300, None] and timed_out.closed
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"]["PYTHONPATH"] == str(tmp_path)
    assert "--records" in calls[0][0]


def test_timeout_handles_already_exited_group(monkeypatch, tmp_path):
    processes(monkeypatch, FakeProcess(timeout=True))

    def vanished(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(run.os, "killpg", vanished)
    assert run.ask(run.Case("q", "recall"), tmp_path / "db", tmp_path, 0).error == "timeout"


@pytest.mark.parametrize("stderr,answer,returncode,infra", [
    ("Error code: 429 rate limit", "", 0, True),
    ("429 Too Many Requests", "", 0, True),
    ("503", "", 0, True),
    ("HTTP 500 Internal Server Error", "partial", 0, True),
    ("Error code: 503 service_overloaded", "", 0, True),
    ("HTTP status 599", "partial", 0, True),
    ("provider service_overloaded", "partial", 0, True),
    ("request timed out", "", 0, True),
    ("LLMError: writer model failed", "", 0, True),
    ("writer failed: invalid API key", "", 0, True),
    ("Provider returned error: no routes", "", 0, True),
    ("", "", 1, True),
    ("Traceback: unexpected local error", "", 1, True),
    ("[1.0s] done (retrieve 0.5s, first chunk +0.2s)", "answer", 0, False),
    ("A record with 503 widgets", "answer", 0, False),
    ("", "", 0, False),
])
def test_answer_failure_detection(monkeypatch, tmp_path, stderr, answer, returncode, infra):
    processes(monkeypatch, FakeProcess(answer, stderr, returncode))
    result = run.ask(run.Case("q", "recall"), tmp_path / "db", tmp_path, 0)
    assert result.infra_error is infra
    assert bool(result.error) is infra


def test_successful_answer_retains_existing_metrics(monkeypatch, tmp_path):
    processes(monkeypatch, FakeProcess("answer", "record needle\n[1.0s] done (retrieve 0.5s, first chunk +0.2s)"))
    result = run.ask(run.Case("q", "recall", expect_any=["needle"]), tmp_path / "db", tmp_path)
    assert result.rank == 1 and result.records == ["record needle"]
    assert result.retrieval_ms == 500 and result.first_chunk_ms == 700


@pytest.mark.parametrize("failed", [True, False])
def test_outage_mentions_in_records_are_not_provider_diagnostics(monkeypatch, tmp_path, failed):
    stderr = "[0.5s] mode=ask intent=recall\nThe user saw HTTP 503 service_overloaded and timed out.\n"
    stderr += ("Traceback (most recent call last):\nopenai.APIStatusError: Error code: 503" if failed else
               "[1.0s] done (retrieve 0.5s, first chunk +0.2s)")
    processes(monkeypatch, FakeProcess("partial" if failed else "answer", stderr))
    result = run.ask(run.Case("q", "recall"), tmp_path / "db", tmp_path, 0)
    assert result.infra_error is failed


def test_timeout_can_recover_on_retry(monkeypatch, tmp_path):
    processes(monkeypatch, FakeProcess(timeout=True), FakeProcess("recovered"))
    kills = []
    sleeps = []
    monkeypatch.setattr(run.os, "killpg", lambda *args: kills.append(args))
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    result = run.ask(run.Case("q", "recall"), tmp_path / "db", tmp_path)
    assert result.answer == "recovered" and result.attempts == 2
    assert not result.infra_error and result.error == ""
    assert len(kills) == 1 and sleeps == [2]


@pytest.mark.parametrize("recover", [True, False])
def test_answer_retries_back_off_and_stop(monkeypatch, tmp_path, recover):
    calls = processes(monkeypatch, FakeProcess(stderr="HTTP 503"), FakeProcess(stderr="HTTP 503"),
                      FakeProcess("answer") if recover else FakeProcess(stderr="HTTP 503"))
    sleeps = []
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    result = run.ask(run.Case("q", "recall"), tmp_path / "db", tmp_path)
    assert len(calls) == result.attempts == 3
    assert sleeps == [2, 4]
    assert result.infra_error is not recover
    assert result.error == ("" if recover else "HTTP 503")


@pytest.mark.parametrize("recover", [True, False])
def test_judge_failure_retries_without_reasking(monkeypatch, recover):
    result = run.CaseResult("q", "recall", False, answer="answer")
    calls = []
    sleeps = []

    def chat(model, prompt):
        calls.append(prompt)
        if recover and len(calls) == 3:
            return {"supported": True, "answers_question": True, "reason": "ok"}
        return {"reason": "judge error: HTTP 503"}

    monkeypatch.setattr(run, "_chat_json", chat)
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    run.judge_with_retries(result, "fake", 2)
    assert result.judge_attempts == len(calls) == 3
    assert result.answer == "answer" and result.attempts == 1
    assert sleeps == [2, 4] and result.infra_error is not recover
    assert result.error == ("" if recover else "judge error: HTTP 503")


def test_judge_skips_answer_infrastructure_and_accepts_negative_verdict(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "_chat_json", lambda *args: calls.append(args) or
                        {"supported": False, "answers_question": False, "reason": "wrong"})
    failed = run.CaseResult("failed", "recall", False, infra_error=True, error="timeout")
    wrong = run.CaseResult("wrong", "recall", False, answer="bad")
    run.judge_with_retries(failed, "fake", 2)
    run.judge_with_retries(wrong, "fake", 2)
    assert len(calls) == 1 and wrong.judge_attempts == 1
    assert not wrong.correct and not wrong.infra_error
    assert failed.error == "timeout" and failed.judge_attempts == 0


def test_judge_exception_is_an_infrastructure_failure(monkeypatch):
    def failed(*args):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(run, "judge", failed)
    result = run.CaseResult("q", "recall", False)
    run.judge_with_retries(result, "fake", 0)
    assert result.infra_error and result.error == "judge error: judge timed out"


@pytest.mark.parametrize("verdict", [{}, {"supported": "true", "answers_question": True}])
def test_malformed_judge_verdict_is_infrastructure(monkeypatch, verdict):
    monkeypatch.setattr(run, "_chat_json", lambda *args: verdict)
    result = run.CaseResult("q", "recall", False)
    run.judge_with_retries(result, "fake", 0)
    assert result.infra_error


def test_summary_excludes_infrastructure_from_all_accuracy_denominators():
    good = run.CaseResult("good", "recall", False, rank=2, supported=True, answers_question=True, strings_ok=True)
    bad = run.CaseResult("bad", "recall", False, supported=False, strings_ok=False)
    refusal = run.CaseResult("refusal", "unanswerable", True, supported=True, answers_question=True, strings_ok=True)
    failed = run.CaseResult("failed", "recall", False, infra_error=True, error="timeout")
    failed_refusal = run.CaseResult("failed refusal", "unanswerable", True, infra_error=True)
    summary = run.summarize([good, bad, refusal, failed, failed_refusal])
    assert summary["n"] == 5 and summary["n_scored"] == 3
    assert summary["infra_errors"] == 2 and summary["infra_error_rate"] == 0.4
    assert summary["answer_correct"] == summary["hit_at_k"] == 0.5
    assert summary["mrr"] == 0.25 and summary["refusal_correct"] == 1
    assert summary["overall_correct"] == 0.667
    assert summary["by_category"] == {"recall": "1/2", "unanswerable": "1/1"}
    empty = run.summarize([failed])
    assert empty["overall_correct"] is None and empty["by_category"] == {}


def endpoint(name="writer") -> run.ModelEndpoint:
    return run.ModelEndpoint(name, "https://fake.invalid/v1", "fake-key", "fake")


@pytest.mark.parametrize("fail_at", [None, "writer", "judge"])
def test_global_preflight_probes_each_actual_endpoint_once(monkeypatch, tmp_path, fail_at):
    configs = []
    probes = []

    def config(code, model=None):
        configs.append((code, model))
        return endpoint("judge" if model else "writer")

    def probe(ep):
        probes.append(ep.name)
        if ep.name == fail_at:
            raise RuntimeError("HTTP 503")

    monkeypatch.setattr(run, "_endpoint", config)
    monkeypatch.setattr(run, "_probe_endpoint", probe)
    if fail_at:
        with pytest.raises(RuntimeError, match="endpoint preflight failed.*503"):
            run.preflight(tmp_path, "fake-judge")
    else:
        run.preflight(tmp_path, "fake-judge")
    assert probes == (["writer"] if fail_at == "writer" else ["writer", "judge"])
    assert configs[0] == (tmp_path, None)
    if fail_at != "writer":
        assert configs[1] == (run.ROOT, "fake-judge")


@pytest.mark.parametrize("response", [{"choices": [{"message": {"content": ""}}]}, {"error": "overloaded"}, {}])
def test_probe_is_one_small_bounded_request(monkeypatch, response):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(response).encode()

    def open_response(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(run.urllib.request, "urlopen", open_response)
    if response.get("choices"):
        run._probe_endpoint(endpoint())
    else:
        with pytest.raises(RuntimeError, match="provider returned"):
            run._probe_endpoint(endpoint())
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "https://fake.invalid/v1/chat/completions" and timeout == 30
    assert json.loads(request.data)["max_tokens"] == 16


def test_probe_http_failure_is_not_retried(monkeypatch):
    calls = []

    def down(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 503, "overloaded", {}, None)

    monkeypatch.setattr(run.urllib.request, "urlopen", down)
    with pytest.raises(HTTPError):
        run._probe_endpoint(endpoint())
    assert len(calls) == 1


def test_unconfigured_endpoint_does_not_issue_request():
    ep = replace(endpoint(), api_key=None)
    with pytest.raises(RuntimeError, match="writer endpoint is not configured"):
        run._probe_endpoint(ep)


@pytest.mark.parametrize("response", [
    {"choices": [{"message": {"content": '{"supported":false,"answers_question":false}'}}]},
    {"choices": [{"message": {"content": ""}}]}, {"error": {"code": 503}},
])
def test_judge_transport_failure_is_reported_without_hidden_retries(monkeypatch, response):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(response).encode()

    def open_response(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(run, "_endpoint", lambda *args: endpoint("judge"))
    monkeypatch.setattr(run.urllib.request, "urlopen", open_response)
    result = run._chat_json("fake", "judge this")
    assert len(calls) == 1 and calls[0][1] == 120
    if response.get("choices", [{}])[0].get("message", {}).get("content"):
        assert result == {"supported": False, "answers_question": False}
    else:
        assert result["reason"].startswith("judge error:")


def test_endpoint_resolution_uses_selected_checkout_config(monkeypatch, tmp_path):
    calls = []

    def config(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps(asdict(endpoint())), returncode=0)

    monkeypatch.setattr(run.subprocess, "run", config)
    assert run._endpoint(tmp_path) == endpoint()
    assert run._endpoint(tmp_path) == endpoint()  # Resolution is cached across workers.
    assert len(calls) == 1
    assert "writer_endpoint" in calls[0][0][2]
    assert calls[0][1]["cwd"] == tmp_path and calls[0][1]["env"]["PYTHONPATH"] == str(tmp_path)


def setup_main(monkeypatch, tmp_path, cases, *extra):
    db = tmp_path / "input.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE perf_calls (cost REAL, at REAL)")
    results = tmp_path / "results"
    monkeypatch.setattr(run, "RESULTS", results)
    monkeypatch.setattr(run, "load_cases", lambda path: cases)
    monkeypatch.setattr(run, "preflight", lambda code, model: None)
    monkeypatch.setattr(run, "judge", verdict)
    monkeypatch.setattr(sys, "argv", ["run_g46.py", "--db", str(db), "--name", "fake", *extra])
    return db, results


@pytest.mark.parametrize("failures,total,limit,valid", [(1, 10, 0.1, True), (2, 10, 0.1, False),
                                                       (0, 1, 0.1, True), (1, 1, 1.0, False)])
def test_main_writes_compatible_json_and_returns_validity(monkeypatch, tmp_path, capsys, failures, total, limit, valid):
    cases = [run.Case(f"q{i}", "recall") for i in range(total)]
    db, results = setup_main(monkeypatch, tmp_path, cases, "--max-infra-errors", str(limit), "--infra-retries", "0")
    processes(monkeypatch, *(FakeProcess(timeout=True) if i < failures else FakeProcess("answer") for i in range(total)))
    monkeypatch.setattr(run.os, "killpg", lambda *args: None)
    assert run.main() == (0 if valid else 1)
    saved = json.loads((results / "fake.json").read_text())
    assert saved["valid"] is valid and saved["summary"]["infra_errors"] == failures
    assert saved["summary"]["n"] == total and saved["summary"]["n_scored"] == total - failures
    assert saved["db"] == str(db) and saved["name"] == "fake"
    assert saved["cases"][0]["error"] == ("timeout" if failures else "")
    assert all(r["judge_attempts"] == (0 if r["infra_error"] else 1) for r in saved["cases"])
    output = capsys.readouterr()
    assert ("INVALID run" in output.err) is not valid


def test_main_preflight_aborts_before_copy_or_scoring(monkeypatch, tmp_path, capsys):
    _, results = setup_main(monkeypatch, tmp_path, [run.Case("q", "recall")])

    def down(*args):
        raise RuntimeError("answer endpoint HTTP 503 overloaded")

    monkeypatch.setattr(run, "preflight", down)
    assert run.main() == 1 and not results.exists()
    assert "Preflight failed; scoring aborted" in capsys.readouterr().err


def test_main_marks_judge_outage_invalid_and_excludes_accuracy(monkeypatch, tmp_path):
    _, results = setup_main(monkeypatch, tmp_path, [run.Case("q", "recall")], "--infra-retries", "0")
    processes(monkeypatch, FakeProcess("answer"))

    def down(*args):
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(run, "judge", down)
    assert run.main() == 1
    saved = json.loads((results / "fake.json").read_text())
    assert saved["valid"] is False and saved["summary"]["overall_correct"] is None
    assert saved["cases"][0]["infra_error"] and saved["cases"][0]["answer"] == "answer"


@pytest.mark.parametrize("extra", [["--infra-retries", "-1"], ["--jobs", "0"],
                                   ["--max-infra-errors", "1.1"], ["--max-infra-errors", "nan"]])
def test_cli_rejects_invalid_limits_before_preflight(monkeypatch, tmp_path, extra):
    setup_main(monkeypatch, tmp_path, [run.Case("q", "recall")], *extra)
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2


def test_compare_handles_legacy_and_invalid_runs_without_scores(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    row = run.CaseResult("q", "recall", False, supported=True, answers_question=True, strings_ok=True)
    legacy = {"summary": run.summarize([row]), "cases": [asdict(row) | {"correct": True}]}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    invalid = legacy | {"valid": False, "summary": legacy["summary"] | {"overall_correct": 0.123}}
    (tmp_path / "invalid.json").write_text(json.dumps(invalid))
    run.compare(["legacy", "invalid"])
    output = capsys.readouterr().out
    assert "INVALID" in output and "0.123" not in output
    assert "✓" in output and "overall_correct" in output


def test_compare_marks_excluded_case_as_infra(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    row = run.CaseResult("q", "recall", False, infra_error=True, error="timeout")
    data = {"valid": True, "summary": run.summarize([row]), "cases": [asdict(row) | {"correct": False}]}
    (tmp_path / "partial.json").write_text(json.dumps(data))
    run.compare(["partial"])
    assert "INFRA" in capsys.readouterr().out


@pytest.mark.parametrize("cached,valid,exitcode", [(False, False, 1), (False, False, 0),
                                                  (True, False, 0), (True, True, 0)])
def test_matrix_stops_on_failed_or_cached_invalid_runs(tmp_path, cached, valid, exitcode):
    """Execute only score(), in a disposable tree with a fake runner."""
    root = tmp_path / "matrix"
    bench = root / "bench"
    results = bench / "private/results"
    results.mkdir(parents=True)
    venv = root / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    fake = bench / "run_g46.py"
    fake.write_text(f"""
import json
from pathlib import Path
Path('bench/private/results/fake.json').write_text(json.dumps({{'valid': {valid!r}}}))
print('fake provider status')
raise SystemExit({exitcode})
""")
    if cached:
        (results / "fake.json").write_text(json.dumps({"valid": valid}))
    script = (run.HERE / "run_matrix.sh").read_text().split("\nmkdir -p $P/results", 1)[0]
    script += "\nscore fake ignored ignored ignored\necho MATRIX_CONTINUED\n"
    # $0 gives the real script the disposable bench directory as its base.
    process = subprocess.run(["bash", "-c", script, str(bench / "run_matrix.sh")],
                             capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    expected_success = cached and valid
    assert process.returncode == (0 if expected_success else 1)
    assert ("MATRIX_CONTINUED" in process.stdout) is expected_success
    if not expected_success:
        assert "matrix stopped" in process.stderr


def test_an_answered_question_is_never_an_infrastructure_failure():
    """DIscrete logs 'support verification timed out' as a warning and still answers."""
    case = run.Case(question="Why did I open ChatGPT?", category="why")
    stderr = "WARNING discrete.retrieval.support retrieval support verification timed out; results left unverified\n"
    assert run._infra_reason(stderr, "You asked what the repo is about.", 0, case) == ""
    # An empty answer with a provider error is still infrastructure.
    assert run._infra_reason("openai.APIStatusError: 503 service_overloaded", "", 0, case)
