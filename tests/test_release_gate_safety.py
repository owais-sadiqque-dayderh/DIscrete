import argparse
from contextlib import asynccontextmanager

from discrete import cli
from discrete.evaluation.harness import Case, CaseResult, SuiteResult, compare, save_baseline
from discrete.evaluation.invariants import CheckReport, CheckResult
from discrete.models import ObservationDraft


def test_unmeasured_release_gate_fails_but_legacy_audit_stays_available():
    report = CheckReport('synthetic', [CheckResult('slo:capture.text', None, {'reason': 'no samples'})])
    assert report.passed
    assert not report.release_passed


async def test_stale_eval_cannot_save_successful_baseline(monkeypatch, tmp_path):
    result = SuiteResult(cases=[CaseResult(Case('test', expect_any=['test']), 1, 'grounded', 1)], stale=True)
    @asynccontextmanager
    async def session():
        yield object()
    async def run(*args, **kwargs):
        return result
    monkeypatch.setattr(cli, '_session', session)
    monkeypatch.setattr(cli, 'run_suite', run)
    path = tmp_path / 'cases.yaml'
    args = argparse.Namespace(cases=str(path), answers=False, verbose=False, save_baseline=True)
    assert await cli.cmd_eval(args) != 0
    assert not path.with_suffix('.baseline.json').exists()


def test_baseline_snapshot_mismatch_is_incomparable(tmp_path):
    path = tmp_path / 'cases.yaml'
    result = SuiteResult(cases=[CaseResult(Case('test'), 1, 'grounded', 1)], snapshot='old')
    save_baseline(result, path)
    result.snapshot = 'new'
    comparison = compare(result, path)
    assert comparison['incomparable']
    assert not comparison['regressions']


async def test_snapshot_changes_when_content_or_config_changes_without_counts(store, monkeypatch):
    from discrete.config import T
    oid, _ = await store.add_observation(ObservationDraft(
        app='test', caption='synthetic', screen_text='one', started_at=100, ended_at=101))
    first = await store.snapshot_id()
    await store.execute('UPDATE observations SET screen_text=? WHERE id=?', ('two', oid))
    second = await store.snapshot_id()
    assert first != second
    monkeypatch.setattr(T, 'TOP_K', T.TOP_K + 1)
    assert second != await store.snapshot_id()
    # Recording this evaluation's telemetry does not invalidate its input snapshot.
    third = await store.snapshot_id()
    await store.bump_metric('synthetic-counter')
    assert third == await store.snapshot_id()
