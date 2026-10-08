"""Offline regression gates. No migrations, model calls, or writes to the store.

Expectation YAML: screen_text_cap (optional integer), chrome_max_share (0..1),
claims: [{name, text_regex, action?, kind?, min_count?}]. All patterns are
case-insensitive and match live claims. Missing columns or telemetry report n/a.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .. import textkit as tk
from ..store.health import DraftCoverage
from .slos import evaluate_slos

SCREEN_TEXT_CAP = 16_000
LEGACY_SCREEN_TEXT_CAP = 12_000
CHROME_MAX_SHARE = 0.30
CHROME_RECURRENCE = 0.30
CHROME_MIN_CAPTURES = 5
TEXT_FIELDS = ("screen_text", "caption", "window_title", "page_title")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool | None
    details: dict[str, Any] = field(default_factory=dict)
    status: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", "n/a" if self.passed is None else "pass" if self.passed else "fail")


@dataclass(frozen=True)
class CheckReport:
    store: str
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.passed is not False for check in self.checks)

    @property
    def release_passed(self) -> bool:
        return bool(self.checks) and all(check.passed is True for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed, "release_passed": self.release_passed}


@dataclass(frozen=True)
class ExpectedClaim:
    name: str
    text_regex: str
    action: str | None = None
    kind: str | None = None
    min_count: int = 1


@dataclass(frozen=True)
class Expectations:
    claims: list[ExpectedClaim] = field(default_factory=list)
    screen_text_cap: int | None = None
    chrome_max_share: float = CHROME_MAX_SHARE


def load_expectations(path: str | Path | None) -> Expectations:
    if path is None:
        return Expectations()
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {"claims", "screen_text_cap", "chrome_max_share"}:
        raise ValueError("expectations must be a mapping with claims, screen_text_cap, chrome_max_share")
    claims = raw.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    expected = []
    for item in claims:
        if not isinstance(item, dict):
            raise ValueError("each expected claim must be a mapping")
        try:
            claim = ExpectedClaim(**item)
        except TypeError as exc:
            raise ValueError(f"invalid expected claim: {exc}") from exc
        if (not isinstance(claim.name, str) or not claim.name or not isinstance(claim.text_regex, str)
                or not claim.text_regex or type(claim.min_count) is not int or claim.min_count < 1
                or claim.action is not None and not isinstance(claim.action, str)
                or claim.kind is not None and not isinstance(claim.kind, str)):
            raise ValueError("claims need nonempty name/text_regex, positive min_count and string action/kind")
        try:
            re.compile(claim.text_regex)
        except re.error as exc:
            raise ValueError(f"invalid text_regex for {claim.name}: {exc}") from exc
        expected.append(claim)
    cap = raw.get("screen_text_cap")
    share = raw.get("chrome_max_share", CHROME_MAX_SHARE)
    if cap is not None and (type(cap) is not int or cap < 2):
        raise ValueError("screen_text_cap must be an integer >= 2")
    if type(share) not in (int, float) or not 0 <= share <= 1:
        raise ValueError("chrome_max_share must be between 0 and 1")
    return Expectations(expected, cap, share)


def _require_columns(name: str, schema: dict[str, set[str]], required: dict[str, set[str]]) -> CheckResult | None:
    missing = {table: sorted(columns - schema.get(table, set())) for table, columns in required.items()
               if not columns <= schema.get(table, set())}
    if missing:
        return CheckResult(name, None, {"reason": "required store tables/columns are absent", "missing_columns": missing})
    return None


def _coverage(conn: sqlite3.Connection, meta: dict[str, str]) -> CheckResult:
    missing: dict[int, list[int]] = {}
    unknown = []
    invalid = []
    drafted = conn.execute("SELECT id FROM episodes WHERE status='closed' AND draft_attempts>0").fetchall()
    for episode in drafted:
        eid = episode["id"]
        ids = {r[0] for r in conn.execute("SELECT id FROM observations WHERE episode_id=? AND origin='user'", (eid,))}
        if not ids:
            continue
        if f"draft_coverage:{eid}" not in meta:
            unknown.append(eid)
            continue
        try:
            record = DraftCoverage.from_json(meta[f"draft_coverage:{eid}"])
            # These ids are recorded only after a chunk response arrives (or is reused).
            reached = {oid for chunk in record.chunks.values() for oid in chunk}
        except (ValueError, TypeError, AttributeError):
            invalid.append(eid)
            continue
        if dropped := sorted(ids - reached):
            missing[eid] = dropped
    details = {"drafted_episodes": len(drafted), "missing_observations": missing,
               "unverifiable_episodes": unknown, "invalid_records": invalid}
    if unknown:
        details["reason"] = "draft input telemetry is absent for some episodes; coverage cannot be inferred from claims"
    # Recorded omissions still fail even if other episodes lack legacy telemetry.
    return CheckResult("draft_coverage", False if missing or invalid else None if unknown else True, details)


def _truncation(rows: list[dict[str, Any]], meta: dict[str, str], caps: list[int]) -> CheckResult:
    at_cap = [r["id"] for r in rows if any(cap - 1 <= len(r["screen_text"]) <= cap for cap in caps)
              or len(r["screen_text"]) > max(caps)]
    flagged = [oid for oid in at_cap if meta.get(f"screen_text_truncated:{oid}")]
    # The first instrumented row marks where a missing flag becomes a violation.
    # Earlier rows remain unmeasured even when new captures are added to an old DB.
    first = int(meta["screen_text_flags_from"]) if "screen_text_flags_from" in meta else None
    unflagged = sorted(set(at_cap) - set(flagged))
    unknown = [oid for oid in unflagged if first is None or oid < first]
    violations = sorted(set(unflagged) - set(unknown))
    details = {"caps": caps, "at_cap": at_cap, "flagged": flagged, "unflagged": violations,
               "unverifiable_observations": unknown}
    unmeasured = bool(unknown) or first is None and not flagged
    if unmeasured:
        details["reason"] = "capture truncation telemetry is absent for legacy observations"
    return CheckResult("truncation_flagged", False if violations else None if unmeasured else True, details)


def _chrome(rows: list[dict[str, Any]], maximum: float) -> CheckResult:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["origin"] == "user" and row["screen_text"].strip():
            groups[(row["app"], row["site"])].append(row)
    total = repeated = covered = 0
    for group in groups.values():
        chrome = tk.chrome_lines([r["screen_text"] for r in group], share=CHROME_RECURRENCE,
                                 min_texts=CHROME_MIN_CAPTURES)
        if len(group) >= CHROME_MIN_CAPTURES:
            covered += len(group)
        for row in group:
            for line in row["screen_text"].splitlines():
                if not line.strip():
                    continue
                total += len(line)
                if tk.norm_key(line) in chrome:
                    repeated += len(line)
    share = repeated / total if total else 0.0
    return CheckResult("chrome_share", share < maximum if covered else None,
                       {"share": share, "threshold": maximum, "covered_rows": covered,
                        "text_rows": sum(map(len, groups.values())),
                        **({"reason": "not enough captures per app/site to measure recurring chrome"} if not covered else {}),
                        "note": "raw text recurrence per (app, site); persistent content also counts"})


def check_store(path: str | Path, *, expect: str | Path | None = None, slo: bool = False, release: bool = False) -> CheckReport:
    """Return structured pass/fail/n/a checks from one consistent read-only snapshot.

    `slo=True` adds all three latency gates. Absent measurements are n/a.
    Rows within one character of a cap are flagged conservatively (truncate's ellipsis).
    """
    expected = load_expectations(expect)
    path = Path(path).expanduser().resolve()
    from ..encryption import open_database

    conn = open_database(str(path), read_only=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        schema = {table: {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
                  for table in ("observations", "claims", "evidence", "episodes", "meta", "perf_samples")}
        rows = [dict(r) for r in conn.execute("SELECT * FROM observations")] if "observations" in tables else []
        by_id = {r["id"]: r for r in rows} if "id" in schema["observations"] else {}
        claims = ([dict(r) for r in conn.execute("SELECT * FROM claims WHERE status='live'")]
                  if "status" in schema["claims"] else [])
        evidence: dict[int, list[sqlite3.Row]] = defaultdict(list)
        if {"claim_id", "observation_id"} <= schema["evidence"]:
            for row in conn.execute("SELECT * FROM evidence"):
                evidence[row["claim_id"]].append(row)
        evidence_columns = {"claims": {"id", "status"}, "evidence": {"claim_id", "observation_id"},
                            "observations": {"id"}}
        self_unavailable = _require_columns("no_self_evidence", schema,
                                            {**evidence_columns, "observations": {"id", "origin"}})
        quote_unavailable = _require_columns("grounded_evidence", schema,
                                             {**evidence_columns, "evidence": {"claim_id", "observation_id", "quote"},
                                              "observations": {"id", *TEXT_FIELDS}})
        self_cites = []
        ungrounded = []
        bad_quotes = []
        for claim in claims:
            if "id" not in claim:
                continue
            authentic = bool(evidence[claim["id"]])
            for ev in evidence[claim["id"]]:
                obs = by_id.get(ev["observation_id"])
                if obs is None:
                    authentic = False
                    bad_quotes.append({"claim_id": claim["id"], "observation_id": ev["observation_id"]})
                    continue
                if self_unavailable is None and obs["origin"] != "user":
                    self_cites.append({"claim_id": claim["id"], "observation_id": obs["id"], "origin": obs["origin"]})
                if quote_unavailable is None:
                    quote = tk.norm_key(ev["quote"] or "")
                    if not quote or not any(quote in tk.norm_key(obs.get(f) or "") for f in TEXT_FIELDS):
                        authentic = False
                        bad_quotes.append({"claim_id": claim["id"], "observation_id": obs["id"]})
            if not authentic:
                ungrounded.append(claim["id"])
        meta = dict(conn.execute("SELECT key, value FROM meta")) if {"key", "value"} <= schema["meta"] else {}
        coverage_unavailable = _require_columns("draft_coverage", schema,
                                               {"episodes": {"id", "status", "draft_attempts"},
                                                "observations": {"id", "episode_id", "origin"},
                                                "meta": {"key", "value"}})
        checks = [self_unavailable or CheckResult("no_self_evidence", not self_cites, {"violations": self_cites}),
                  quote_unavailable or CheckResult("grounded_evidence", not ungrounded,
                                                   {"claim_ids": ungrounded, "violations": bad_quotes}),
                  coverage_unavailable or _coverage(conn, meta)]
        # Capture caps changed between the golden sessions; an explicit YAML value
        # removes ambiguity. In its absence check both known historical boundaries.
        caps = [expected.screen_text_cap] if expected.screen_text_cap else [LEGACY_SCREEN_TEXT_CAP, SCREEN_TEXT_CAP]
        checks.append(_require_columns("truncation_flagged", schema,
                                       {"observations": {"id", "screen_text"}, "meta": {"key", "value"}})
                      or _truncation(rows, meta, caps))
        checks.append(_require_columns("chrome_share", schema,
                                       {"observations": {"origin", "app", "site", "screen_text"}})
                      or _chrome(rows, expected.chrome_max_share))
        for assertion in expected.claims:
            name = f"expected:{assertion.name}"
            columns = {"id", "text", "status"}
            if assertion.action is not None:
                columns.add("action")
            if assertion.kind is not None:
                columns.add("kind")
            unavailable = _require_columns(name, schema, {"claims": columns})
            if unavailable is not None:
                checks.append(unavailable)
                continue
            pattern = re.compile(assertion.text_regex, re.I | re.S)
            matched = [c["id"] for c in claims if pattern.search(c["text"])
                       and (assertion.action is None or c["action"] == assertion.action)
                       and (assertion.kind is None or c["kind"] == assertion.kind)]
            checks.append(CheckResult(name, len(matched) >= assertion.min_count,
                                      {"claim_ids": matched, "min_count": assertion.min_count}))
        if release:
            from ..store.index import _LIVE

            required = {"embeddings": {"ref_kind", "ref_id"}, "index_outbox": {"ref_kind", "ref_id"}}
            for table in required:
                schema[table] = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            unavailable = _require_columns("index_ready", schema, required)
            if unavailable is not None or any(table not in tables for table, _ in _LIVE.values()):
                checks.append(unavailable or CheckResult("index_ready", None, {"reason": "index sources absent"}))
            else:
                missing = {}
                for kind, (table, live) in _LIVE.items():
                    count = conn.execute(f"SELECT COUNT(*) FROM {table} t WHERE {live} AND NOT EXISTS "
                                         "(SELECT 1 FROM embeddings e WHERE e.ref_kind=? AND e.ref_id=t.id)",
                                         (kind,)).fetchone()[0]
                    if count:
                        missing[kind] = count
                pending = conn.execute("SELECT COUNT(*) FROM index_outbox").fetchone()[0]
                checks.append(CheckResult("index_ready", not missing and not pending,
                                          {"missing": missing, "outbox_pending": pending}))
        if slo or release:
            unavailable = _require_columns("slo", schema, {"perf_samples": {"stage", "ms", "src"}})
            samples = conn.execute("SELECT stage, ms, src FROM perf_samples").fetchall() if unavailable is None else []
            if release:
                samples = [sample for sample in samples if sample["src"] == "live"]
            for result in evaluate_slos(samples):
                details = asdict(result)
                if unavailable is not None:
                    details.update(unavailable.details)
                checks.append(CheckResult(f"slo:{result.stage}", result.passed, details))
        return CheckReport(str(path), checks)
    finally:
        conn.close()
