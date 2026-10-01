"""Golden-set evaluation (planS3 §10).

A case file is either a list of cases or {snapshot: <store snapshot id>, cases: [...]}.
A case whose file snapshot differs from the store's is reported as stale
rather than scored as a regression (insights.md §4).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import T, golden_dir
from ..retrieval.grounding import evaluate
from ..retrieval.hybrid import retrieve
from ..store import Store


@dataclass
class Case:
    question: str
    category: str = "recall"
    expect_any: list[str] = field(default_factory=list)
    expect_none: list[str] = field(default_factory=list)
    unanswerable: bool = False


@dataclass
class CaseResult:
    case: Case
    rank: int | None  # 1-based rank of the first hit containing an expected string
    mode: str
    latency_ms: float
    contaminated: bool = False

    @property
    def correct(self) -> bool:
        if self.case.unanswerable:
            return self.mode in ("thin", "empty")
        return self.rank is not None and self.rank <= T.EVAL_K


@dataclass
class SuiteResult:
    cases: list[CaseResult] = field(default_factory=list)
    health: dict = field(default_factory=dict)
    stale: bool = False
    snapshot: str = ""

    def metrics(self) -> dict:
        ans = [c for c in self.cases if not c.case.unanswerable]
        unans = [c for c in self.cases if c.case.unanswerable]
        by_cat: dict[str, list[bool]] = {}
        for c in self.cases:
            by_cat.setdefault(c.case.category, []).append(c.correct)
        return {
            "n": len(self.cases),
            "hit_at_k": round(sum(c.correct for c in ans) / len(ans), 3) if ans else None,
            "mrr": round(sum(1 / c.rank for c in ans if c.rank) / len(ans), 3) if ans else None,
            "refusal_accuracy": round(sum(c.correct for c in unans) / len(unans), 3) if unans else None,
            "grounded_rate_answerable": round(sum(c.mode == "grounded" for c in ans) / len(ans), 3) if ans else None,
            "contamination": sum(c.contaminated for c in self.cases),
            "latency_ms_p50": round(statistics.median(c.latency_ms for c in self.cases)) if self.cases else None,
            "by_category": {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()},
        }


def load_cases(path: Path) -> tuple[list[Case], str]:
    if not path.exists():
        return [], ""
    raw = yaml.safe_load(path.read_text()) or []
    snapshot = ""
    if isinstance(raw, dict):
        snapshot = str(raw.get("snapshot") or "")
        raw = raw.get("cases") or []
    cases = [
        Case(
            question=str(item["question"]),
            category=str(item.get("category") or "recall"),
            expect_any=[str(x) for x in item.get("expect_any") or []],
            expect_none=[str(x) for x in item.get("expect_none") or []],
            unanswerable=bool(item.get("unanswerable")),
        )
        for item in raw
    ]
    return cases, snapshot


async def health_checks(store: Store) -> dict:
    s = await store.stats()
    confs = await store.confidence_histogram(decimals=3)
    top_share = max((n for _, n in confs), default=0) / max(1, s["claims_live"])
    return {
        "index_coverage": s["index_coverage"],
        "index_coverage_ok": all(v >= 1.0 for v in s["index_coverage"].values()),
        "outbox_pending": s["outbox_pending"],
        "confidence_top_share": round(top_share, 3),
        "confidence_degenerate": s["claims_live"] >= 10 and top_share > 0.5,
        "lineage_anomalies": await store.lineage_anomalies(),
        "episodes_backlog": {k: v for k, v in s["episodes_by_status"].items() if k != "closed"},
    }


async def run_case(store: Store, case: Case) -> CaseResult:
    t0 = time.perf_counter()
    result = await retrieve(store, case.question)
    decision = evaluate(result)
    latency = (time.perf_counter() - t0) * 1000
    rank = None
    for i, h in enumerate(result.hits, 1):
        text = h.text.lower()
        if any(e.lower() in text for e in case.expect_any):
            rank = i
            break
    blob = "\n".join(h.text.lower() for h in result.hits)
    contaminated = any(bad.lower() in blob for bad in case.expect_none)
    return CaseResult(case=case, rank=rank, mode=decision.mode, latency_ms=latency, contaminated=contaminated)


async def run_suite(store: Store, path: Path) -> SuiteResult:
    cases, snapshot = load_cases(path)
    current = await store.snapshot_id()
    results = [await run_case(store, c) for c in cases]
    return SuiteResult(cases=results, health=await health_checks(store),
                       stale=bool(snapshot) and snapshot != current, snapshot=current)


def baseline_path(cases_path: Path) -> Path:
    return cases_path.with_suffix(".baseline.json")


def save_baseline(result: SuiteResult, cases_path: Path) -> None:
    baseline_path(cases_path).write_text(json.dumps({"snapshot": result.snapshot, **result.metrics()}, indent=2))


def compare(result: SuiteResult, cases_path: Path) -> dict:
    """Regression = a tracked metric dropped by more than EVAL_TOLERANCE."""
    path = baseline_path(cases_path)
    if not path.exists():
        return {"baseline": None, "regressions": []}
    base = json.loads(path.read_text())
    cur = result.metrics()
    regressions = []
    for m in ("hit_at_k", "mrr", "refusal_accuracy"):
        if base.get(m) is not None and cur.get(m) is not None and cur[m] < base[m] - T.EVAL_TOLERANCE:
            regressions.append(f"{m}: {base[m]} → {cur[m]}")
    if len(result.cases) < T.EVAL_MIN_CASES_FOR_SIG:
        note = f"n={len(result.cases)} < {T.EVAL_MIN_CASES_FOR_SIG}: trend only, not significance-gated"
    else:
        note = ""
    return {"baseline": base, "regressions": regressions, "note": note,
            "baseline_snapshot_matches": base.get("snapshot") == result.snapshot}


def default_cases() -> Path:
    return golden_dir() / "cases.yaml"
