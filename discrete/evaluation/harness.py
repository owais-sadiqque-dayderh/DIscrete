"""Golden-set evaluation (planS3 §10)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import T, golden_dir
from ..retrieval.grounding import evaluate
from ..retrieval.search import retrieve
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
    hit: bool
    mode: str
    latency_ms: float
    contaminated: bool = False


@dataclass
class SuiteResult:
    cases: list[CaseResult] = field(default_factory=list)
    health: dict = field(default_factory=dict)

    @property
    def hit_at_k(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.hit) / len(self.cases)

    @property
    def refusal_accuracy(self) -> float:
        unans = [c for c in self.cases if c.case.unanswerable]
        if not unans:
            return 1.0
        return sum(1 for c in unans if c.mode in ("empty", "thin")) / len(unans)

    def by_category(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for cr in self.cases:
            buckets.setdefault(cr.case.category, []).append(cr.hit)
        return {k: sum(v) / len(v) for k, v in buckets.items()}


def load_cases(path: Path | None = None) -> list[Case]:
    path = path or golden_dir() / "cases.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    out: list[Case] = []
    for item in raw:
        out.append(
            Case(
                question=str(item["question"]),
                category=str(item.get("category") or "recall"),
                expect_any=list(item.get("expect_any") or []),
                expect_none=list(item.get("expect_none") or []),
                unanswerable=bool(item.get("unanswerable")),
            )
        )
    return out


async def health_checks(store: Store) -> dict:
    stats = await store.stats()
    obs, embeds, outbox = stats["observations"], stats["embeddings"], stats["outbox_pending"]
    coverage = 1.0 if obs == 0 else min(1.0, embeds / max(obs, 1))
    claims = await store.live_claims()
    conf_values = [r["confidence"] for r in claims]
    top_conf = max((conf_values.count(v) for v in set(conf_values)), default=0)
    conf_degen = (top_conf / len(conf_values)) if conf_values else 0.0
    return {
        "index_coverage_ok": outbox == 0 and coverage >= 0.95,
        "embedding_coverage": coverage,
        "outbox_pending": outbox,
        "confidence_degenerate": conf_degen > 0.5,
        "observations": obs,
    }


async def run_case(store: Store, case: Case) -> CaseResult:
    t0 = time.perf_counter()
    result = await retrieve(store, case.question)
    decision = evaluate(result, intent=result.plan.intent)
    latency = (time.perf_counter() - t0) * 1000
    blob = "\n".join(h.body.lower() for h in result.hits)
    hit = any(exp.lower() in blob for exp in case.expect_any) if case.expect_any else bool(result.hits)
    contaminated = any(bad.lower() in blob for bad in case.expect_none)
    if case.unanswerable:
        hit = decision.mode in ("empty", "thin")
    return CaseResult(case=case, hit=hit, mode=decision.mode, latency_ms=latency, contaminated=contaminated)


async def run_suite(store: Store, cases: list[Case] | None = None) -> SuiteResult:
    cases = cases or load_cases()
    results = [await run_case(store, c) for c in cases]
    health = await health_checks(store)
    return SuiteResult(cases=results, health=health)


def compare(current: SuiteResult, baseline: SuiteResult, *, tolerance: float | None = None) -> dict:
    tol = tolerance if tolerance is not None else T.EVAL_TOLERANCE
    delta = current.hit_at_k - baseline.hit_at_k
    ok = delta >= -tol and current.health.get("index_coverage_ok", True)
    out = {
        "ok": ok,
        "delta_hit_at_k": delta,
        "current": current.hit_at_k,
        "baseline": baseline.hit_at_k,
        "health": current.health,
        "by_category": current.by_category(),
    }
    if len(current.cases) >= T.EVAL_MIN_CASES_FOR_SIG:
        try:
            from ranx import Qrels, Run, evaluate as ranx_eval

            qrels = Qrels()
            run = Run()
            for i, cr in enumerate(current.cases):
                qid = str(i)
                qrels.add_qrel(qid, "0", 1 if cr.hit else 0)
                run.add_score(qid, "0", 1.0 if cr.hit else 0.0)
            out["ndcg@8"] = ranx_eval(qrels, run, "ndcg@8")
        except Exception:
            pass
    return out


def save_baseline(result: SuiteResult, path: Path | None = None) -> None:
    path = path or golden_dir() / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"hit_at_k": result.hit_at_k, "refusal_accuracy": result.refusal_accuracy, "health": result.health},
            indent=2,
        )
    )
