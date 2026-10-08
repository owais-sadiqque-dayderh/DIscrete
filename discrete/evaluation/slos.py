"""Local latency gates, shared by bench and the read-only invariant checker."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SLO_LIMITS_MS = {"capture.text": 1200.0, "enrich.total": 30_000.0, "segment.lag": 60_000.0}
# Match bench's source policy: query/evaluation measurements are not pipeline work.
EXCLUDED_SOURCES = frozenset({"eval", "query", "ask", "bench", "check"})


@dataclass(frozen=True)
class StageSLO:
    stage: str
    n: int
    p90_ms: float | None
    limit_ms: float
    passed: bool | None
    status: str
    reason: str | None = None


def evaluate_slos(samples: Iterable[Mapping[str, Any]], *, pipeline_only: bool = True) -> list[StageSLO]:
    """Linear p90 (same as bench); an unmeasured stage is not applicable."""
    timed: dict[str, list[float]] = {stage: [] for stage in SLO_LIMITS_MS}
    for sample in samples:
        if pipeline_only and sample["src"] in EXCLUDED_SOURCES:
            continue
        stage, ms = sample["stage"], sample["ms"]
        if stage in timed and ms is not None and math.isfinite(ms) and ms >= 0:
            timed[stage].append(float(ms))
    results = []
    for stage, limit in SLO_LIMITS_MS.items():
        values = sorted(timed[stage])
        rank = (len(values) - 1) * 0.9
        p90 = (values[math.floor(rank)] + (values[math.ceil(rank)] - values[math.floor(rank)])
               * (rank - math.floor(rank))) if values else None
        passed = p90 < limit if p90 is not None else None
        results.append(StageSLO(stage, len(values), p90, limit, passed,
                                "n/a" if p90 is None else "pass" if passed else "fail",
                                "no valid timing samples for this stage" if p90 is None else None))
    return results
