"""Step-0 measurement (docs/C7.md): one record per model call and one sample per
pipeline stage, buffered in memory and written to the store with the usage flush.

    PERF.sample("enrich.vision", ms)          # a timed stage
    with PERF.timer("draft.episode", ref_id=7) as t:
        ...
        t.n = claims_added                    # optional count carried with the sample
    PERF.count("capture.frame_dup", 3)         # an event counter (no duration)

Every record carries `PERF.source` ("live" for the daemon, "ask", "eval", …) so a
benchmark over the daemon's numbers is not skewed by eval or debugging runs.
No dependencies: any module may import this.
"""

from __future__ import annotations

import collections
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

_MAX_BUFFERED = 50_000  # a process that never flushes (tests, a crash loop) stays bounded


@dataclass
class Call:
    """One model call. `ms` is the whole call; `first_ms` the first streamed token."""

    component: str
    model: str
    at: float = 0.0
    ms: float = 0.0
    first_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None  # USD, as reported by the provider (OpenRouter sends it)
    status: str = "ok"  # ok | error | cancelled
    source: str = ""


@dataclass
class Sample:
    """A stage duration (`ms`) or an event count (`ms` None, `n` the count)."""

    stage: str
    ms: float | None
    at: float = 0.0
    n: int | None = None
    ref_id: int | None = None
    source: str = ""


@dataclass
class _Timer:
    n: int | None = None
    ref_id: int | None = None


class PerfLog:
    def __init__(self) -> None:
        self.source = "live"
        self._calls: collections.deque[Call] = collections.deque(maxlen=_MAX_BUFFERED)
        self._samples: collections.deque[Sample] = collections.deque(maxlen=_MAX_BUFFERED)

    def call(self, rec: Call) -> None:
        rec.source = rec.source or self.source
        self._calls.append(rec)

    def sample(self, stage: str, ms: float, *, n: int | None = None, ref_id: int | None = None) -> None:
        self._samples.append(Sample(stage, float(ms), time.time(), n, ref_id, self.source))

    def count(self, name: str, n: int = 1) -> None:
        if n:
            self._samples.append(Sample(name, None, time.time(), int(n), None, self.source))

    @contextmanager
    def timer(self, stage: str, *, n: int | None = None, ref_id: int | None = None) -> Iterator[_Timer]:
        """Time the block; recorded only when it finishes without an exception."""
        t = _Timer(n, ref_id)
        t0 = time.perf_counter()
        yield t
        self.sample(stage, (time.perf_counter() - t0) * 1000, n=t.n, ref_id=t.ref_id)

    def drain(self) -> tuple[list[Call], list[Sample]]:
        calls, samples = list(self._calls), list(self._samples)
        self._calls.clear()
        self._samples.clear()
        return calls, samples


PERF = PerfLog()


def since_ms(t0: float) -> float:
    """Milliseconds since a `time.perf_counter()` reading."""
    return (time.perf_counter() - t0) * 1000
