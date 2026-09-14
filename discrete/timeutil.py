"""Time: every stored timestamp is a float of Unix epoch seconds.

An epoch float cannot be timezone-naive, which removes the naive/aware bug class
that silently disabled WorkAA's revise cooldown (insights.md §7.1). Local
wall-clock time appears only at the edges: display and query-time parsing.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta


def now() -> float:
    return time.time()


def to_local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts).astimezone()


def from_local(d: datetime) -> float:
    """Naive datetimes are interpreted as local time."""
    if d.tzinfo is None:
        d = d.astimezone()
    return d.timestamp()


def iso(ts: float | None) -> str:
    return to_local(ts).isoformat(timespec="seconds") if ts else ""


def day_key(ts: float) -> str:
    return to_local(ts).strftime("%Y-%m-%d")


def day_start(ts: float) -> float:
    d = to_local(ts)
    return from_local(d.replace(hour=0, minute=0, second=0, microsecond=0))


def day_bounds(ts: float) -> tuple[float, float]:
    start = day_start(ts)
    end = from_local((to_local(start) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
    return start, end


def age_days(ts: float, ref: float | None = None) -> float:
    return max(0.0, ((ref if ref is not None else now()) - ts) / 86400.0)
