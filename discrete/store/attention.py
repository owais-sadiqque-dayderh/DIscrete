"""Attention: where the user's time went, from every sighting of every window.

Capture is event-driven: a new capture comes when something happens (a click, a key,
a scroll, a switch, a heartbeat). So a sighting's dwell is the time until the next
sighting of anything, capped (`DWELL_GAP_CAP_S`) so that a walk away from the desk
doesn't count as reading. A sighting is *active* when the user scrolled, clicked or
typed during it. Totals per site, app or page answer "what do I spend my time on"
without a model, and dreaming reads them as context for interests and habits.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import T
from ..timeutil import day_key
from .base import StoreBase


@dataclass
class Attention:
    key: str
    seconds: float
    active_seconds: float
    visits: int
    days: int

    def describe(self) -> str:
        return (f"{self.key}: {_dur(self.seconds)} ({_dur(self.active_seconds)} active) over {self.visits} "
                f"visit{'s' * (self.visits != 1)} on {self.days} day{'s' * (self.days != 1)}")


def _dur(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{max(m, 1) if seconds else 0} min"


class AttentionMixin(StoreBase):
    async def attention(self, start: float, end: float, *, by: str = "site", limit: int = 20) -> list[Attention]:
        """Time per `by` (site | app | page) between start and end, most first."""
        rows = await self.q(
            "SELECT s.obs_id, s.started_at, s.keys + s.clicks + s.scrolls + s.clipboard AS acts, o.app, o.site, "
            "o.page_title, o.window_title FROM sightings s JOIN observations o ON o.id=s.obs_id "
            "WHERE o.origin='user' AND s.started_at >= ? AND s.started_at < ? ORDER BY s.started_at, s.id",
            (start, end))
        acc: dict[str, list] = {}
        for i, r in enumerate(rows):
            nxt = rows[i + 1]["started_at"] if i + 1 < len(rows) else r["started_at"] + T.DWELL_LAST_S
            dwell = max(0.0, min(nxt - r["started_at"], T.DWELL_GAP_CAP_S))
            if by == "app":
                key = r["app"] or "(unknown)"
            elif by == "page":
                key = f"{r['page_title'] or r['window_title']} ({r['site'] or r['app']})"
            else:
                key = r["site"] or r["app"] or "(unknown)"
            a = acc.setdefault(key, [0.0, 0.0, set(), set()])
            a[0] += dwell
            a[1] += dwell if r["acts"] else 0.0
            a[2].add(r["obs_id"])
            a[3].add(day_key(r["started_at"]))
        out = [Attention(k, v[0], v[1], len(v[2]), len(v[3])) for k, v in acc.items()]
        return sorted(out, key=lambda a: -a.seconds)[:limit]
