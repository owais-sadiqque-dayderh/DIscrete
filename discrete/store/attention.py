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
        """Time per `by` (site | app | page) between start and end, most first. Aggregated
        in SQL (a window function for the gap to the next sighting), so a long interval
        costs one indexed scan, not a Python loop; intervals are capped at ATTENTION_MAX_DAYS."""
        start = max(start, end - T.ATTENTION_MAX_DAYS * 86400)
        key = {"app": "COALESCE(NULLIF(o.app, ''), '(unknown)')",
               "page": "COALESCE(NULLIF(o.page_title, ''), o.window_title) || ' (' || "
                       "COALESCE(NULLIF(o.site, ''), o.app) || ')'"}.get(
            by, "COALESCE(NULLIF(o.site, ''), NULLIF(o.app, ''), '(unknown)')")
        rows = await self.q(
            f"""WITH s AS (
                  SELECT s.obs_id, s.started_at, (s.keys + s.clicks + s.scrolls + s.clipboard) > 0 AS active,
                         MIN(COALESCE(LEAD(s.started_at) OVER (ORDER BY s.started_at, s.id) - s.started_at, ?), ?)
                           AS dwell
                  FROM sightings s JOIN observations o ON o.id=s.obs_id
                  WHERE o.origin='user' AND s.started_at >= ? AND s.started_at < ?)
                SELECT {key} AS k, SUM(MAX(s.dwell, 0)) AS seconds,
                       SUM(CASE WHEN s.active THEN MAX(s.dwell, 0) ELSE 0 END) AS active_seconds,
                       COUNT(DISTINCT s.obs_id) AS visits,
                       COUNT(DISTINCT date(s.started_at, 'unixepoch', 'localtime')) AS days
                FROM s JOIN observations o ON o.id=s.obs_id
                GROUP BY k ORDER BY seconds DESC LIMIT ?""",
            (T.DWELL_LAST_S, T.DWELL_GAP_CAP_S, start, end, limit))
        return [Attention(r["k"], r["seconds"] or 0.0, r["active_seconds"] or 0.0, r["visits"], r["days"])
                for r in rows]
