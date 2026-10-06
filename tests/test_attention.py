"""Attention: time per site/app from every sighting, duplicates included."""

from __future__ import annotations

from discrete.models import ObservationDraft
from discrete.retrieval.query import fallback_plan

T0 = 1_790_000_000.0


async def _see(store, t: float, site: str, text: str, *, clicks: int = 0) -> int:
    oid, _ = await store.add_observation(ObservationDraft(
        app="Arc", window_title=site, url=f"https://{site}/", site=site, screen_text=text * 6, caption=text,
        session_id="s", started_at=t, ended_at=t + 2, clicks=clicks))
    return oid


async def test_a_collapsed_duplicate_still_counts_as_time(store):
    a = await _see(store, T0, "x.com", "home timeline post ", clicks=1)
    b = await _see(store, T0 + 60, "github.com", "pull request diff ")
    again = await _see(store, T0 + 120, "x.com", "home timeline post ")  # same text: collapsed into a
    assert again == a and b != a
    att = {x.key: x for x in await store.attention(T0 - 1, T0 + 1000, by="site")}
    # x.com: 60 s (until github) + the last sighting's 30 s; github.com: 60 s.
    assert round(att["x.com"].seconds) == 90 and att["x.com"].visits == 1
    assert round(att["x.com"].active_seconds) == 60  # only the first sighting had input
    assert round(att["github.com"].seconds) == 60


async def test_idle_gaps_are_capped(store):
    await _see(store, T0, "x.com", "a post ")
    await _see(store, T0 + 3600, "github.com", "a diff ")  # an hour away from the desk
    att = {x.key: x for x in await store.attention(T0 - 1, T0 + 9999, by="site")}
    assert att["x.com"].seconds == 180


def test_time_questions_are_planned_as_time_spent():
    assert fallback_plan("what do I spend most time on?").time_spent
    assert fallback_plan("how much time did I spend on twitter today").time_spent
    assert not fallback_plan("what did I read on twitter today").time_spent
