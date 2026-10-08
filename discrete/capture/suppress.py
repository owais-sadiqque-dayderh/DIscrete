"""Prevent the system from modelling its own output (planS1 §9)."""

from __future__ import annotations

import re
import time

from .. import textkit as tk
from ..config import T

_suppressed_until = 0.0
_holds = 0
_recent_agent_text: list[str] = []

# DIscrete's own output read back off the screen: a terminal running `jarvis ask`/`discrete
# query`, the records it prints, or its database open in a viewer. Memories drafted from
# these are about DIscrete, not the user (claims built from a printed answer, seen live).
_SELF_COMMAND = re.compile(
    r"\b(?:jarvis(?:-exp)?|discrete) (?:ask|search|why|query|memories|episodes|captured|profile|patterns|"
    r"changes|history|logs|status|graph|cost|stats|dream|export)\b", re.I)
_SELF_RECORD = re.compile(r"\[(?:claim|screen|episode|entity) \d+ ·")
_SELF_PLACES = ("discrete.db", "/.cache/discrete", "/.cache/exp_discrete", "jarvis/discrete.log")


def suppress(seconds: float = 3.0) -> None:
    global _suppressed_until
    _suppressed_until = max(_suppressed_until, time.monotonic() + seconds)


def hold() -> None:
    """Suppress capture until the matching release(); Bubble holds it for a whole run."""
    global _holds
    _holds += 1


def release(tail_s: float = 2.0) -> None:
    """End a hold; keep suppressing briefly so the app's reaction to our typing is not captured."""
    global _holds
    _holds = max(0, _holds - 1)
    suppress(tail_s)


def is_suppressed() -> bool:
    return _holds > 0 or time.monotonic() < _suppressed_until


def note_agent_output(text: str) -> None:
    global _recent_agent_text
    _recent_agent_text = ([text] + _recent_agent_text)[:8]


def classify_origin(app: str, window_title: str, screen_text: str) -> str:
    app_l = (app or "").lower()
    title_l = (window_title or "").lower()
    if any(s and s in app_l for s in T.SELF_APPS):
        return "agent"
    # A terminal running the daemon is an ordinary app; its window title is the tell.
    if any(p and p in title_l for p in T.SELF_WINDOW_PATTERNS):
        return "agent"
    if _SELF_COMMAND.search(window_title or "") or any(p in title_l for p in _SELF_PLACES):
        return "agent"
    if len(_SELF_RECORD.findall(screen_text or "")) >= 2:
        return "agent"  # Bubble's evidence pack, printed by `ask --records` or `query`
    grams = tk.trigrams(screen_text)
    for prior in _recent_agent_text:
        if prior and tk.containment(tk.trigrams(prior), grams) >= 0.6:
            return "agent"
    return "user"
