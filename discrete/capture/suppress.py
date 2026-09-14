"""Prevent the system from modelling its own output."""

from __future__ import annotations

import time

from ..config import T

_suppressed_until = 0.0
_recent_agent_text: list[str] = []


def suppress(seconds: float = 3.0) -> None:
    global _suppressed_until
    _suppressed_until = time.monotonic() + seconds


def is_suppressed() -> bool:
    return time.monotonic() < _suppressed_until


def note_agent_output(text: str) -> None:
    global _recent_agent_text
    _recent_agent_text = ([text] + _recent_agent_text)[:8]


def classify_origin(app: str, screen_text: str) -> str:
    app_l = (app or "").lower()
    if any(s in app_l for s in T.SELF_APPS):
        return "agent"
    for prior in _recent_agent_text:
        if prior and tk_overlap(prior, screen_text) >= 0.6:
            return "agent"
    return "user"


def tk_overlap(a: str, b: str) -> float:
    from .. import textkit as tk

    return tk.jaccard(tk.trigrams(a), tk.trigrams(b))
