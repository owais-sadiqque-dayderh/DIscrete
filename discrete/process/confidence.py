"""Derived confidence from evidence (planS2 §8)."""

from __future__ import annotations

import math

from ..config import T

_MODALITY_W = {"stated": 1.0, "engaged": 0.8, "passive": 0.4}


def modality_of(keys: int, clicks: int, clipboard: int) -> str:
    """planS2 §5.2: engaged iff the observation had any input."""
    return "engaged" if (keys or clicks or clipboard) else "passive"


def compute_confidence(episode_modalities: list[str], distinct_days: int) -> float:
    """`episode_modalities`: one entry per distinct supporting episode (its best
    modality), plus "stated" for user testimony."""
    e = sum(_MODALITY_W.get(m, 0.4) for m in episode_modalities)
    e += 0.5 * max(0, distinct_days - 1)
    return 1.0 - math.exp(-e / T.CONFIDENCE_SCALE)


def half_life_days(kind: str) -> float | None:
    if kind == "event":
        return None
    if kind == "preference":
        return T.PREF_HALF_LIFE_DAYS
    if kind == "fact":
        return T.FACT_HALF_LIFE_DAYS
    return T.RECENCY_HALF_LIFE_DAYS  # episodes, observations
