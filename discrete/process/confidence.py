"""Derived confidence from evidence (planS2 §8)."""

from __future__ import annotations

import math

from ..config import T

_MODALITY_W = {"stated": 1.0, "engaged": 0.8, "passive": 0.4}


def compute_confidence(episode_modalities: list[str], distinct_days: int) -> float:
    e = 0.0
    for mod in set(episode_modalities):
        w = _MODALITY_W.get(mod, 0.4)
        e += w
    e += 0.5 * max(0, distinct_days - 1)
    return 1.0 - math.exp(-e / T.CONFIDENCE_SCALE)


def half_life_days(kind: str) -> float | None:
    if kind == "event":
        return None
    if kind == "fact":
        return T.FACT_HALF_LIFE_DAYS
    if kind == "preference":
        return T.PREF_HALF_LIFE_DAYS
    return T.FACT_HALF_LIFE_DAYS
