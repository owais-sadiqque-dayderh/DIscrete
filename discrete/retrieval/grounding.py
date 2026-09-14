"""Grounding posture (planS3 §7)."""

from __future__ import annotations

from dataclasses import dataclass

from .search import RetrievalResult


@dataclass
class GroundingDecision:
    mode: str
    strong: int
    weak: int


def evaluate(result: RetrievalResult, *, intent: str = "recall") -> GroundingDecision:
    if intent == "edit":
        return GroundingDecision(mode="edit", strong=0, weak=0)
    if intent == "continuation":
        return GroundingDecision(mode="continuation", strong=len(result.hits), weak=0)
    strong = sum(1 for h in result.hits if h.strong)
    weak = len(result.hits) - strong
    if result.exhaustive and result.episodes:
        return GroundingDecision(mode="grounded", strong=max(1, strong), weak=weak)
    if strong >= 1 or any(h.ref_kind == "entity" and h.strong for h in result.hits):
        mode = "grounded"
    elif weak:
        mode = "thin"
    else:
        mode = "empty"
    return GroundingDecision(mode=mode, strong=strong, weak=weak)
