"""Grounding posture (planS3 §7): a pure function of counts."""

from __future__ import annotations

from dataclasses import dataclass

from .search import RetrievalResult


@dataclass
class GroundingDecision:
    mode: str
    strong: int
    weak: int


def evaluate(result: RetrievalResult) -> GroundingDecision:
    intent = result.plan.intent
    if intent == "edit":
        return GroundingDecision("edit", 0, 0)
    strong = sum(1 for h in result.hits if h.strong)
    weak = len(result.hits) - strong
    if intent == "continuation":
        return GroundingDecision("continuation", strong, weak)
    if strong or (result.exhaustive and result.hits):
        return GroundingDecision("grounded", strong, weak)
    return GroundingDecision("thin" if weak else "empty", strong, weak)
