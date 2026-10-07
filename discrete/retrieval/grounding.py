"""Grounding posture (planS3 §7): a pure function of counts."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import RetrievalResult
from .query import EventQueryPlan


@dataclass
class GroundingDecision:
    mode: str
    strong: int
    weak: int


def evaluate(result: RetrievalResult) -> GroundingDecision:
    if result.why_chain and result.why_chain.candidates:
        return GroundingDecision("ambiguous", 0, 0)
    if result.page_answer and (result.plan.page_operation or isinstance(result.plan, EventQueryPlan)):
        return GroundingDecision("pages", len(result.hits), 0)
    intent = result.plan.intent
    if intent == "edit":
        return GroundingDecision("edit", 0, 0)
    eligible = [h for h in result.hits if h.eligible]
    strong = sum(1 for h in eligible if h.strong)
    weak = len(eligible) - strong
    if intent == "continuation":
        return GroundingDecision("continuation", strong, weak)
    if intent == "enumerate":
        return GroundingDecision("enumerate" if eligible else "empty", strong, weak)
    if result.why_chain and result.why_chain.edges and not result.why_chain.verified:
        return GroundingDecision("why", strong, weak)
    if intent == "why" and any(h.verification == "timed_out" and h.via.startswith(
            ("attention provenance", "led to claim", "a step towards claim", "before ", "the page "))
            for h in result.hits):
        return GroundingDecision("why", strong, weak)
    if intent == "why" and strong:
        context = any(h.strong and h.via.startswith(("attention provenance", "led to claim", "a step towards claim",
                                                    "before ", "the page ")) for h in eligible)
        return GroundingDecision("why" if context else "thin", strong, weak)
    if strong or (result.exhaustive and eligible):
        return GroundingDecision("grounded", strong, weak)
    return GroundingDecision("thin" if weak else "empty", strong, weak)
