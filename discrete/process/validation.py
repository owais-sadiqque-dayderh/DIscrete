"""Deterministic claim validation (planS2 §5.2)."""

from __future__ import annotations

import re

from .. import textkit as tk
from ..config import T

_AND_JOIN = re.compile(r"\band\b|\bwhile\b", re.I)


def atomic(text: str, entities: list[str] | None = None) -> bool:
    sents = tk.sentences(text)
    if len(sents) > 2:
        return False
    if len(text) > T.CLAIM_MAX_CHARS:
        return False
    if entities and len(entities) > 3 and _AND_JOIN.search(text):
        return False
    return True


def verify_quote(quote: str, obs_text: str, obs_caption: str) -> tuple[str, bool]:
    body = f"{obs_text}\n{obs_caption}"
    if quote and tk.norm_key(quote) in tk.norm_key(body):
        return quote, True
    terms = set(tk.content_tokens(body))
    snippet, overlap = tk.best_window(body, terms)
    return snippet, overlap >= T.EVIDENCE_MIN_OVERLAP


def modality_from_obs(obs) -> str:
    if obs["keys"] or obs["clicks"] or obs["clipboard"]:
        return "engaged"
    return "passive"


def injection_drop(claim_text: str, evidence_texts: list[str]) -> bool:
    for ev in evidence_texts:
        for sent in tk.sentences(ev):
            if tk.looks_imperative(sent) and tk.jaccard(tk.trigrams(claim_text), tk.trigrams(sent)) >= 0.8:
                return True
    return False


def subject_ok(text: str) -> bool:
    low = text.lower()
    bad = ("the page says", "the website says", "according to the page", "the article says")
    return not any(b in low for b in bad)
