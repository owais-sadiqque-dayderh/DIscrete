"""Deterministic claim validation (planS2 §5.2). No model involved."""

from __future__ import annotations

import re

from .. import textkit as tk
from ..config import T

_AND_JOIN = re.compile(r"\band\b|\bwhile\b", re.I)
_THIRD_PARTY = ("the page says", "the website says", "according to the page", "the article says", "the post says")


def atomic(text: str, entities: list[str] | None = None) -> bool:
    if not text or len(text) > T.CLAIM_MAX_CHARS:
        return False
    if len(tk.sentences(text)) > 2:
        return False
    if entities and len(entities) > 3 and _AND_JOIN.search(text):
        return False
    return True


_FIELDS = ("screen_text", "caption", "window_title")


def verify_evidence(claim_text: str, quote: str, obs) -> tuple[str, bool, str]:
    """Accept a quote that is a literal (normalized) substring of the observation.
    Otherwise substitute the observation's best window for the claim's content
    words, accepted only at overlap ≥ EVIDENCE_MIN_OVERLAP. Returns (quote, ok, field);
    the returned quote is always real text, and field names where it was found
    (screen_text first: what the user saw verbatim, not a model's description)."""
    fields = [(name, obs[name]) for name in _FIELDS if obs[name]]
    q = tk.norm_key(quote)
    terms = set(tk.content_tokens(claim_text))
    if not terms:
        return "", False, ""
    if len(q) >= 4:
        found = next((name for name, f in fields if q in tk.norm_key(f)), "")
        # Authenticity alone says nothing about support. A genuine but unrelated
        # quote must pass the same relevance gate as a replacement excerpt.
        if found and len(terms & set(tk.content_tokens(quote))) / len(terms) >= T.EVIDENCE_MIN_OVERLAP:
            return quote.strip(), True, found
    # Best window per field, so a substitute quote never spans two fields.
    best = max(((tk.best_window(f, terms, width=24), name) for name, f in fields), key=lambda x: x[0][1],
               default=(("", 0.0), ""))
    (snippet, overlap), name = best
    ok = bool(snippet) and overlap >= T.EVIDENCE_MIN_OVERLAP
    return snippet, ok, name if ok else ""


def injection_drop(claim_text: str, evidence_texts: list[str]) -> bool:
    """An imperative sentence copied from screen text into a claim is dropped."""
    grams = tk.trigrams(claim_text)
    for ev in evidence_texts:
        for sent in tk.sentences(ev):
            if tk.looks_imperative(sent) and tk.containment(tk.trigrams(sent), grams) >= 0.8:
                return True
    return False


def subject_ok(text: str) -> bool:
    low = text.lower()
    return not any(b in low for b in _THIRD_PARTY)
