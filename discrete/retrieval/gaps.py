"""Knowledge gap verification (planS3 §8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import T, golden_dir
from ..retrieval.grounding import evaluate
from ..retrieval.search import brute_verify, retrieve
from ..store import ClaimDraft, Store
from ..process.confidence import compute_confidence

log = logging.getLogger("discrete.retrieval.gaps")


@dataclass
class GapOutcome:
    kind: str  # retrieval_miss | store_lacks
    gap_id: int | None = None


async def maybe_record_gap(store: Store, question: str, *, field_text: str = "") -> GapOutcome:
    result = await retrieve(store, question, field_text=field_text)
    decision = evaluate(result, intent=result.plan.intent)
    if decision.mode not in ("thin", "empty"):
        return GapOutcome(kind="ok")

    miss = await brute_verify(store, question, result.plan)
    if miss:
        await store.record_retrieval_miss(question, miss, "post_grounding")
        _append_pending_case(question, miss)
        return GapOutcome(kind="retrieval_miss")

    gap_id = await store.record_gap(question)
    return GapOutcome(kind="store_lacks", gap_id=gap_id)


async def answer_and_learn(store: Store, gap_id: int, answer: str) -> dict:
    gap = await store.answer_gap(gap_id, answer)
    if gap is None:
        return {"ok": False}
    draft = ClaimDraft(
        kind="fact",
        text=answer.strip(),
        modality="stated",
        reasoning=f"User answered gap: {gap['question']!r}",
        occurred_from=gap["answered_at"],
        occurred_to=gap["answered_at"],
    )
    cid = await store.create_claim(draft, confidence=compute_confidence(["stated"], 1))
    await store._call(
        lambda: store.conn.execute("UPDATE gaps SET claim_id=? WHERE id=?", (cid, gap_id))
    )
    return {"ok": True, "claim_id": cid, "gap_id": gap_id}


def _append_pending_case(question: str, missed: str) -> None:
    path = golden_dir() / "auto_cases.pending.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    if path.exists():
        cases = yaml.safe_load(path.read_text()) or []
    cases.append({"question": question, "expect_any": [missed], "source": "retrieval_miss"})
    path.write_text(yaml.safe_dump(cases, sort_keys=False))
