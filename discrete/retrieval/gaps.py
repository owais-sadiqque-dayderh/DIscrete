"""Knowledge gaps with a verification pass (planS3 §8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

from ..config import golden_dir
from ..store import ClaimDraft, Store
from .grounding import evaluate
from .search import RetrievalResult, brute_verify, retrieve

log = logging.getLogger("discrete.retrieval.gaps")


@dataclass
class GapOutcome:
    kind: str  # ok | retrieval_miss | store_lacks
    gap_id: int | None = None
    missed: str = ""


async def maybe_record_gap(store: Store, question: str, *, result: RetrievalResult | None = None,
                           field_text: str = "") -> GapOutcome:
    result = result or await retrieve(store, question, field_text=field_text)
    if evaluate(result).mode not in ("thin", "empty"):
        return GapOutcome(kind="ok")
    missed = await brute_verify(store, result.plan, {(h.ref_kind, h.ref_id) for h in result.hits})
    if missed:
        await store.record_retrieval_miss(question, missed, "post_grounding")
        _append_pending_case(question, missed)
        return GapOutcome(kind="retrieval_miss", missed=missed)
    return GapOutcome(kind="store_lacks", gap_id=await store.record_gap(question))


async def answer_and_learn(store: Store, gap_id: int, answer: str) -> dict:
    gap = await store.answer_gap(gap_id, answer)
    if gap is None:
        return {"ok": False}
    draft = ClaimDraft(
        kind="fact", text=answer.strip(), modality="stated",
        reasoning=f"user answered gap: {gap['question']!r}",
        occurred_from=gap["answered_at"], occurred_to=gap["answered_at"],
    )
    cid = await store.create_claim(draft)
    await store.execute("UPDATE gaps SET claim_id=? WHERE id=?", (cid, gap_id))
    return {"ok": True, "claim_id": cid, "gap_id": gap_id}


def _append_pending_case(question: str, missed: str) -> None:
    """Never touches reviewed golden files (insights.md §4)."""
    path = golden_dir() / "auto_cases.pending.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = (yaml.safe_load(path.read_text()) or []) if path.exists() else []
    cases.append({"question": question, "expect_ref": missed, "source": "retrieval_miss"})
    path.write_text(yaml.safe_dump(cases, sort_keys=False))
