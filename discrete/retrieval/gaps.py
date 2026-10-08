"""Knowledge gaps with a verification pass (planS3 §8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import data_dir
from ..models import ClaimDraft, RetrievalResult
from ..store import Store
from .grounding import evaluate
from .hybrid import brute_verify, retrieve

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
    cid = await store.create_claim(draft, actor="user", reason=f"answered gap #{gap_id}")
    await store.link_gap_claim(gap_id, cid)
    return {"ok": True, "claim_id": cid, "gap_id": gap_id}


def pending_cases_path() -> Path:
    """Runtime output lives in the data directory, never in the source tree; reviewed
    golden files are never touched (insights.md §4)."""
    return data_dir() / "auto_cases.pending.yaml"


def _append_pending_case(question: str, missed: str) -> None:
    path = pending_cases_path()
    cases = (yaml.safe_load(path.read_text()) or []) if path.exists() else []
    cases.append({"question": question, "expect_ref": missed, "source": "retrieval_miss"})
    path.write_text(yaml.safe_dump(cases, sort_keys=False))
