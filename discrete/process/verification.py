"""Independent semantic verification shared by drafting and consolidation."""

from __future__ import annotations

import json

from .. import textkit as tk
from ..config import T, verifier_endpoint
from ..llm import LLMError, complete_json
from ..models import ClaimDraft
from .prompts import VERIFY_CLAIMS_PROMPT


async def verify_claims(drafts: list[ClaimDraft], by_id: dict) -> list[bool]:
    if not drafts:
        return []
    candidates = []
    for i, cd in enumerate(drafts):
        observations = []
        for ev in cd.evidence:
            r = by_id.get(ev["observation_id"])
            if r is None:
                continue
            observations.append({"id": r["id"], "quote": ev["quote"], "app": r["app"],
                "window_title": r["window_title"], "caption": r["caption"],
                "screen_text": tk.truncate(r["screen_text"], T.SCREEN_TEXT_MAX_CHARS),
                "keys": r["keys"], "clicks": r["clicks"], "clipboard": r["clipboard"]})
        candidates.append({"index": i, "kind": cd.kind, "text": cd.text, "observations": observations})
    data = await complete_json(verifier_endpoint(), [{"role": "user", "content": VERIFY_CLAIMS_PROMPT.format(
        candidates=json.dumps(candidates, ensure_ascii=False))}], max_tokens=400 + 200 * len(drafts))
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        raise LLMError("claim verifier: missing decisions")
    decisions = {}
    for item in data["decisions"]:
        if not isinstance(item, dict) or type(item.get("index")) is not int or type(item.get("supported")) is not bool:
            raise LLMError("claim verifier: malformed decision")
        i = item["index"]
        if i in decisions or not 0 <= i < len(drafts):
            raise LLMError("claim verifier: duplicate or unknown candidate")
        decisions[i] = item["supported"]
    if len(decisions) != len(drafts):
        raise LLMError("claim verifier: incomplete decisions")
    return [decisions[i] and bool(candidates[i]["observations"]) for i in range(len(drafts))]
