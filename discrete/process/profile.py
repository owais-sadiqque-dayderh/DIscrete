"""The profile: one living page about the user, rewritten by each dream.

Honcho keeps a "peer card", Hindsight "mental models", Letta core-memory blocks:
a short document an agent reads first. Here it is written from lasting memories
(patterns, facts, preferences) and recent completed actions, and every point cites
the memories it comes from. Points citing nothing real are dropped in code, and the
rest are checked by the independent verifier, so the page cannot say more than the
memories do. It is what an action model would read about the user, what `discrete
profile` prints, the top of the Markdown export, and a record Bubble can use for
"what do you know about me" questions.
"""

from __future__ import annotations

import logging

from .. import textkit as tk
from ..config import dream_endpoint
from ..llm import complete_json
from ..store import Store
from ..timeutil import now, to_local
from .dream import _verify

log = logging.getLogger("discrete.process.profile")

SECTIONS = ("Work and projects", "Tools", "Interests", "Habits and routines", "Preferences", "People")

PROFILE_PROMPT = """Write a short profile of the user from their memories: what an assistant acting on their
behalf should know about them. Each memory line is [id] (kind, strength) text; strength says on
how many episodes and days it was seen.

{memories}

Sections, in this order, omitting empty ones: {sections}.
Rules:
- Every point cites the memory ids it comes from (claim_ids). Use only what those memories say.
- Viewing is not doing: a page the user only read is an interest, not something they use or did.
- Word tentative things (seen once or on one day) as recent or possible ("recently", "seems to").
- At most 6 points per section, each at most 160 characters, addressed in third person ("They …").
Return strict JSON:
{{"sections": [{{"heading": "<one of the sections>", "points": [{{"text": "<string>", "claim_ids": [<int>]}}]}}]}}"""


async def profile_inputs(store: Store) -> list:
    lasting = await store.q("SELECT * FROM claims WHERE status='live' AND kind IN ('pattern','fact','preference') "
                            "AND flag NOT IN ('stale','contested') ORDER BY confidence DESC LIMIT 150")
    actions = await store.q("SELECT * FROM claims WHERE status='live' AND kind='event' AND action != '' "
                            "AND occurred_to >= ? ORDER BY occurred_from DESC LIMIT 40", (now() - 14 * 86400,))
    return [*lasting, *actions]


async def write_profile(store: Store, *, run_id: str = "") -> dict:
    ep = dream_endpoint()
    rows = await profile_inputs(store)
    if not ep.configured or not rows:
        return {"skipped": "no model" if not ep.configured else "no lasting memories"}
    strengths = await store.strengths([r["id"] for r in rows])
    lines = []
    for r in rows:
        st = strengths.get(r["id"])
        kind = f"{r['kind']} ({r['action']})" if r["action"] else r["kind"]
        lines.append(f"[{r['id']}] ({kind}, {st.describe() if st else 'stated'}) {r['text']}")
    data = await complete_json(ep, [{"role": "user", "content": PROFILE_PROMPT.format(
        memories="\n".join(lines), sections="; ".join(SECTIONS))}], max_tokens=1800)
    by_id = {r["id"]: r for r in rows}
    points: list[dict] = []
    for sec in (data or {}).get("sections") or []:
        if not isinstance(sec, dict) or sec.get("heading") not in SECTIONS:
            continue
        for pt in (sec.get("points") or [])[:6]:
            if not isinstance(pt, dict):
                continue
            try:
                ids = sorted({int(i) for i in pt.get("claim_ids") or [] if int(i) in by_id})
            except (TypeError, ValueError):
                continue
            text = tk.truncate(str(pt.get("text") or "").strip(), 200)
            if ids and text:
                points.append({"section": sec["heading"], "text": text, "support": ids})
    decisions = await _verify(points, by_id)
    kept = [p for p, ok in zip(points, decisions, strict=True) if ok]
    if not kept:
        return {"points": 0, "proposed": len(points)}
    sections = [{"heading": h, "points": [{"text": p["text"], "claims": p["support"]} for p in kept
                                          if p["section"] == h]} for h in SECTIONS]
    profile = {"written_at": now(), "sections": [s for s in sections if s["points"]]}
    await store.save_profile(profile, run_id=run_id)
    return {"points": len(kept), "proposed": len(points)}


def render_profile(profile: dict, *, with_ids: bool = False) -> str:
    if not profile.get("sections"):
        return ""
    out = [f"Profile (written {to_local(profile['written_at']):%a %b %d %H:%M})"]
    for sec in profile["sections"]:
        out.append(f"\n{sec['heading']}:")
        for p in sec["points"]:
            ids = f" [{', '.join(f'#{c}' for c in p['claims'])}]" if with_ids else ""
            out.append(f"- {p['text']}{ids}")
    return "\n".join(out)
