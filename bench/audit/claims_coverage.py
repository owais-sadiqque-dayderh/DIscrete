"""Claim evidence provenance and observation coverage, read-only (docs/audit_inputs.md).

    .venv/bin/python bench/audit/claims_coverage.py [--db ~/.cache/exp_discrete/discrete.db]

For each evidence quote: which observation field holds it (caption, screen_text,
window_title; normalized substring, as process/validation.verify_evidence does).
Per episode: observations, how many the drafter actually saw (the last
DRAFT_MAX_OBS, process/pipeline._call_draft), claims, observations cited.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from discrete import textkit as tk  # noqa: E402
from discrete.encryption import open_database  # noqa: E402

DRAFT_MAX_OBS = 40
DRAFT_TEXT_CHARS_PER_OBS = 1200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/.cache/exp_discrete/discrete.db")
    args = ap.parse_args()
    conn = open_database(args.db, read_only=True)
    obs = {r["id"]: r for r in conn.execute("SELECT * FROM observations")}
    ev = conn.execute("SELECT e.*, c.kind, c.episode_id FROM evidence e JOIN claims c ON c.id=e.claim_id").fetchall()

    where = collections.Counter()
    for e in ev:
        o = obs[e["observation_id"]]
        q = tk.norm_key(e["quote"])
        hit = [f for f in ("caption", "screen_text", "window_title") if q and q in tk.norm_key(o[f] or "")]
        where[" + ".join(hit) or "none (substituted window)"] += 1
        # screen_text quotes: OCR part or AX part? OCR-novel lines go first in ax+ocr rows.
    print(f"evidence rows: {len(ev)}; quote found in: {dict(where)}")
    first_hit = collections.Counter()
    for e in ev:
        o = obs[e["observation_id"]]
        q = tk.norm_key(e["quote"])
        for f in ("caption", "screen_text", "window_title"):
            if q and q in tk.norm_key(o[f] or ""):
                first_hit[f] += 1
                break
    print(f"first field that holds the quote (caption checked first): {dict(first_hit)}")
    only_caption = sum(1 for e in ev if (q := tk.norm_key(e["quote"])) and q in tk.norm_key(obs[e["observation_id"]]["caption"] or "")
                       and q not in tk.norm_key(obs[e["observation_id"]]["screen_text"] or ""))
    print(f"quotes present in the caption but not in screen_text: {only_caption}/{len(ev)}")
    src = collections.Counter(obs[e["observation_id"]]["text_source"] for e in ev)
    print(f"text_source of cited observations: {dict(src)}")

    user = [o for o in obs.values() if o["origin"] == "user" and o["episode_id"] is not None]
    cited = {e["observation_id"] for e in ev}
    print(f"\nuser observations in episodes: {len(user)}; cited by any claim: {len(cited)} "
          f"({100 * len(cited) / max(1, len(user)):.0f}%)")
    eng = [o for o in user if o["keys"] or o["clicks"] or o["clipboard"]]
    print(f"engaged (keys/clicks/copies) observations: {len(eng)}; cited: {len(cited & {o['id'] for o in eng})}")

    print("\nepisode | obs | seen_by_drafter | dropped(first N) | claims | cited obs | minutes | apps")
    for ep in conn.execute("SELECT * FROM episodes ORDER BY id"):
        rows = sorted([o for o in user if o["episode_id"] == ep["id"]], key=lambda r: (r["started_at"], r["id"]))
        n_claims = conn.execute("SELECT count(*) FROM claims WHERE episode_id=?", (ep["id"],)).fetchone()[0]
        c = len({o["id"] for o in rows} & cited)
        seen = min(len(rows), DRAFT_MAX_OBS)
        mins = (ep["ended_at"] - ep["started_at"]) / 60
        apps = sorted({o["app"] for o in rows})
        print(f"{ep['id']:7} | {len(rows):3} | {seen:15} | {len(rows) - seen:16} | {n_claims:6} | {c:9} | {mins:7.1f} | {apps}")

    # What the drafter is shown per observation, after cross-row line dedup + 1200-char cut.
    shown = full = 0
    for ep in conn.execute("SELECT id FROM episodes"):
        rows = sorted([o for o in user if o["episode_id"] == ep["id"]], key=lambda r: (r["started_at"], r["id"]))
        seen_lines: set[str] = set()
        for r in rows[-DRAFT_MAX_OBS:]:
            kept = []
            for line in (r["screen_text"] or "").splitlines():
                k = tk.norm_key(line)
                if k and k not in seen_lines:
                    seen_lines.add(k)
                    kept.append(line)
            text = "\n".join(kept)
            full += len(text)
            shown += min(len(text), DRAFT_TEXT_CHARS_PER_OBS)
        for r in rows[:-DRAFT_MAX_OBS]:
            full += len(r["screen_text"] or "")
    print(f"\nnovel (cross-row deduped) screen text in episodes: {full} chars; shown to drafter: {shown} "
          f"({100 * shown / max(1, full):.0f}%)")


if __name__ == "__main__":
    main()
