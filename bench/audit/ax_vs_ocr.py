"""AX text vs what is visible in the saved frame (local Apple Vision OCR), read-only.

    .venv/bin/python bench/audit/ax_vs_ocr.py [--db ~/.cache/exp_discrete/discrete.db] [--limit N]

Per row with a saved frame: re-OCR the frame (capture/ocr.py, accurate level) and
measure the share of the stored screen_text's content tokens that do NOT appear
in that OCR. For text_source=ocr rows this is the OCR noise floor (JPEG re-encode,
nondeterminism); for ax / ax+ocr rows the excess over that floor is text AX
exposed that is not legible on screen (offscreen scrollback, truncated previews,
aria labels, hidden subtrees). Also: single-word lines and within-row duplicate
lines by text_source (AX fragmentation), and OCR latency.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from discrete import textkit as tk  # noqa: E402
from discrete.capture import ocr  # noqa: E402
from discrete.encryption import open_database, open_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/.cache/exp_discrete/discrete.db")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    conn = open_database(args.db, read_only=True)
    rows = conn.execute("SELECT * FROM observations WHERE origin='user' AND screen_text != '' ORDER BY id").fetchall()

    frag = collections.defaultdict(lambda: [0, 0, 0])  # lines, single-token lines, duplicate lines
    for r in rows:
        lines = [ln.strip() for ln in r["screen_text"].splitlines() if ln.strip()]
        f = frag[r["text_source"]]
        f[0] += len(lines)
        f[1] += sum(1 for ln in lines if len(tk.tokens(ln)) == 1)
        f[2] += len(lines) - len({tk.norm_key(ln) for ln in lines})
    print("text_source | lines | single-token lines | duplicate lines within row")
    for src, (n, one, dup) in frag.items():
        print(f"{src:8} | {n:6} | {100 * one / n:5.1f}% | {100 * dup / n:5.1f}%")

    hidden = collections.defaultdict(list)
    per_app = collections.defaultdict(list)
    lat = []
    examples = []
    for r in rows[: args.limit]:
        p = r["frame_path"]
        if not p or not Path(p).exists():
            continue
        with open_frame(p) as im:
            img = im.convert("RGB")
        t = time.perf_counter()
        seen = set(tk.content_tokens(ocr.ocr_image(img)))
        lat.append(time.perf_counter() - t)
        toks = tk.content_tokens(r["screen_text"])
        if len(toks) < 20:
            continue
        miss = [x for x in toks if x not in seen]
        share = len(miss) / len(toks)
        hidden[r["text_source"]].append(share)
        if r["text_source"] != "ocr":
            per_app[r["app"]].append(share)
            examples.append((share, r["id"], r["app"], " ".join(dict.fromkeys(miss))[:160]))
    print("\nshare of stored content tokens absent from a fresh OCR of the saved frame")
    for src, xs in hidden.items():
        print(f"{src:8} n={len(xs):3} median={statistics.median(xs):.2f} mean={statistics.mean(xs):.2f}")
    print("by app (ax, ax+ocr rows):")
    for app, xs in sorted(per_app.items(), key=lambda x: -len(x[1])):
        print(f"  {app[:24]:24} n={len(xs):3} median={statistics.median(xs):.2f}")
    print("highest examples:")
    for share, oid, app, miss in sorted(examples, reverse=True)[:8]:
        print(f"  {oid} {app} {share:.2f}: {miss}")
    if lat:
        lat.sort()
        print(f"\nOCR latency per frame (accurate, language correction): n={len(lat)} "
              f"median={statistics.median(lat) * 1000:.0f} ms p90={lat[int(.9 * len(lat))] * 1000:.0f} ms "
              f"(first call {lat[0] * 1000:.0f} ms min)")


if __name__ == "__main__":
    main()
