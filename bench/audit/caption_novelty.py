"""How much a VLM caption adds beyond the stored text, read-only (docs/audit_inputs.md).

    .venv/bin/python bench/audit/caption_novelty.py [--db ~/.cache/exp_discrete/discrete.db]

Novel tokens: caption content tokens absent from screen_text, window_title, app,
url and page_title, after removing caption boilerplate ("user", "viewing", ...).
"""
import argparse
import os
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from discrete import textkit as tk  # noqa: E402

BOILER = set("""user users viewing views viewed view screen shows showing shown page open opened displayed visible
scrolling scrolls scrolled scroll clicking click clicks clicked typing typed type no interaction interactions
capture during while browsing browses passively passive reading reads read looking currently moment final
occurs occurring just without having has their they through also""".split())

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="~/.cache/exp_discrete/discrete.db")
a = ap.parse_args()
conn = sqlite3.connect(f"file:{os.path.expanduser(a.db)}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM observations WHERE origin='user' AND enrich_state='done' AND caption != ''").fetchall()
nov, titleonly, lens, verbs = [], 0, [], 0
ACT = ("typed", "typing", "selected", "clicked", "copied", "scrolled", "opened", "submitted", "searched", "installing",
       "downloaded", "dragging", "playing", "watching", "comparing")
for r in rows:
    cap = [t for t in tk.content_tokens(r["caption"]) if t not in BOILER]
    known = set(tk.content_tokens(" ".join([r["screen_text"] or "", r["window_title"] or "", r["app"] or "",
                                             r["url"] or "", r["page_title"] or ""])))
    title = set(tk.content_tokens((r["window_title"] or "") + " " + (r["app"] or "")))
    lens.append(len(r["caption"]))
    if not cap:
        continue
    nov.append(sum(1 for t in cap if t not in known) / len(cap))
    if all(t in title for t in cap):
        titleonly += 1
    if any(v in r["caption"].lower() for v in ACT):
        verbs += 1
print(f"captions: {len(rows)}; chars median {statistics.median(lens):.0f}")
print(f"share of caption content tokens not in text/title/url: median {statistics.median(nov):.2f}, mean {statistics.mean(nov):.2f}")
print(f"captions with >=30% novel tokens: {sum(1 for x in nov if x >= .3)}/{len(nov)}")
print(f"captions that only restate app/title words: {titleonly}/{len(nov)}")
print(f"captions naming a user action verb ({', '.join(ACT[:6])}, ...): {verbs}/{len(rows)}")
txt = [r for r in rows if len(r["screen_text"] or "") < 300]
print(f"rows with <300 chars of screen text (caption is the main content): {len(txt)}/{len(rows)}")
