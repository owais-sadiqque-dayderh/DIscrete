"""Print one observation's stored fields for frame spot checks (read-only).

    .venv/bin/python bench/audit/show_obs.py ID [ID...] [--db PATH] [--chars N]
"""
import argparse
import os
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="+", type=int)
ap.add_argument("--db", default="~/.cache/exp_discrete/discrete.db")
ap.add_argument("--chars", type=int, default=900)
a = ap.parse_args()
conn = sqlite3.connect(f"file:{os.path.expanduser(a.db)}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
for i in a.ids:
    r = conn.execute("SELECT * FROM observations WHERE id=?", (i,)).fetchone()
    print(f"### {r['id']} {r['app']} | {r['window_title'][:90]} | {r['url'][:80]} | {r['text_source']} | {r['event_kind']} "
          f"k/c/s={r['keys']}/{r['clicks']}/{r['scrolls']} dup={r['dup_count']}")
    print(f"CAPTION: {r['caption']}")
    print(f"PAGE: {r['page_title']!r} listing={r['is_listing']} from={r['navigated_from']!r} ents={r['entities']}")
    print(f"FRAME: {r['frame_path']}")
    t = r["screen_text"] or ""
    print(f"TEXT ({len(t)} chars, {len(t.splitlines())} lines): " + t[: a.chars].replace("\n", " ⏎ "))
    print()
