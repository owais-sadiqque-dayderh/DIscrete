"""Turn a DIscrete store's raw observations into a replay manifest.

The replay (tools/replay_session.py) then rebuilds memory from exactly the same
captures with any checkout's pipeline, so a baseline and a branch are compared on
identical input. Episode boundaries are kept (one manifest group per episode).

    python bench/export_discrete_session.py --db ~/.cache/exp_discrete/discrete.db --out bench/private/live1006
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

FIELDS = ("source", "origin", "event_kind", "session_id", "started_at", "ended_at", "app", "bundle_id",
          "window_title", "url", "site", "screen_text", "text_source", "keys", "clicks", "scrolls", "clipboard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{args.db.expanduser()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM observations ORDER BY started_at, id").fetchall()
    n_frames = 0
    with (args.out / "manifest.jsonl").open("w") as f:
        for r in rows:
            item = {"label": f"o{r['id']}", "group": f"ep{r['episode_id'] or 0:04d}",
                    "draft": {k: r[k] for k in FIELDS}}
            frame = Path(r["frame_path"]) if r["frame_path"] else None
            if frame is not None and frame.exists() and r["origin"] == "user":
                item["image"] = str(frame)
                n_frames += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(rows)} observations, {n_frames} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
