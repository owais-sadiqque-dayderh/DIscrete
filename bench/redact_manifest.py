"""Apply capture-time secret redaction to a replay manifest, in place.

The replay feeds observations straight to the store, past ScreenObserver._capture
where live capture redacts, so a manifest built from older captures can still hold
a key. Text is redacted exactly as capture would; a capture with a confident secret
loses its frame, as it would live.

    .venv/bin/python bench/redact_manifest.py bench/private/live1006/manifest.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from discrete.capture.redact import redact, redact_url


def main(paths: list[str]) -> int:
    for path in map(Path, paths):
        out, masked, framed = [], 0, 0
        for line in path.read_text().splitlines():
            item = json.loads(line)
            d = item["draft"]
            t, u, x = redact(d.get("window_title") or ""), redact_url(d.get("url") or ""), redact(d.get("screen_text") or "")
            d["window_title"], d["url"], d["screen_text"] = t.text, u.text, x.text
            masked += bool(t.count + u.count + x.count)
            if t.confident + u.confident + x.confident and item.pop("image", None):
                framed += 1
                print(f"  frame dropped: {item['label']} {d['app']} {d['window_title'][:50]!r}")
            out.append(json.dumps(item, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n")
        print(f"{path}: {masked} captures masked, {framed} frames dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
