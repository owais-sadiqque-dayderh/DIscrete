"""Rebuild the 46-question bench session as a replay manifest, from a Hindsight bank.

The original DIscrete store of 2026-09-30/10-01 (162 observations, 132 frames) was
replayed into the Hindsight bank ``bench-v0`` with transcript format v1, which keeps
every observation's full screen text and its frame. This reads those transcripts and
attachments back out of Hindsight's Postgres (through ``psql``) and writes a manifest
that ``tools/replay_session.py`` turns into a fresh DIscrete database.

    python bench/export_hindsight_session.py --out bench/private/g46

Writes ``manifest.jsonl`` and ``frames/*.jpg`` under ``--out``. The output holds the
user's real screen text: keep it under ``bench/private/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

PSQL = Path.home() / ".pg0/installation/18.1.0/bin/psql"
ATT = re.compile(r"⟦hs-att:([0-9a-f]{12})⟧")
HEADER = re.compile(r"^\[o(\d+)\]\s+(\{.*\})\s*$")
LINE = re.compile(r"^o(\d+)\|\s?(.*)$")


def psql(sql: str, port: int) -> str:
    env = os.environ | {"PGPASSWORD": "hindsight"}
    cmd = [str(PSQL), "-h", "127.0.0.1", "-p", str(port), "-U", "hindsight", "-d", "hindsight", "-At", "-c", sql]
    return subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout


def site_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


def parse(text: str) -> list[dict]:
    """Observations in transcript order; a frame belongs to the next observation."""
    obs: dict[int, dict] = {}
    order: list[int] = []
    pending: list[str] = []
    for raw in text.splitlines():
        pieces = ATT.split(raw)
        # ATT.split alternates text, id, text, id, ...
        for i, piece in enumerate(pieces):
            if i % 2:
                pending.append(piece)
                continue
            line = piece.strip()
            if not line:
                continue
            if m := HEADER.match(line):
                n = int(m.group(1))
                if n not in obs:
                    obs[n] = {"header": json.loads(m.group(2)), "lines": [], "frames": []}
                    order.append(n)
                obs[n]["frames"] += pending
                pending = []
            elif m := LINE.match(line):
                n = int(m.group(1))
                if n in obs:
                    obs[n]["lines"].append(m.group(2))
    return [obs[n] for n in order]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", default="bench-v0")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)

    docs = psql(
        "select id || E'\\x1f' || replace(original_text, E'\\x1e', '') || E'\\x1e' from documents "
        f"where bank_id='{args.bank}' and id like 'activity:discrete-replay:%' order by id",
        args.port,
    )
    keys = dict(
        line.split("|", 1)
        for line in psql(f"select short_id || '|' || storage_key from attachments where bank_id='{args.bank}'",
                         args.port).splitlines()
        if line
    )
    n_obs = n_frames = 0
    with (out / "manifest.jsonl").open("w") as manifest:
        for record in docs.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            doc_id, text = record.split("\x1f", 1)
            first = text.splitlines()[0]
            ep_header = json.loads(first.split(" ", 1)[1])
            group = ep_header["episode_id"].split("-p")[0]
            for o in parse(text):
                h = o["header"]
                draft = {
                    "source": "screen", "origin": "user", "event_kind": h.get("trigger") or "replay",
                    "session_id": ep_header["session_id"],
                    "started_at": datetime.fromisoformat(h["started_at"]).timestamp(),
                    "ended_at": datetime.fromisoformat(h["ended_at"]).timestamp(),
                    "app": h.get("app", ""), "bundle_id": h.get("bundle_id", ""),
                    "window_title": h.get("window_title", ""), "url": h.get("url", ""),
                    "site": site_of(h.get("url", "")), "screen_text": "\n".join(o["lines"]),
                    "text_source": h.get("text_source", ""),
                    **{k: int(h.get(k) or 0) for k in ("keys", "clicks", "scrolls", "clipboard")},
                }
                item = {"label": f"{group}-{n_obs}", "group": group, "draft": draft}
                if o["frames"] and o["frames"][-1] in keys:
                    short = o["frames"][-1]
                    path = out / "frames" / f"{short}.jpg"
                    if not path.exists():
                        b64 = psql(f"select encode(data, 'base64') from file_storage "
                                   f"where storage_key='{keys[short]}'", args.port)
                        path.write_bytes(base64.b64decode(b64.replace("\n", "")))
                    item["image"] = str(path)
                    n_frames += 1
                manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_obs += 1
    print(f"{n_obs} observations, {n_frames} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
