"""One-off migration: replay the WorkAA prototype's observations through Stage 2.

No capture and no vision call. WorkAA's `app` was model-written (insights.md
§7.9): an `app` that looks like a web address becomes `site`, with app
"Browser"; a real application name is kept. `dense_text` stands in for
screen_text, and interaction is approximated from event_kind.

    python tools/replay_workaa.py --fresh            # → $DISCRETE_DATA_DIR/replay-workaa.db
    discrete --db ~/.cache/discrete/replay-workaa.db eval --cases golden/workaa_replay.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from discrete import textkit as tk
from discrete.config import data_dir
from discrete.llm import flush_usage
from discrete.models import ObservationDraft
from discrete.process.consolidate import run_consolidation
from discrete.process.embed_worker import drain_outbox, embed_row, observation_text
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.store import Store


def app_and_site(raw_app: str, url: str) -> tuple[str, str]:
    """A domain-looking `app` was a model's guess at the site, not an application."""
    site = tk.site_from_url(url if url.startswith("http") else f"https://{url}") if url else ""
    if tk.looks_like_domain(raw_app):
        return "Browser", site or tk.site_from_url(f"https://{raw_app}")
    return (raw_app or ("Browser" if site else "")), site


async def replay(source: Path, target: Path, *, limit: int, consolidate: bool) -> int:
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM observations WHERE origin='user' ORDER BY id").fetchall()[: limit or None]
    store = await Store.open(str(target))
    tracker = EpisodeTracker(store)
    new = 0
    try:
        for r in rows:
            ts = datetime.fromisoformat(r["created_at"]).replace(tzinfo=UTC).timestamp()
            url = r["url"] or ""
            app, site = app_and_site(r["app"] or "", url)
            draft = ObservationDraft(
                source="replay", event_kind=r["event_kind"], session_id=r["session_id"],
                started_at=ts, ended_at=ts + 5, app=app, window_title=r["window_title"] or "", url=url, site=site,
                screen_text=r["dense_text"] or "", text_source="replay", caption=r["caption"] or "",
                keys=1 if r["event_kind"] == "keyboard" else 0, clicks=1 if r["event_kind"] == "mouse" else 0,
            )
            oid, is_new = await store.add_observation(draft)
            if is_new and oid:
                new += 1
                row = await store.get_observation(oid)
                vec = await embed_row(store, "observation", oid, observation_text(row))
                await tracker.on_observation(oid, row, session_id=draft.session_id, vec=vec)
        await tracker.close_current()
        print(f"ingested {new}/{len(rows)} observations (rest collapsed as duplicates)")
        pipe = ProcessPipeline(store)
        while batch := await pipe.process_ready(limit=5):
            for res in batch:
                print(f"episode {res.episode_id}: +{res.added} ~{res.reinforced} !{res.invalidated} "
                      f"rejected={res.rejected}")
        print(f"embedded {await drain_outbox(store)}")
        if consolidate:
            print(json.dumps(await run_consolidation(store), default=str))
    finally:
        await flush_usage(store)
        await store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--source", default="~/.cache/workaa/workaa.db")
    ap.add_argument("--db", default=str(data_dir() / "replay-workaa.db"), help="store to build")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="delete the target store first")
    ap.add_argument("--no-consolidate", action="store_true")
    args = ap.parse_args()
    target = Path(args.db).expanduser()
    if target.exists():
        if not args.fresh:
            print(f"{target} exists; pass --fresh to rebuild it", file=sys.stderr)
            return 1
        for suffix in ("", "-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
    return asyncio.run(replay(Path(args.source).expanduser(), target, limit=args.limit,
                              consolidate=not args.no_consolidate))


if __name__ == "__main__":
    sys.exit(main())
