"""Copy a rebuilt store and clear everything drafting produced, so the current code drafts it again.

A full replay pays for every caption again; when only drafting, reconcile, trails or
consolidation changed, the captures and their captions are the same and only the memory
built on them needs rebuilding. On the copy this keeps observations, captions, sightings,
episodes (boundaries) and entity *candidates* (they come from capture), and deletes
claims and everything derived from them: evidence, mentions, links/trails, the change
log, digests, gaps, promoted entities, aliases, edges, tasks, derived embeddings,
outbox rows and coverage records. Non-merged episodes go back to ``ready`` with
their boundaries locked against further pre-draft merging; then run

    .venv/bin/python -m discrete.cli --db OUT process --recover --consolidate

    .venv/bin/python bench/redraft.py bench/private/g46/branch.db bench/private/g46/branch-v2.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

#: Tables wholly derived from drafting and consolidation.
DERIVED = (
    "evidence", "mentions", "links", "memory_event_refs", "memory_events", "digests", "gaps",
    "entity_edges", "entity_aliases", "entities_fts", "entities", "suppressions", "retrieval_misses", "claims",
    "episode_draft_chunks", "episode_tasks", "tasks",
)
DERIVED_KINDS = ("claim", "entity", "digest", "episode", "task")


def redraft(src: Path, out: Path) -> dict[str, int]:
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    with sqlite3.connect(src) as a, sqlite3.connect(out) as b:
        a.backup(b)
    conn = sqlite3.connect(out)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts: dict[str, int] = {}
    with conn:
        for table in DERIVED:
            if table in tables:
                counts[table] = conn.execute(f"DELETE FROM {table}").rowcount  # noqa: S608 - fixed names
        marks = ",".join("?" * len(DERIVED_KINDS))
        counts["embeddings"] = conn.execute(f"DELETE FROM embeddings WHERE ref_kind IN ({marks})", DERIVED_KINDS).rowcount
        conn.execute(f"DELETE FROM index_outbox WHERE ref_kind IN ({marks})", DERIVED_KINDS)
        # The dense-index cache is keyed on this counter: bump it so no stale matrix is reused.
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key LIKE 'vectors:%'")
        conn.execute("DELETE FROM meta WHERE key IN ('last_consolidation', 'last_dream')")
        conn.execute("DELETE FROM meta WHERE key LIKE 'draft_coverage:%'")
        # Resetting attempts removes the merger's usual prior-draft guard. Lock
        # every retained boundary so a redraft remains a drafting-only rebuild.
        conn.execute("INSERT OR IGNORE INTO meta(key,value) "
                     "SELECT 'episode_boundary_locked:' || id, '1' FROM episodes WHERE status!='merged'")
        counts["episodes"] = conn.execute(
            "UPDATE episodes SET status='ready', title='', summary='', apps='[]', sites='[]', "
            "engaged_fraction=0, draft_attempts=0 WHERE status!='merged'"
        ).rowcount
    # Entity FTS owns its content (no delete triggers): clear it above. Only the
    # external-content indexes can rebuild from their emptied/reset source rows.
    for fts in ("claims_fts", "episodes_fts"):
        if fts in tables:
            with conn:
                conn.execute(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
    conn.execute("VACUUM")
    conn.close()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(redraft(args.src, args.out))
