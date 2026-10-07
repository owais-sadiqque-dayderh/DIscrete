"""Copy a rebuilt store and clear everything drafting produced, so the current code drafts it again.

A full replay pays for every caption again; when only drafting, reconcile, trails or
consolidation changed, the captures and their captions are the same and only the memory
built on them needs rebuilding. On the copy this keeps observations, captions, sightings,
episodes (boundaries) and entity *candidates* (they come from capture), and deletes
claims and everything derived from them: evidence, mentions, links/trails, the change
log, digests, gaps, promoted entities, aliases, edges, their embeddings and outbox rows.
Episodes go back to ``ready``; then run

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
    "entity_edges", "entity_aliases", "entities", "suppressions", "retrieval_misses", "claims",
    "episode_draft_chunks",
)
DERIVED_KINDS = ("claim", "entity", "digest")


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
        counts["episodes"] = conn.execute(
            "UPDATE episodes SET status='ready', title='', summary='', draft_attempts=0"
        ).rowcount
    for fts in ("claims_fts", "episodes_fts", "entities_fts"):
        if fts in tables:
            try:
                with conn:
                    conn.execute(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                pass  # not an external-content index: its delete triggers already kept it in sync
    conn.execute("VACUUM")
    conn.close()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(redraft(args.src, args.out))
