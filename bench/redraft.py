"""Copy a rebuilt store and clear everything drafting produced, so the current code drafts it again.

A full replay pays for every caption again; when only drafting, reconcile, trails or
consolidation changed, the captures and their captions are the same and only the memory
built on them needs rebuilding. On the copy this keeps observations, captions, sightings,
episodes (boundaries), suppressions, user forget/correct events and their referenced
claim lineages. Other claims, evidence, mentions, links/trails, change-log rows,
digests, gaps, promoted entities, aliases, edges, tasks, derived embeddings,
outbox rows and coverage records are cleared. Untagged entity candidates are rebuilt
from stored captures. User undo/redo or unsupported directives refuse the rebuild.
Non-merged episodes go back to ``ready`` with
their boundaries locked against further pre-draft merging; then run

    .venv/bin/python -m discrete.cli --db OUT process --recover --consolidate

    .venv/bin/python bench/redraft.py bench/private/g46/branch.db bench/private/g46/branch-v2.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from discrete import textkit as tk
from discrete.encryption import open_database
from discrete.process.entities import structural_candidates
from discrete.store.db import clone_database, migrate

#: Tables cleared except for rows required by preserved user directives.
DERIVED = (
    "evidence", "mentions", "links", "memory_event_refs", "memory_events", "digests", "gaps",
    "entity_edges", "entity_aliases", "entities_fts", "entities", "retrieval_misses", "claims",
    "episode_draft_chunks", "episode_tasks", "tasks",
)
DERIVED_KINDS = ("claim", "entity", "digest", "episode", "task")


def _preserve_directives(conn: sqlite3.Connection, tables: set[str]) -> None:
    """Retain the original IDs and lineage inputs needed to read/undo directives."""
    conn.execute("CREATE TEMP TABLE kept_claims(id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE kept_events(id INTEGER PRIMARY KEY)")
    if 'memory_events' in tables:
        conn.execute("INSERT INTO kept_events SELECT id FROM memory_events WHERE actor='user' "
                     "AND (op IN ('forget','correct') OR (op='add' AND claim_id IN "
                     "(SELECT id FROM claims WHERE modality='stated')))")
        conn.execute("INSERT OR IGNORE INTO kept_claims SELECT claim_id FROM memory_event_refs "
                     "WHERE event_id IN (SELECT id FROM kept_events)")
    if 'suppressions' in tables:
        conn.execute("INSERT OR IGNORE INTO kept_claims SELECT claim_id FROM suppressions WHERE claim_id IS NOT NULL")
    # A correction points back to its old version. Keep lineage and provenance
    # closure rather than leaving JSON IDs or state changes pointing at deleted rows.
    while True:
        inserted = conn.execute(
            "INSERT OR IGNORE INTO kept_claims SELECT id FROM claims WHERE lineage_id IN "
            "(SELECT lineage_id FROM claims WHERE id IN (SELECT id FROM kept_claims)) "
            "OR id IN (SELECT superseded_by FROM claims WHERE id IN (SELECT id FROM kept_claims)) "
            "OR id IN (SELECT refines_id FROM claims WHERE id IN (SELECT id FROM kept_claims)) "
            "OR id IN (SELECT invalidates_id FROM claims WHERE id IN (SELECT id FROM kept_claims)) "
            "OR id IN (SELECT j.value FROM claims c, json_each(c.derived_from) j "
            "WHERE c.id IN (SELECT id FROM kept_claims))").rowcount
        if not inserted:
            break


def _rebuild_candidates(conn: sqlite3.Connection) -> None:
    """Candidates have no source tag: reconstruct only the capture-side inputs."""
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT * FROM observations WHERE origin='user' AND episode_id IS NOT NULL"):
        names = [n for n in tk.parse_json_list(row['entities']) if isinstance(n, str)]
        if row['text_source'] == 'ax':
            names += structural_candidates(row['screen_text'])
        if row['keys'] or row['clicks'] or row['clipboard']:
            names += [n for n in (row['app'], row['site']) if n]
        for name in names:
            key = tk.entity_key(name)
            if not key:
                continue
            existing = conn.execute("SELECT episodes FROM entity_candidates WHERE name_norm=?", (key,)).fetchone()
            eps = sorted(set(tk.parse_json_list(existing[0]) if existing else []) | {row['episode_id']})
            conn.execute("INSERT INTO entity_candidates(name_norm,name,episodes,first_seen,last_seen) VALUES (?,?,?,?,?) "
                         "ON CONFLICT(name_norm) DO UPDATE SET episodes=excluded.episodes, "
                         "first_seen=MIN(first_seen,excluded.first_seen), last_seen=MAX(last_seen,excluded.last_seen)",
                         (key, name, json.dumps(eps), row['started_at'], row['ended_at']))
    conn.row_factory = None


def redraft(src: Path, out: Path) -> dict[str, int]:
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    with closing(open_database(str(src), read_only=True)) as a:
        tables = {r[0] for r in a.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'memory_events' in tables:
            unsupported = a.execute("SELECT id,op FROM memory_events WHERE actor='user' "
                                    "AND (op NOT IN ('add','forget','correct') OR undone_by IS NOT NULL)").fetchall()
            if unsupported:
                raise SystemExit(f"cannot redraft user directives with undo/unsupported operations: {[tuple(r) for r in unsupported]}")
    clone_database(src, out)
    conn = open_database(str(out))
    # Upgrade only the copy before using the episode state columns.
    migrate(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts: dict[str, int] = {}
    with conn:
        conn.execute("BEGIN")
        _preserve_directives(conn, tables)
        retained = {
            'claims': 'id IN (SELECT id FROM kept_claims)',
            'evidence': 'claim_id IN (SELECT id FROM kept_claims)',
            'memory_events': 'id IN (SELECT id FROM kept_events)',
            'memory_event_refs': 'event_id IN (SELECT id FROM kept_events)',
            'links': 'event_id IN (SELECT id FROM kept_events)',
        }
        for table in DERIVED:
            if table in tables:
                where = f" WHERE NOT ({retained[table]})" if table in retained else ''
                # NULL event_id is draft-derived too.
                if table == 'links':
                    where = ' WHERE event_id IS NULL OR NOT (' + retained[table] + ')'
                counts[table] = conn.execute(f"DELETE FROM {table}{where}").rowcount  # noqa: S608 - fixed names
        counts['entity_candidates'] = conn.execute("DELETE FROM entity_candidates").rowcount
        _rebuild_candidates(conn)
        marks = ",".join("?" * len(DERIVED_KINDS))
        counts["embeddings"] = conn.execute(f"DELETE FROM embeddings WHERE ref_kind IN ({marks})", DERIVED_KINDS).rowcount
        conn.execute(f"DELETE FROM index_outbox WHERE ref_kind IN ({marks})", DERIVED_KINDS)
        conn.execute("INSERT INTO index_outbox(ref_kind,ref_id,enqueued_at,attempts) "
                     "SELECT 'claim',id,updated_at,0 FROM claims WHERE status='live'")
        # The dense-index cache is keyed on this counter: bump it so no stale matrix is reused.
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key LIKE 'vectors:%'")
        conn.execute("DELETE FROM meta WHERE key IN ('last_consolidation', 'last_dream')")
        conn.execute("DELETE FROM meta WHERE key LIKE 'draft_coverage:%'")
        # Resetting attempts removes the merger's usual prior-draft guard. Lock
        # every retained boundary so a redraft remains a drafting-only rebuild.
        counts["episodes"] = conn.execute(
            "UPDATE episodes SET status='ready', title='', summary='', apps='[]', sites='[]', "
            "engaged_fraction=0, draft_attempts=0, boundary_locked=1 WHERE status!='merged'"
        ).rowcount
    # Entity FTS owns its content (no delete triggers): clear it above. Only the
    # external-content indexes can rebuild from their emptied/reset source rows.
    for fts in ("claims_fts", "episodes_fts"):
        if fts in tables:
            with conn:
                conn.execute("BEGIN")
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
