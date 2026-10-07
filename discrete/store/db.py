"""SQLite schema and numbered migrations. Timestamps are epoch floats (see timeutil).

`PRAGMA user_version` is the applied migration count. Each migration runs once,
in order, inside its own transaction; a new schema change is a new function
appended to MIGRATIONS, never an edit to an old one.
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from collections.abc import Callable

log = logging.getLogger("discrete.store.db")

_FTS_TOKENIZE = "tokenize='porter unicode61 remove_diacritics 2'"

_BASE = f"""
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS metrics(name TEXT PRIMARY KEY, value REAL NOT NULL DEFAULT 0, updated_at REAL);

CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  origin TEXT NOT NULL DEFAULT 'user',
  event_kind TEXT NOT NULL,
  session_id TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL NOT NULL,
  app TEXT NOT NULL DEFAULT '',
  bundle_id TEXT NOT NULL DEFAULT '',
  window_title TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  site TEXT NOT NULL DEFAULT '',
  screen_text TEXT NOT NULL DEFAULT '',
  text_source TEXT NOT NULL DEFAULT '',
  caption TEXT NOT NULL DEFAULT '',
  entities TEXT NOT NULL DEFAULT '[]',
  keys INTEGER NOT NULL DEFAULT 0,
  clicks INTEGER NOT NULL DEFAULT 0,
  scrolls INTEGER NOT NULL DEFAULT 0,
  clipboard INTEGER NOT NULL DEFAULT 0,
  frame_phash TEXT NOT NULL DEFAULT '',
  frame_path TEXT NOT NULL DEFAULT '',
  content_hash TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  dup_count INTEGER NOT NULL DEFAULT 1,
  episode_id INTEGER,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_obs_content_hash ON observations(content_hash);
CREATE INDEX IF NOT EXISTS ix_obs_text_hash ON observations(text_hash, app, window_title);
CREATE INDEX IF NOT EXISTS ix_obs_started ON observations(started_at);
CREATE INDEX IF NOT EXISTS ix_obs_episode ON observations(episode_id);

CREATE TABLE IF NOT EXISTS obs_minhash(
  obs_id INTEGER PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
  sig BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
  caption, screen_text, window_title, site, app,
  content='observations', content_rowid='id', {_FTS_TOKENIZE});
CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
  INSERT INTO observations_fts(rowid, caption, screen_text, window_title, site, app)
  VALUES (new.id, new.caption, new.screen_text, new.window_title, new.site, new.app);
END;
CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, caption, screen_text, window_title, site, app)
  VALUES ('delete', old.id, old.caption, old.screen_text, old.window_title, old.site, old.app);
END;
CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE OF caption, screen_text, window_title, site, app ON observations BEGIN
  INSERT INTO observations_fts(observations_fts, rowid, caption, screen_text, window_title, site, app)
  VALUES ('delete', old.id, old.caption, old.screen_text, old.window_title, old.site, old.app);
  INSERT INTO observations_fts(rowid, caption, screen_text, window_title, site, app)
  VALUES (new.id, new.caption, new.screen_text, new.window_title, new.site, new.app);
END;

CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL NOT NULL,
  apps TEXT NOT NULL DEFAULT '[]',
  sites TEXT NOT NULL DEFAULT '[]',
  title TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  engaged_fraction REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'open',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ep_status ON episodes(status);
CREATE INDEX IF NOT EXISTS ix_ep_time ON episodes(started_at, ended_at);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
  title, summary, content='episodes', content_rowid='id', {_FTS_TOKENIZE});
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.id, new.title, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.id, old.title, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF title, summary ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, title, summary) VALUES ('delete', old.id, old.title, old.summary);
  INSERT INTO episodes_fts(rowid, title, summary) VALUES (new.id, new.title, new.summary);
END;

CREATE TABLE IF NOT EXISTS claims(
  id INTEGER PRIMARY KEY,
  lineage_id TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  context TEXT NOT NULL DEFAULT '',
  reasoning TEXT NOT NULL DEFAULT '',
  modality TEXT NOT NULL,
  occurred_from REAL NOT NULL,
  occurred_to REAL NOT NULL,
  valid_from REAL NOT NULL,
  valid_to REAL,
  status TEXT NOT NULL DEFAULT 'live',
  superseded_by INTEGER,
  refines_id INTEGER,
  invalidates_id INTEGER,
  episode_id INTEGER,
  confidence REAL NOT NULL DEFAULT 0,
  flag TEXT NOT NULL DEFAULT '',
  derived_from TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_claims_status ON claims(status, kind);
CREATE INDEX IF NOT EXISTS ix_claims_lineage ON claims(lineage_id);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
  text, context, content='claims', content_rowid='id', {_FTS_TOKENIZE});
CREATE TRIGGER IF NOT EXISTS claims_ai AFTER INSERT ON claims BEGIN
  INSERT INTO claims_fts(rowid, text, context) VALUES (new.id, new.text, new.context);
END;
CREATE TRIGGER IF NOT EXISTS claims_ad AFTER DELETE ON claims BEGIN
  INSERT INTO claims_fts(claims_fts, rowid, text, context) VALUES ('delete', old.id, old.text, old.context);
END;
CREATE TRIGGER IF NOT EXISTS claims_au AFTER UPDATE OF text, context ON claims BEGIN
  INSERT INTO claims_fts(claims_fts, rowid, text, context) VALUES ('delete', old.id, old.text, old.context);
  INSERT INTO claims_fts(rowid, text, context) VALUES (new.id, new.text, new.context);
END;

CREATE TABLE IF NOT EXISTS evidence(
  claim_id INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
  observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  quote TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  PRIMARY KEY (claim_id, observation_id)
);
CREATE INDEX IF NOT EXISTS ix_evidence_obs ON evidence(observation_id);

CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'unknown',
  summary TEXT NOT NULL DEFAULT '',
  page_updated_at REAL NOT NULL DEFAULT 0,
  merged_into INTEGER,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_aliases(
  alias_norm TEXT PRIMARY KEY,
  entity_id INTEGER NOT NULL REFERENCES entities(id),
  alias TEXT NOT NULL,
  norm_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alias_entity ON entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS ix_alias_key ON entity_aliases(norm_key);
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(name, aliases, summary, {_FTS_TOKENIZE});

CREATE TABLE IF NOT EXISTS entity_candidates(
  name_norm TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  episodes TEXT NOT NULL DEFAULT '[]',
  first_seen REAL NOT NULL,
  last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions(
  entity_id INTEGER NOT NULL,
  ref_kind TEXT NOT NULL,
  ref_id INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (entity_id, ref_kind, ref_id)
);
CREATE INDEX IF NOT EXISTS ix_mentions_ref ON mentions(ref_kind, ref_id);
CREATE TABLE IF NOT EXISTS entity_edges(
  a INTEGER NOT NULL,
  b INTEGER NOT NULL,
  weight REAL NOT NULL DEFAULT 0,
  last_seen REAL NOT NULL,
  PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS embeddings(
  ref_kind TEXT NOT NULL,
  ref_id INTEGER NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  text_hash TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (ref_kind, ref_id)
);
CREATE TABLE IF NOT EXISTS index_outbox(
  ref_kind TEXT NOT NULL,
  ref_id INTEGER NOT NULL,
  enqueued_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (ref_kind, ref_id)
);

CREATE TABLE IF NOT EXISTS token_df(token TEXT PRIMARY KEY, df INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS digests(day TEXT PRIMARY KEY, text TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS gaps(
  id INTEGER PRIMARY KEY,
  question TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  answer TEXT NOT NULL DEFAULT '',
  claim_id INTEGER,
  created_at REAL NOT NULL,
  answered_at REAL
);
CREATE TABLE IF NOT EXISTS retrieval_misses(
  id INTEGER PRIMARY KEY,
  query TEXT NOT NULL,
  missed TEXT NOT NULL,
  stage TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


def _add_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Stores created before numbered migrations may already have the column."""
    if col not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def m001_base(conn: sqlite3.Connection) -> None:
    for stmt in _statements(_BASE):
        conn.execute(stmt)


def m002_draft_attempts_and_digest_counts(conn: sqlite3.Connection) -> None:
    _add_column(conn, "episodes", "draft_attempts", "INTEGER NOT NULL DEFAULT 0")
    _add_column(conn, "digests", "n_episodes", "INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_obs_app_site ON observations(app, site, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_mentions_entity ON mentions(entity_id)")


def m003_structured_page_fields(conn: sqlite3.Connection) -> None:
    """Page identity as columns from the vision call, deferred enrichment state,
    and page_title in the observation FTS index."""
    _add_column(conn, "observations", "page_title", "TEXT NOT NULL DEFAULT ''")
    _add_column(conn, "observations", "is_listing", "INTEGER NOT NULL DEFAULT 0")
    _add_column(conn, "observations", "navigated_from", "TEXT NOT NULL DEFAULT ''")
    _add_column(conn, "observations", "enrich_state", "TEXT NOT NULL DEFAULT ''")
    for trig in ("observations_ai", "observations_ad", "observations_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.execute("DROP TABLE IF EXISTS observations_fts")
    cols = "caption, screen_text, window_title, site, app, page_title"
    new = ", ".join(f"new.{c.strip()}" for c in cols.split(","))
    old = ", ".join(f"old.{c.strip()}" for c in cols.split(","))
    for stmt in (
        f"CREATE VIRTUAL TABLE observations_fts USING fts5({cols}, "
        f"content='observations', content_rowid='id', {_FTS_TOKENIZE})",
        f"CREATE TRIGGER observations_ai AFTER INSERT ON observations BEGIN "
        f"INSERT INTO observations_fts(rowid, {cols}) VALUES (new.id, {new}); END",
        f"CREATE TRIGGER observations_ad AFTER DELETE ON observations BEGIN "
        f"INSERT INTO observations_fts(observations_fts, rowid, {cols}) VALUES ('delete', old.id, {old}); END",
        f"CREATE TRIGGER observations_au AFTER UPDATE OF {cols} ON observations BEGIN "
        f"INSERT INTO observations_fts(observations_fts, rowid, {cols}) VALUES ('delete', old.id, {old}); "
        f"INSERT INTO observations_fts(rowid, {cols}) VALUES (new.id, {new}); END",
        "INSERT INTO observations_fts(observations_fts) VALUES ('rebuild')",
        "CREATE INDEX IF NOT EXISTS ix_obs_enrich ON observations(enrich_state) WHERE enrich_state != ''",
    ):
        conn.execute(stmt)


def m004_minhash_raw_bytes(conn: sqlite3.Connection) -> None:
    """Pickled datasketch objects → raw uint64 hash values. A pickle is slow to load
    and breaks when datasketch changes its class layout; the bytes do not. Runs
    once, before which every stored signature was a pickle."""
    for oid, sig in conn.execute("SELECT obs_id, sig FROM obs_minhash").fetchall():
        try:
            raw = pickle.loads(sig).hashvalues.astype("<u8").tobytes()
        except Exception as exc:  # unreadable: the row just loses near-dup matching
            log.warning("dropping unreadable minhash for observation %s: %s", oid, exc)
            conn.execute("DELETE FROM obs_minhash WHERE obs_id=?", (oid,))
            continue
        conn.execute("UPDATE obs_minhash SET sig=? WHERE obs_id=?", (raw, oid))


def m005_perf_log(conn: sqlite3.Connection) -> None:
    """Step-0 measurement (docs/C7.md): one row per model call, one per stage sample
    or event count. `src` separates the daemon ("live") from ask/eval/query runs."""
    conn.execute("""CREATE TABLE perf_calls(
      id INTEGER PRIMARY KEY,
      at REAL NOT NULL,
      src TEXT NOT NULL,
      component TEXT NOT NULL,
      model TEXT NOT NULL DEFAULT '',
      status TEXT NOT NULL,
      ms REAL NOT NULL,
      first_ms REAL,
      prompt_tokens INTEGER NOT NULL DEFAULT 0,
      completion_tokens INTEGER NOT NULL DEFAULT 0,
      cost REAL)""")
    conn.execute("CREATE INDEX ix_perf_calls_at ON perf_calls(at)")
    conn.execute("""CREATE TABLE perf_samples(
      id INTEGER PRIMARY KEY,
      at REAL NOT NULL,
      src TEXT NOT NULL,
      stage TEXT NOT NULL,
      ms REAL,
      n INTEGER,
      ref_id INTEGER)""")
    conn.execute("CREATE INDEX ix_perf_samples_at ON perf_samples(at)")


def m006_unsegmented_index(conn: sqlite3.Connection) -> None:
    # The runtime segments from the store (oldest unsegmented user row first). Without
    # this, each lookup would scan every discarded and self-capture row too.
    conn.execute("CREATE INDEX ix_obs_unsegmented ON observations(id) WHERE origin='user' AND episode_id IS NULL")


def m007_memory_evolution(conn: sqlite3.Connection) -> None:
    """The change log (store/history.py), typed links between memories (causal trails
    and the graph, store/links.py) and suppressions for forgotten claims."""
    conn.execute("""CREATE TABLE memory_events(
      id INTEGER PRIMARY KEY,
      at REAL NOT NULL,
      op TEXT NOT NULL,
      actor TEXT NOT NULL,
      claim_id INTEGER,
      created TEXT NOT NULL DEFAULT '[]',
      changes TEXT NOT NULL DEFAULT '[]',
      reason TEXT NOT NULL DEFAULT '',
      run_id TEXT NOT NULL DEFAULT '',
      data TEXT NOT NULL DEFAULT '{}',
      undone_by INTEGER)""")
    conn.execute("CREATE INDEX ix_events_at ON memory_events(at)")
    conn.execute("""CREATE TABLE memory_event_refs(
      event_id INTEGER NOT NULL REFERENCES memory_events(id) ON DELETE CASCADE,
      claim_id INTEGER NOT NULL,
      PRIMARY KEY (event_id, claim_id))""")
    conn.execute("CREATE INDEX ix_event_refs_claim ON memory_event_refs(claim_id)")
    conn.execute("""CREATE TABLE links(
      id INTEGER PRIMARY KEY,
      src_kind TEXT NOT NULL,
      src_id INTEGER NOT NULL,
      dst_kind TEXT NOT NULL,
      dst_id INTEGER NOT NULL,
      relation TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0,
      reason TEXT NOT NULL DEFAULT '',
      evidence TEXT NOT NULL DEFAULT '[]',
      status TEXT NOT NULL DEFAULT 'live',
      event_id INTEGER,
      created_at REAL NOT NULL,
      UNIQUE (src_kind, src_id, dst_kind, dst_id, relation))""")
    conn.execute("CREATE INDEX ix_links_dst ON links(dst_kind, dst_id, relation)")
    conn.execute("CREATE INDEX ix_links_src ON links(src_kind, src_id, relation)")
    conn.execute("""CREATE TABLE suppressions(
      id INTEGER PRIMARY KEY,
      kind TEXT NOT NULL,
      text TEXT NOT NULL,
      norm_key TEXT NOT NULL,
      vector BLOB,
      claim_id INTEGER,
      event_id INTEGER,
      created_at REAL NOT NULL)""")
    conn.execute("CREATE INDEX ix_suppressions_key ON suppressions(norm_key)")
    _add_column(conn, "claims", "action", "TEXT NOT NULL DEFAULT ''")


def m008_sightings(conn: sqlite3.Connection) -> None:
    """Every time a window was seen, including captures collapsed into an earlier row as
    duplicates, which used to keep only a widened interval. Attention (time on a site,
    app or page, and whether the user was acting) is the strongest implicit preference
    signal, and it needs these (store/attention.py)."""
    conn.execute("""CREATE TABLE sightings(
      id INTEGER PRIMARY KEY,
      obs_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
      started_at REAL NOT NULL,
      ended_at REAL NOT NULL,
      keys INTEGER NOT NULL DEFAULT 0,
      clicks INTEGER NOT NULL DEFAULT 0,
      scrolls INTEGER NOT NULL DEFAULT 0,
      clipboard INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("CREATE INDEX ix_sightings_at ON sightings(started_at)")
    conn.execute("CREATE INDEX ix_sightings_obs ON sightings(obs_id)")
    conn.execute("INSERT INTO sightings(obs_id, started_at, ended_at, keys, clicks, scrolls, clipboard) "
                 "SELECT id, started_at, ended_at, keys, clicks, scrolls, clipboard FROM observations "
                 "WHERE origin='user'")


def m009_evidence_field(conn: sqlite3.Connection) -> None:
    """Which field of the observation a quote was found in (screen_text, caption, title,
    url): screen text is what the user saw verbatim; a caption is a model's description."""
    _add_column(conn, "evidence", "field", "TEXT NOT NULL DEFAULT ''")


def m010_episode_draft_chunks(conn: sqlite3.Connection) -> None:
    """Checkpoint successful draft calls independently of the final claims transaction."""
    conn.execute("""CREATE TABLE episode_draft_chunks(
      episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      input_hash TEXT NOT NULL,
      result TEXT NOT NULL,
      PRIMARY KEY (episode_id, chunk_index))""")
    # Both normal drafting and the no-model path close episodes. Clear checkpoints
    # atomically with closure, so a failed final transaction keeps them for retry.
    conn.execute("""CREATE TRIGGER episodes_clear_draft_chunks AFTER UPDATE OF status ON episodes
      WHEN new.status='closed' BEGIN
        DELETE FROM episode_draft_chunks WHERE episode_id=new.id;
      END""")


def m011_tasks(conn: sqlite3.Connection) -> None:
    """Tasks span drafted episodes; each episode belongs to at most one task."""
    conn.execute("""CREATE TABLE tasks(
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'active',
      first_seen REAL NOT NULL,
      last_seen REAL NOT NULL)""")
    conn.execute("CREATE INDEX ix_tasks_time ON tasks(last_seen)")
    conn.execute("""CREATE TABLE episode_tasks(
      episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
      task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
      entities TEXT NOT NULL DEFAULT '[]')""")
    conn.execute("CREATE INDEX ix_episode_tasks_task ON episode_tasks(task_id)")


def m012_episode_draft_state(conn: sqlite3.Connection) -> None:
    """Keep per-episode grace timers and boundary locks with their owning rows."""
    _add_column(conn, "episodes", "ready_since", "REAL")
    _add_column(conn, "episodes", "boundary_locked", "INTEGER NOT NULL DEFAULT 0")
    conn.execute("UPDATE episodes SET ready_since=(SELECT CAST(value AS REAL) FROM meta "
                 "WHERE key='episode_ready_since:' || episodes.id), "
                 "boundary_locked=EXISTS(SELECT 1 FROM meta WHERE key='episode_boundary_locked:' || episodes.id)")
    # Locks previously depended on key presence, even with a NULL or '0' value.
    # Drop orphaned state too; deleting an episode now deletes its state with it.
    conn.execute("DELETE FROM meta WHERE key GLOB 'episode_ready_since:*' "
                 "OR key GLOB 'episode_boundary_locked:*'")
    conn.execute("CREATE INDEX ix_ep_status_time ON episodes(status, started_at, id)")
    conn.execute("CREATE INDEX ix_ep_unmerged ON episodes(started_at, id) WHERE status!='merged'")


def m013_attention_provenance(conn: sqlite3.Connection) -> None:
    """Quoted, deterministic visit-to-visit connections; cascade with raw evidence."""
    conn.execute("""CREATE TABLE attention_provenance(
      source INTEGER NOT NULL REFERENCES sightings(id) ON DELETE CASCADE,
      target INTEGER NOT NULL REFERENCES sightings(id) ON DELETE CASCADE,
      kind TEXT NOT NULL,
      confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
      source_field TEXT NOT NULL, source_quote TEXT NOT NULL,
      target_field TEXT NOT NULL, target_quote TEXT NOT NULL,
      gap_s REAL NOT NULL CHECK(gap_s >= 0),
      CHECK(source != target), PRIMARY KEY(source, target, kind))""")
    conn.execute("CREATE INDEX ix_provenance_target ON attention_provenance(target, confidence DESC)")
    conn.execute("""CREATE TABLE provenance_inputs(
      sighting_id INTEGER PRIMARY KEY REFERENCES sightings(id) ON DELETE CASCADE,
      updated_at REAL NOT NULL, rules_key TEXT NOT NULL)""")
    # Enrichment/redaction changes can invalidate either end. Rebuild destinations
    # in their bounded neighborhood, including out-of-order capture imports.
    conn.execute("""CREATE TRIGGER provenance_observation_changed AFTER UPDATE OF
      screen_text, caption, window_title, page_title, url, navigated_from, origin ON observations BEGIN
        DELETE FROM provenance_inputs WHERE sighting_id IN
          (SELECT target FROM attention_provenance WHERE source IN (SELECT id FROM sightings WHERE obs_id=new.id));
        DELETE FROM attention_provenance WHERE source IN (SELECT id FROM sightings WHERE obs_id=new.id)
          OR target IN (SELECT id FROM sightings WHERE obs_id=new.id);
        DELETE FROM provenance_inputs WHERE sighting_id IN (SELECT id FROM sightings WHERE obs_id=new.id);
      END""")


def m014_pages(conn: sqlite3.Connection) -> None:
    """Materialized page totals, with transactional invalidation of raw inputs."""
    conn.execute("""CREATE TABLE pages(
      id INTEGER PRIMARY KEY, canonical_url TEXT NOT NULL UNIQUE,
      site TEXT NOT NULL, title TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('article','repo','video','product','chat','file','listing','search','doc','other')),
      first_seen REAL NOT NULL DEFAULT 0, last_seen REAL NOT NULL DEFAULT 0,
      visits INTEGER NOT NULL DEFAULT 0, dwell_s REAL NOT NULL DEFAULT 0,
      active_dwell_s REAL NOT NULL DEFAULT 0, engaged INTEGER NOT NULL DEFAULT 0)""")
    _add_column(conn, "observations", "page_id", "INTEGER REFERENCES pages(id) ON DELETE SET NULL")
    conn.execute("CREATE INDEX ix_obs_page ON observations(page_id)")
    conn.execute("CREATE INDEX ix_pages_kind ON pages(kind, active_dwell_s DESC)")
    conn.execute("CREATE INDEX ix_sightings_order ON sightings(started_at, id)")
    conn.execute("CREATE TABLE page_dirty_observations(obs_id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE page_dirty_totals(page_id INTEGER PRIMARY KEY)")
    conn.execute("""CREATE TRIGGER page_claim_forgotten AFTER UPDATE OF status ON claims
      WHEN old.status='forgotten' OR new.status='forgotten' BEGIN
        INSERT OR IGNORE INTO page_dirty_observations
          SELECT e.observation_id FROM evidence e JOIN claims c ON c.id=e.claim_id
          WHERE c.lineage_id=new.lineage_id;
      END""")
    # A new/removed sighting changes the preceding dwell and the following visit.
    for event, ref in (("INSERT", "new"), ("DELETE", "old"), ("UPDATE", "old"), ("UPDATE", "new")):
        timing = "AFTER" if ref == "new" else "BEFORE"
        conn.execute(f"""CREATE TRIGGER page_sighting_{event.lower()}_{ref} {timing} {event} ON sightings BEGIN
          INSERT OR IGNORE INTO page_dirty_observations VALUES ({ref}.obs_id);
          INSERT OR IGNORE INTO page_dirty_observations
            SELECT obs_id FROM sightings s JOIN observations o ON o.id=s.obs_id
            WHERE o.origin='user' AND (s.started_at,s.id)<({ref}.started_at,{ref}.id)
            ORDER BY s.started_at DESC,s.id DESC LIMIT 1;
          INSERT OR IGNORE INTO page_dirty_observations
            SELECT obs_id FROM sightings s JOIN observations o ON o.id=s.obs_id
            WHERE o.origin='user' AND (s.started_at,s.id)>({ref}.started_at,{ref}.id)
            ORDER BY s.started_at,s.id LIMIT 1;
        END""")
    conn.execute("""CREATE TRIGGER page_observation_insert AFTER INSERT ON observations BEGIN
      INSERT OR IGNORE INTO page_dirty_observations VALUES (new.id);
    END""")
    conn.execute("""CREATE TRIGGER page_observation_change AFTER UPDATE OF
      url,site,app,bundle_id,window_title,page_title,is_listing,origin ON observations BEGIN
      INSERT OR IGNORE INTO page_dirty_observations VALUES (new.id);
      INSERT OR IGNORE INTO page_dirty_totals SELECT old.page_id WHERE old.page_id IS NOT NULL;
      INSERT OR IGNORE INTO page_dirty_observations
        SELECT neighbor FROM (SELECT (SELECT s.obs_id FROM sightings s JOIN observations o ON o.id=s.obs_id
          WHERE o.origin='user' AND (s.started_at,s.id)<(v.started_at,v.id)
          ORDER BY s.started_at DESC,s.id DESC LIMIT 1) AS neighbor
          FROM sightings v WHERE v.obs_id=new.id) WHERE neighbor IS NOT NULL;
      INSERT OR IGNORE INTO page_dirty_observations
        SELECT neighbor FROM (SELECT (SELECT s.obs_id FROM sightings s JOIN observations o ON o.id=s.obs_id
          WHERE o.origin='user' AND (s.started_at,s.id)>(v.started_at,v.id)
          ORDER BY s.started_at,s.id LIMIT 1) AS neighbor
          FROM sightings v WHERE v.obs_id=new.id) WHERE neighbor IS NOT NULL;
    END""")
    conn.execute("""CREATE TRIGGER page_observation_delete BEFORE DELETE ON observations BEGIN
      INSERT OR IGNORE INTO page_dirty_totals SELECT old.page_id WHERE old.page_id IS NOT NULL;
    END""")


def m015_provenance_term(conn: sqlite3.Connection) -> None:
    """The visible term or value that links two attention moments."""
    _add_column(conn, "attention_provenance", "term", "TEXT")


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    m001_base,
    m002_draft_attempts_and_digest_counts,
    m003_structured_page_fields,
    m004_minhash_raw_bytes,
    m005_perf_log,
    m006_unsegmented_index,
    m007_memory_evolution,
    m008_sightings,
    m009_evidence_field,
    m010_episode_draft_chunks,
    m011_tasks,
    m012_episode_draft_state,
    m013_attention_provenance,
    m014_pages,
    m015_provenance_term,
]

SCHEMA_VERSION = len(MIGRATIONS)


def _statements(script: str) -> list[str]:
    """Split a script on complete statements (triggers contain inner semicolons)."""
    out, buf = [], ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    return out


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations; returns how many ran."""
    ran = 0
    # The version decision and all migrations share one write lock: another opener
    # cannot decide from a stale version while this connection upgrades the schema.
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ValueError(f"Database schema {version} is newer than supported schema {SCHEMA_VERSION}")
        for n, step in enumerate(MIGRATIONS[version:], start=version + 1):
            step(conn)
            conn.execute(f"PRAGMA user_version = {n}")
            conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(n),))
            log.info("applied migration %d (%s)", n, step.__name__)
            ran += 1
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return ran


def connect(path: str, *, read_only: bool = False) -> sqlite3.Connection:
    from ..encryption import open_database

    conn = open_database(path, read_only=read_only)
    try:
        if read_only:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError(f"Database schema {version} is newer than supported schema {SCHEMA_VERSION}")
            # A second connection for retrieval: WAL lets it read while the writer writes.
            conn.execute("PRAGMA query_only=ON")
            return conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        migrate(conn)
        return conn
    except BaseException:
        conn.close()
        raise


def clone_database(source, target) -> None:
    """Consistent benchmark copy, preserving encryption and its key-id sidecar."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    from ..encryption import atomic_write, open_database, raw_database, state_path

    target = Path(target).expanduser().resolve()
    if target.exists() or state_path(target).exists():
        raise FileExistsError(f"Clone destination already exists: {target}")
    src = open_database(str(source), read_only=True)
    try:
        state = getattr(src, "_discrete_encryption_state", None)
        dst = raw_database(target, state.key_id if state else None)
        try:
            src.backup(dst)
        finally:
            dst.close()
        if state:
            atomic_write(state_path(target), json.dumps(asdict(state)).encode())
        target.chmod(0o600)
    except BaseException:
        target.unlink(missing_ok=True)
        state_path(target).unlink(missing_ok=True)
        raise
    finally:
        src.close()
