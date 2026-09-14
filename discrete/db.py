"""SQLite schema. Timestamps are epoch floats (see timeutil)."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

_FTS_TOKENIZE = "tokenize='porter unicode61 remove_diacritics 2'"

SCHEMA = f"""
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


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    return conn
