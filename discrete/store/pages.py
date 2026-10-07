"""Pages from sanitized observations; dwell and returns from individual sightings."""

from __future__ import annotations

import sqlite3
from urllib.parse import urlsplit

from ..capture import privacy
from ..capture.redact import redact, redact_url
from ..config import T
from ..page_identity import canonical_url, classify_kind, document_identity
from .base import StoreBase


def _identity(row: sqlite3.Row) -> tuple[str, str, str, str] | None:
    if row["origin"] != "user" or privacy.denied(row["app"], row["bundle_id"], row["url"]) \
            or privacy.denied(url="https://" + row["site"]):
        return None
    title = (row["page_title"] or row["window_title"]).strip()
    url = row["url"]
    if T.REDACT_SECRETS:
        title, url = redact(title).text, redact_url(url).text
    identity = canonical_url(url)
    if not identity and not url:
        # AXDocument is already stored in url by window_context. Without it,
        # only the captured window title determines a document's identity;
        # a vision-generated page_title can label it, never invent its path.
        window_title = redact(row["window_title"]).text if T.REDACT_SECRETS else row["window_title"]
        identity = document_identity(row["app"], window_title)
    if not identity:
        return None
    parts = urlsplit(identity)
    site = row["app"] if parts.scheme in ("file", "doc", "file-title") else (parts.hostname or row["app"])
    # Redaction can change host/path; apply policy to the normalized identity too.
    if privacy.denied(url=identity):
        return None
    return identity, site, title or identity, classify_kind(identity, site=site, is_listing=bool(row["is_listing"]))


def _metrics(conn: sqlite3.Connection, page_id: int, start: float, end: float) -> sqlite3.Row:
    # Indexed neighbor lookups avoid scanning the entire store on each capture.
    # The next sighting is global, even outside the requested interval: its real
    # boundary must be respected before clipping dwell to the interval end.
    return conn.execute("""WITH s AS (
      SELECT s.*, o.page_id,
        (SELECT n.started_at FROM sightings n JOIN observations no ON no.id=n.obs_id
          WHERE no.origin='user' AND (n.started_at,n.id)>(s.started_at,s.id)
          ORDER BY n.started_at,n.id LIMIT 1) AS next_at,
        (SELECT p.id FROM sightings p JOIN observations po ON po.id=p.obs_id
          WHERE po.origin='user' AND (p.started_at,p.id)<(s.started_at,s.id)
          ORDER BY p.started_at DESC,p.id DESC LIMIT 1) AS previous_id
      FROM sightings s JOIN observations o ON o.id=s.obs_id
      WHERE o.origin='user' AND o.page_id=? AND s.started_at < ?),
      d AS (SELECT s.*, MAX(0, MIN(COALESCE(next_at-started_at, ?), ?, ?-started_at)
        - MAX(0, ?-started_at)) AS dwell,
        (keys+clicks+scrolls+clipboard)>0 AS active,
        CASE WHEN previous_id IS NULL OR
          (SELECT o.page_id FROM sightings p JOIN observations o ON o.id=p.obs_id WHERE p.id=previous_id)
            IS NOT page_id OR
          started_at-(SELECT started_at FROM sightings WHERE id=previous_id)>=? THEN 1 ELSE 0 END AS visit
        FROM s)
      SELECT MIN(started_at) AS first_seen, MAX(ended_at) AS last_seen,
        COALESCE(SUM(CASE WHEN started_at>=? THEN visit ELSE 0 END),0) AS visits,
        COALESCE(SUM(dwell),0) AS dwell_s,
        COALESCE(SUM(CASE WHEN active THEN dwell ELSE 0 END),0) AS active_dwell_s,
        COALESCE(MAX(CASE WHEN started_at>=? OR dwell>0 THEN active ELSE 0 END),0) AS engaged,
        COALESCE(SUM(CASE WHEN started_at>=? AND keys>0 THEN 1 ELSE 0 END),0) AS typing_sightings
      FROM d WHERE started_at>=? OR dwell>0""",
      (page_id, end, T.DWELL_LAST_S, T.DWELL_GAP_CAP_S, end, start, T.PAGE_VISIT_GAP_S,
       start, start, start, start)).fetchone()


def sync_pages(conn: sqlite3.Connection) -> None:
    """Drain transactional invalidations; called before each store transaction commits."""
    # Apply changed privacy/attention settings before any materialized data is
    # committed or read. Newly allowed legacy rows still need explicit backfill.
    policy = repr((sorted(privacy._APPS), sorted(privacy._SITES), T.REDACT_SECRETS, T.PAGES,
                   T.PAGE_VISIT_GAP_S, T.DWELL_GAP_CAP_S, T.DWELL_LAST_S))
    old_policy = conn.execute("SELECT value FROM meta WHERE key='pages_policy'").fetchone()
    if not old_policy or old_policy[0] != policy:
        conn.execute("INSERT OR IGNORE INTO page_dirty_observations "
                     "SELECT id FROM observations WHERE page_id IS NOT NULL")
        conn.execute("INSERT INTO meta(key,value) VALUES ('pages_policy',?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (policy,))
    dirty = conn.execute("SELECT obs_id FROM page_dirty_observations").fetchall()
    for (oid,) in dirty:
        row = conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
        if row is None:
            continue
        old = row["page_id"]
        excluded = conn.execute("SELECT 1 FROM evidence e JOIN claims c ON c.id=e.claim_id "
                                "JOIN claims f ON f.lineage_id=c.lineage_id "
                                "WHERE e.observation_id=? AND f.status='forgotten' LIMIT 1", (oid,)).fetchone()
        identity = _identity(row) if T.PAGES and not excluded else None
        pid = None
        if identity:
            conn.execute("INSERT INTO pages(canonical_url,site,title,kind) VALUES (?,?,?,?) "
                         "ON CONFLICT(canonical_url) DO UPDATE SET site=excluded.site,title=excluded.title,kind=excluded.kind",
                         identity)
            pid = conn.execute("SELECT id FROM pages WHERE canonical_url=?", (identity[0],)).fetchone()[0]
        conn.execute("UPDATE observations SET page_id=? WHERE id=?", (pid, oid))
        if old != pid:
            conn.execute("""INSERT OR IGNORE INTO page_dirty_totals
              SELECT o.page_id FROM observations o WHERE o.page_id IS NOT NULL AND o.id IN (
                SELECT (SELECT n.obs_id FROM sightings n JOIN observations no ON no.id=n.obs_id
                  WHERE no.origin='user' AND (n.started_at,n.id)>(s.started_at,s.id)
                  ORDER BY n.started_at,n.id LIMIT 1) FROM sightings s WHERE s.obs_id=?)""", (oid,))
        for p in (old, pid):
            if p is not None:
                conn.execute("INSERT OR IGNORE INTO page_dirty_totals VALUES (?)", (p,))
    conn.execute("DELETE FROM page_dirty_observations")
    for (pid,) in conn.execute("SELECT page_id FROM page_dirty_totals").fetchall():
        m = _metrics(conn, pid, 0, float("inf"))
        if m["first_seen"] is None:
            conn.execute("DELETE FROM pages WHERE id=?", (pid,))
            continue
        latest = conn.execute("SELECT o.* FROM observations o JOIN sightings s ON s.obs_id=o.id "
                              "WHERE o.page_id=? ORDER BY s.started_at DESC,s.id DESC LIMIT 1", (pid,)).fetchone()
        identity = _identity(latest)
        if identity:
            conn.execute("UPDATE pages SET site=?,title=?,kind=? WHERE id=?", (*identity[1:], pid))
        conn.execute("UPDATE pages SET first_seen=?,last_seen=?,visits=?,dwell_s=?,active_dwell_s=?,engaged=? WHERE id=?",
                     (m["first_seen"], m["last_seen"], m["visits"], m["dwell_s"], m["active_dwell_s"], m["engaged"], pid))
    conn.execute("DELETE FROM page_dirty_totals")


class PagesMixin(StoreBase):
    async def backfill_pages(self) -> int:
        """Rebuild links/totals with today's privacy and canonicalization rules, idempotently."""
        def run():
            with self._tx() as conn:
                conn.execute("INSERT OR IGNORE INTO page_dirty_observations SELECT id FROM observations")
                sync_pages(conn)
                return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        return await self._call(run)

    async def list_pages(self, *, start: float = 0, end: float = float("inf"), kind: str = "",
                         limit: int = 20, order: str = "dwell", search: str = "",
                         edited: bool = False, kinds: tuple[str, ...] = ()) -> list[dict]:
        """Windowed metrics use sightings, never widened observation intervals.

        Typing is an edit signal, not proof that a document was saved or changed.
        """
        if not T.PAGES:
            return []
        def run():
            with self._tx() as conn:
                sync_pages(conn)
                rows = conn.execute("SELECT * FROM pages WHERE last_seen>=? AND first_seen<? "
                                    "AND (?='' OR kind=?)", (start - T.DWELL_GAP_CAP_S, end, kind, kind)).fetchall()
                found = []
                for row in rows:
                    if kinds and row["kind"] not in kinds:
                        continue
                    if search and search.casefold() not in (row["title"] + " " + row["canonical_url"]).casefold():
                        continue
                    m = dict(_metrics(conn, row["id"], start, end))
                    if m["first_seen"] is None or (edited and not m["typing_sightings"]):
                        continue
                    found.append({**dict(row), **m})
                metric = "visits" if order == "visits" else "dwell_s"
                return sorted(found, key=lambda p: (-p[metric], -p["active_dwell_s"], p["id"]))[:max(0, limit)]
        return await self._call(run)
