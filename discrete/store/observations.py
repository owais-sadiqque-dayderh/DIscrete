"""Observations: dedup on insert, deferred enrichment, retention, time-window reads."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

import numpy as np

from .. import textkit as tk
from ..config import T
from ..models import ObservationDraft, world_text
from ..timeutil import now
from .base import StoreBase, rowid

_INSERT_COLS = (
    "source", "origin", "event_kind", "session_id", "started_at", "ended_at", "app", "bundle_id",
    "window_title", "url", "site", "screen_text", "text_source", "caption", "entities", "keys", "clicks",
    "scrolls", "clipboard", "frame_phash", "frame_path", "page_title", "is_listing", "navigated_from",
    "enrich_state", "content_hash", "text_hash", "created_at", "updated_at",
)
_INSERT_SQL = (f"INSERT INTO observations({', '.join(_INSERT_COLS)}) "
               f"VALUES ({', '.join('?' * len(_INSERT_COLS))})")


def _ph(ids: Sequence) -> str:
    return ",".join("?" * len(ids))


class ObservationsMixin(StoreBase):
    async def pending_enrichment(self, *, limit: int = 16) -> list[sqlite3.Row]:
        return await self.q("SELECT * FROM observations WHERE origin='user' AND enrich_state='pending' "
                            "ORDER BY id LIMIT ?", (limit,))

    async def unsegmented_observations(self, *, limit: int = 1) -> list[sqlite3.Row]:
        """User rows not yet in an episode, oldest first: the runtime's segmentation queue."""
        return await self.q("SELECT * FROM observations WHERE origin='user' AND episode_id IS NULL "
                            "ORDER BY id LIMIT ?", (limit,))

    async def add_observation(self, draft: ObservationDraft) -> tuple[int | None, bool]:
        """Insert, or collapse into an existing row. Returns (id, is_new)."""

        def _sighting(conn, oid: int) -> None:
            if draft.origin == "user":
                ts = draft.started_at or now()
                conn.execute("INSERT INTO sightings(obs_id, started_at, ended_at, keys, clicks, scrolls, clipboard) "
                             "VALUES (?,?,?,?,?,?,?)", (oid, ts, max(draft.ended_at or ts, ts), draft.keys,
                                                        draft.clicks, draft.scrolls, draft.clipboard))

        def _bump(conn, oid: int) -> None:
            _sighting(conn, oid)
            conn.execute(
                "UPDATE observations SET dup_count=dup_count+1, ended_at=MAX(ended_at, ?), updated_at=?, "
                "keys=keys+?, clicks=clicks+?, scrolls=scrolls+?, clipboard=clipboard+? WHERE id=?",
                (draft.ended_at or now(), now(), draft.keys, draft.clicks, draft.scrolls, draft.clipboard, oid),
            )
            self._metric(conn, "observations_collapsed")

        def _add():
            with self._tx() as conn:
                if draft.frame_duplicate:
                    row = conn.execute(
                        "SELECT id FROM observations WHERE app=? AND window_title=? AND url=? AND started_at>=? "
                        "ORDER BY id DESC LIMIT 1",
                        (draft.app, draft.window_title, draft.url, (draft.started_at or now()) - 3600),
                    ).fetchone()
                    if row:
                        _bump(conn, row[0])
                        return row[0], False
                    return None, False
                key = draft.dedup_key()
                if not draft.screen_text.strip() and not draft.caption.strip():
                    return None, False
                text_hash = tk.content_hash(key)
                site = draft.site or tk.site_from_url(draft.url)
                # Tier 1: exact on the stable key (planS2 §2.1).
                row = conn.execute(
                    "SELECT id FROM observations WHERE text_hash=? ORDER BY id DESC LIMIT 1", (text_hash,)
                ).fetchone()
                if row:
                    _bump(conn, row[0])
                    return row[0], False
                # Tier 2: near-duplicate within the same app and site (planS2 §2.2).
                sig = tk.minhash(key, permutations=T.DEDUP_MINHASH_PERMUTATIONS)
                if len(tk.word_shingles(key)) >= T.DEDUP_SHORT_SHINGLES:
                    recent = conn.execute(
                        "SELECT o.id, m.sig FROM observations o JOIN obs_minhash m ON m.obs_id=o.id "
                        "WHERE o.app=? AND o.site=? ORDER BY o.id DESC LIMIT 200",
                        (draft.app, site),
                    ).fetchall()
                    if (oid := _near_duplicate(sig, recent)) is not None:
                        _bump(conn, oid)
                        return oid, False
                ts = draft.started_at or now()
                cur = conn.execute(_INSERT_SQL, (
                    draft.source, draft.origin, draft.event_kind, draft.session_id, ts, draft.ended_at or ts,
                    draft.app, draft.bundle_id, draft.window_title, draft.url, site, draft.screen_text,
                    draft.text_source, draft.caption, json.dumps(draft.entities), draft.keys, draft.clicks,
                    draft.scrolls, draft.clipboard, draft.frame_phash, draft.frame_path, draft.page_title,
                    int(draft.is_listing), draft.navigated_from, draft.enrich_state,
                    tk.content_hash(draft.world_text()), text_hash, now(), now(),
                ))
                oid = rowid(cur)
                _sighting(conn, oid)
                conn.execute("INSERT INTO obs_minhash(obs_id, sig) VALUES (?, ?)", (oid, tk.minhash_bytes(sig)))
                if draft.origin == "user":
                    self._bump_df(conn, tk.content_tokens(draft.world_text()))
                    self._enqueue(conn, "observation", oid)
                self._metric(conn, "observations_ingested")
                return oid, True

        return await self._call(_add)

    async def get_observation(self, oid: int) -> sqlite3.Row | None:
        return await self.q1("SELECT * FROM observations WHERE id=?", (oid,))

    async def get_observations(self, ids: Sequence[int], *, user_only: bool = False) -> list[sqlite3.Row]:
        if not ids:
            return []
        extra = " AND origin='user'" if user_only else ""
        rows = await self.q(f"SELECT * FROM observations WHERE id IN ({_ph(ids)}){extra}", list(ids))
        order = {i: n for n, i in enumerate(ids)}
        return sorted(rows, key=lambda r: order.get(r["id"], 1 << 30))

    async def observations_for_episode(self, episode_id: int, *, user_only: bool = True) -> list[sqlite3.Row]:
        extra = " AND origin='user'" if user_only else ""
        return await self.q(
            f"SELECT * FROM observations WHERE episode_id=?{extra} ORDER BY started_at, id", (episode_id,)
        )

    async def observations_in_interval(self, start: float, end: float, *, limit: int = 200) -> list[sqlite3.Row]:
        return await self.q(
            "SELECT * FROM observations WHERE origin='user' AND started_at < ? AND ended_at >= ? "
            "ORDER BY started_at LIMIT ?",
            (end, start, limit),
        )

    async def observation_ids_started_between(self, start: float, end: float) -> list[int]:
        rows = await self.q("SELECT id FROM observations WHERE origin='user' AND started_at >= ? AND started_at < ?",
                            (start, end))
        return [r[0] for r in rows]

    async def observation_ids_navigated_from(self, phrase: str, *, limit: int = 400) -> list[int]:
        """Rows the vision call saw being reached from a page named like `phrase`."""
        phrase = phrase.strip()
        if not phrase:
            return []
        rows = await self.q("SELECT id FROM observations WHERE origin='user' AND navigated_from != '' "
                            "AND navigated_from LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
                            ("%" + phrase.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", limit))
        return [r[0] for r in rows]

    async def observations_started_within(self, start: float, end: float) -> list[sqlite3.Row]:
        """Inclusive window, time-ordered (the enumeration trail)."""
        return await self.q(
            "SELECT id, app, site, url, window_title, page_title, is_listing, navigated_from, caption, "
            "screen_text, started_at, ended_at FROM observations "
            "WHERE origin='user' AND started_at >= ? AND started_at <= ? ORDER BY started_at", (start, end))

    async def page_identity_rows(self, ids: Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        return await self.q(
            f"SELECT id, app, window_title, page_title, url, site, navigated_from, is_listing FROM observations "
            f"WHERE id IN ({_ph(ids)})", list(ids))

    async def assign_episode(self, obs_ids: Sequence[int], episode_id: int) -> None:
        if obs_ids:
            await self.execute(
                f"UPDATE observations SET episode_id=? WHERE id IN ({_ph(obs_ids)})",
                [episode_id, *obs_ids],
            )

    async def recent_texts(self, app: str, site: str, *, before: float, limit: int) -> list[str]:
        """Screen text of the latest captures of one (app, site): the baseline for chrome."""
        rows = await self.q("SELECT screen_text FROM observations WHERE origin='user' AND app=? AND site=? "
                            "AND started_at <= ? AND screen_text != '' ORDER BY started_at DESC LIMIT ?",
                            (app, site, before, limit))
        return [r[0] for r in rows]

    async def observation_count(self) -> int:
        return await self.scalar("SELECT COUNT(*) FROM observations WHERE origin='user'")

    async def df_lookup(self, toks: Sequence[str]) -> tuple[dict[str, int], int]:
        def _df():
            uniq = list({t for t in toks if t})
            n = self.rconn.execute("SELECT COUNT(*) FROM observations WHERE origin='user'").fetchone()[0]
            if not uniq:
                return {}, n
            rows = self.rconn.execute(
                f"SELECT token, df FROM token_df WHERE token IN ({_ph(uniq)})", uniq
            ).fetchall()
            return {r[0]: r[1] for r in rows}, n

        return await self._read(_df)

    # ------------------------------------------------------------- enrichment
    # A row is stored at capture time with a basic caption; the vision caption and
    # page fields arrive later (capture/enrich.py), so a slow model never blocks capture.

    async def enrich_observation(self, oid: int, *, caption: str, entities: list[str], page_title: str,
                                 is_listing: bool, navigated_from: str) -> bool:
        def _run():
            with self._tx() as conn:
                row = conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
                if row is None or row["origin"] != "user":
                    return False
                before = world_text(row["window_title"], row["site"], row["caption"], row["screen_text"],
                                    row["page_title"])
                after = world_text(row["window_title"], row["site"], caption or row["caption"],
                                   row["screen_text"], page_title)
                old_toks, new_toks = set(tk.content_tokens(before)), set(tk.content_tokens(after))
                self._bump_df(conn, old_toks - new_toks, -1)
                self._bump_df(conn, new_toks - old_toks, +1)
                conn.execute(
                    "UPDATE observations SET caption=?, entities=?, page_title=?, is_listing=?, navigated_from=?, "
                    "enrich_state='done', content_hash=?, updated_at=? WHERE id=?",
                    (caption or row["caption"], json.dumps(entities), page_title, int(is_listing), navigated_from,
                     tk.content_hash(after), now(), oid),
                )
                self._enqueue(conn, "observation", oid)
                return True

        return await self._call(_run)

    async def enrichment_failed(self, oid: int) -> None:
        """The basic caption stays; the row is marked degraded (planS1 §6.2)."""
        await self.execute(
            "UPDATE observations SET enrich_state='failed', source='degraded', updated_at=? WHERE id=?", (now(), oid))

    async def discard_observation(self, oid: int) -> None:
        """The vision call judged the frame useless (lock screen, empty desktop). The row
        is kept so an identical frame still dedups against it, but it leaves every
        user-facing index."""

        def _run():
            with self._tx() as conn:
                row = conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()
                if row is None or row["origin"] != "user":
                    return
                self._bump_df(conn, tk.content_tokens(world_text(
                    row["window_title"], row["site"], row["caption"], row["screen_text"], row["page_title"])), -1)
                conn.execute("UPDATE observations SET origin='discarded', enrich_state='done', episode_id=NULL, "
                             "updated_at=? WHERE id=?", (now(), oid))
                self._drop_vector(conn, "observation", oid)
                self._metric(conn, "observations_discarded")

        await self._call(_run)

    async def observations_missing_page_fields(self, limit: int) -> list[sqlite3.Row]:
        """User rows captured before the vision call returned page fields."""
        return await self.q(
            "SELECT * FROM observations WHERE origin='user' AND page_title='' AND navigated_from='' "
            "AND enrich_state IN ('', 'failed') AND caption != '' ORDER BY id LIMIT ?", (limit,))

    async def fail_stale_enrichment(self) -> int:
        """Rows left pending by a process that died keep their basic caption."""

        def _run():
            return self.conn.execute(
                "UPDATE observations SET enrich_state='failed', source='degraded' WHERE enrich_state='pending'"
            ).rowcount

        return await self._call(_run)

    # ------------------------------------------------------------- retention

    async def clear_old_screen_text(self, *, ttl_days: float) -> int:
        """planS2 §12: screen_text is cleared (row, caption, evidence kept); DF decremented."""
        if ttl_days <= 0:
            return 0
        cutoff = now() - ttl_days * 86400

        def _p():
            with self._tx() as conn:
                rows = conn.execute(
                    "SELECT id, screen_text, origin, window_title, site, caption, page_title FROM observations "
                    "WHERE started_at < ? AND screen_text != ''",
                    (cutoff,),
                ).fetchall()
                for r in rows:
                    if r["origin"] == "user":
                        before = set(tk.content_tokens(r["screen_text"]))
                        after = set(tk.content_tokens(" ".join(
                            (r["window_title"], r["site"], r["caption"], r["page_title"]))))
                        self._bump_df(conn, before - after, -1)
                    conn.execute("UPDATE observations SET screen_text='', frame_path='' WHERE id=?", (r["id"],))
                    if r["origin"] == "user":
                        self._enqueue(conn, "observation", r["id"])
                return len(rows)

        return await self._call(_p)


def _near_duplicate(sig: np.ndarray, recent: list) -> int | None:
    """Most recent row whose signature agrees with `sig` on ≥ the Jaccard threshold."""
    ids, sigs = [], []
    for oid, raw in recent:
        s = tk.minhash_from_bytes(raw)
        if len(s) == len(sig):  # a changed permutation count makes old signatures incomparable
            ids.append(oid)
            sigs.append(s)
    if not sigs:
        return None
    sims = (np.vstack(sigs) == sig).mean(axis=1)
    hits = np.flatnonzero(sims >= T.DEDUP_JACCARD_THRESHOLD)
    return int(ids[hits[0]]) if len(hits) else None
