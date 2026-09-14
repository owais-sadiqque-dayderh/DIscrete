"""Storage layer — all SQLite access goes through here."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from uuid import uuid4

from . import textkit as tk
from .config import T, db_path, embedding_endpoint
from .db import connect
from .timeutil import now

log = logging.getLogger("discrete.store")


@dataclass
class ObservationDraft:
    source: str = "screen"
    origin: str = "user"
    event_kind: str = "mouse"
    session_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    site: str = ""
    screen_text: str = ""
    text_source: str = ""
    caption: str = ""
    entities: list[str] = field(default_factory=list)
    keys: int = 0
    clicks: int = 0
    scrolls: int = 0
    clipboard: int = 0
    frame_phash: str = ""
    frame_path: str = ""

    def dedup_key(self) -> str:
        return tk.stable_obs_key(self.app, self.window_title, self.url, self.screen_text)

    def searchable(self) -> str:
        return "\n".join(p for p in (self.app, self.window_title, self.site, self.caption, self.screen_text) if p)


@dataclass
class ClaimDraft:
    kind: str
    text: str
    context: str = ""
    reasoning: str = ""
    modality: str = "engaged"
    episode_id: int | None = None
    occurred_from: float = 0.0
    occurred_to: float = 0.0
    entities: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = asyncio.Lock()
        self._mh_cache: OrderedDict[int, Any] = OrderedDict()

    @classmethod
    async def open(cls, path: str | None = None) -> Store:
        p = str(path or db_path())
        return cls(connect(p))

    async def close(self) -> None:
        async with self._lock:
            self.conn.close()

    async def _call(self, fn, *args, **kwargs):
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    def _bump_df(self, conn: sqlite3.Connection, tokens: Iterable[str], delta: int = 1) -> None:
        uniq = [t for t in {t for t in tokens if len(t) <= 64}]
        for t in uniq:
            conn.execute(
                "INSERT INTO token_df(token, df) VALUES (?, ?) "
                "ON CONFLICT(token) DO UPDATE SET df = MAX(0, token_df.df + ?)",
                (t, delta, delta),
            )

    def _enqueue_embed(self, conn: sqlite3.Connection, ref_kind: str, ref_id: int) -> None:
        conn.execute(
            "INSERT INTO index_outbox(ref_kind, ref_id, enqueued_at, attempts) VALUES (?, ?, ?, 0) "
            "ON CONFLICT(ref_kind, ref_id) DO UPDATE SET enqueued_at=excluded.enqueued_at",
            (ref_kind, ref_id, now()),
        )

    def _metric(self, conn: sqlite3.Connection, name: str, delta: float = 1.0) -> None:
        conn.execute(
            "INSERT INTO metrics(name, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = metrics.value + ?, updated_at = ?",
            (name, delta, now(), delta, now()),
        )

    # ------------------------------------------------------------------ meta

    async def get_meta(self, key: str, default: str = "") -> str:
        def _get():
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

        return await self._call(_get)

    async def set_meta(self, key: str, value: str) -> None:
        def _set():
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

        await self._call(_set)

    async def stats(self) -> dict[str, Any]:
        def _stats():
            c = self.conn
            obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            claims = c.execute("SELECT COUNT(*) FROM claims WHERE status='live'").fetchone()[0]
            ents = c.execute("SELECT COUNT(*) FROM entities WHERE merged_into IS NULL").fetchone()[0]
            eps = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            gaps = c.execute("SELECT COUNT(*) FROM gaps WHERE status='open'").fetchone()[0]
            embeds = c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            outbox = c.execute("SELECT COUNT(*) FROM index_outbox").fetchone()[0]
            rng = c.execute("SELECT MIN(started_at), MAX(started_at) FROM observations").fetchone()
            dupes = c.execute("SELECT COALESCE(SUM(dup_count-1),0) FROM observations").fetchone()[0]
            return {
                "observations": obs,
                "duplicates_collapsed": dupes,
                "claims_live": claims,
                "entities": ents,
                "episodes": eps,
                "open_gaps": gaps,
                "embeddings": embeds,
                "outbox_pending": outbox,
                "first_observation": rng[0],
                "last_observation": rng[1],
            }

        return await self._call(_stats)

    async def df_lookup(self, toks: Sequence[str]) -> tuple[dict[str, int], int]:
        def _df():
            uniq = list({t for t in toks if t})
            if not uniq:
                return {}, 0
            placeholders = ",".join("?" * len(uniq))
            rows = self.conn.execute(
                f"SELECT token, df FROM token_df WHERE token IN ({placeholders})", uniq
            ).fetchall()
            n = self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            return {r[0]: r[1] for r in rows}, n

        return await self._call(_df)

    # ----------------------------------------------------------- observations

    async def add_observation(self, draft: ObservationDraft) -> tuple[int | None, bool]:
        def _add():
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                key = draft.dedup_key()
                if not key.strip():
                    conn.execute("ROLLBACK")
                    return None, False
                text_hash = tk.content_hash(key)
                chash = tk.content_hash(draft.searchable())
                row = conn.execute(
                    "SELECT id FROM observations WHERE text_hash=? ORDER BY id DESC LIMIT 1", (text_hash,)
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE observations SET dup_count=dup_count+1, updated_at=? WHERE id=?",
                        (now(), row[0]),
                    )
                    conn.execute("COMMIT")
                    return row[0], False

                recent = conn.execute(
                    "SELECT id, text_hash FROM observations ORDER BY id DESC LIMIT 50"
                ).fetchall()
                mh_new = tk.minhash(key, permutations=T.DEDUP_MINHASH_PERMUTATIONS)
                for oid, th in recent:
                    sig = conn.execute("SELECT sig FROM obs_minhash WHERE obs_id=?", (oid,)).fetchone()
                    if not sig:
                        continue
                    mh_old = pickle.loads(sig[0])
                    if tk.minhash_jaccard(mh_new, mh_old) >= T.DEDUP_JACCARD_THRESHOLD:
                        conn.execute(
                            "UPDATE observations SET dup_count=dup_count+1, updated_at=? WHERE id=?",
                            (now(), oid),
                        )
                        conn.execute("COMMIT")
                        return oid, False

                ts = draft.started_at or now()
                cur = conn.execute(
                    """INSERT INTO observations(
                      source, origin, event_kind, session_id, started_at, ended_at,
                      app, bundle_id, window_title, url, site, screen_text, text_source,
                      caption, entities, keys, clicks, scrolls, clipboard,
                      frame_phash, frame_path, content_hash, text_hash, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        draft.source,
                        draft.origin,
                        draft.event_kind,
                        draft.session_id,
                        ts,
                        draft.ended_at or ts,
                        draft.app,
                        draft.bundle_id,
                        draft.window_title,
                        draft.url,
                        draft.site or tk.site_from_url(draft.url),
                        draft.screen_text,
                        draft.text_source,
                        draft.caption,
                        json.dumps(draft.entities),
                        draft.keys,
                        draft.clicks,
                        draft.scrolls,
                        draft.clipboard,
                        draft.frame_phash,
                        draft.frame_path,
                        chash,
                        text_hash,
                        ts,
                        ts,
                    ),
                )
                oid = cur.lastrowid
                conn.execute(
                    "INSERT INTO obs_minhash(obs_id, sig) VALUES (?, ?)",
                    (oid, pickle.dumps(mh_new)),
                )
                self._bump_df(conn, tk.content_tokens(draft.searchable()))
                self._enqueue_embed(conn, "observation", oid)
                self._metric(conn, "observations_ingested")
                conn.execute("COMMIT")
                return oid, True
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return await self._call(_add)

    async def get_observation(self, oid: int) -> sqlite3.Row | None:
        def _get():
            return self.conn.execute("SELECT * FROM observations WHERE id=?", (oid,)).fetchone()

        return await self._call(_get)

    async def get_observations(self, ids: Sequence[int]) -> list[sqlite3.Row]:
        if not ids:
            return []

        def _get():
            placeholders = ",".join("?" * len(ids))
            rows = self.conn.execute(
                f"SELECT * FROM observations WHERE id IN ({placeholders})", list(ids)
            ).fetchall()
            order = {i: n for n, i in enumerate(ids)}
            return sorted(rows, key=lambda r: order.get(r["id"], 1 << 30))

        return await self._call(_get)

    async def unprocessed_observation_ids(self, *, limit: int = 500) -> list[int]:
        def _get():
            rows = self.conn.execute(
                "SELECT id FROM observations WHERE episode_id IS NULL ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
            return [r[0] for r in rows]

        return await self._call(_get)

    async def assign_episode(self, obs_ids: Sequence[int], episode_id: int) -> None:
        if not obs_ids:
            return

        def _set():
            placeholders = ",".join("?" * len(obs_ids))
            self.conn.execute(
                f"UPDATE observations SET episode_id=? WHERE id IN ({placeholders})",
                [episode_id, *obs_ids],
            )

        await self._call(_set)

    # --------------------------------------------------------------- episodes

    async def open_episode(self, session_id: str, started_at: float) -> int:
        def _open():
            cur = self.conn.execute(
                """INSERT INTO episodes(session_id, started_at, ended_at, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'open', ?, ?)""",
                (session_id, started_at, started_at, started_at, started_at),
            )
            return cur.lastrowid

        return await self._call(_open)

    async def close_episode(self, episode_id: int, *, title: str, summary: str, apps: list[str], sites: list[str]) -> None:
        def _close():
            ts = now()
            self.conn.execute(
                """UPDATE episodes SET ended_at=?, title=?, summary=?, apps=?, sites=?,
                   status='closed', updated_at=? WHERE id=?""",
                (ts, title, summary, json.dumps(apps), json.dumps(sites), ts, episode_id),
            )
            self._enqueue_embed(self.conn, "episode", episode_id)

        await self._call(_close)

    async def get_open_episode(self) -> sqlite3.Row | None:
        def _get():
            return self.conn.execute(
                "SELECT * FROM episodes WHERE status='open' ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return await self._call(_get)

    async def mark_episode_ready(self, episode_id: int) -> None:
        def _m():
            ts = now()
            self.conn.execute(
                "UPDATE episodes SET status='ready', ended_at=?, updated_at=? WHERE id=?",
                (ts, ts, episode_id),
            )

        await self._call(_m)

    async def episodes_ready_for_draft(self, *, limit: int = 5) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                "SELECT * FROM episodes WHERE status='ready' ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()

        return await self._call(_get)

    async def episodes_for_day(self, day: str) -> list[sqlite3.Row]:
        from .timeutil import day_start, from_local
        from datetime import datetime, timedelta

        start = day_start(from_local(datetime.strptime(day, "%Y-%m-%d")))
        end = start + 86400

        def _get():
            return self.conn.execute(
                "SELECT * FROM episodes WHERE started_at >= ? AND started_at < ? ORDER BY started_at",
                (start, end),
            ).fetchall()

        return await self._call(_get)

    async def episodes_in_interval(self, start: float, end: float) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                "SELECT * FROM episodes WHERE started_at < ? AND ended_at >= ? ORDER BY started_at",
                (end, start),
            ).fetchall()

        return await self._call(_get)

    async def observation_count(self) -> int:
        def _get():
            return self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

        return await self._call(_get)

    # ------------------------------------------------------------------ claims

    async def create_claim(
        self,
        draft: ClaimDraft,
        *,
        lineage_id: str | None = None,
        version: int = 1,
        refines_id: int | None = None,
        confidence: float | None = None,
    ) -> int:
        def _create():
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                ts = now()
                lid = lineage_id or str(uuid4())
                conf = confidence if confidence is not None else min(10.0, max(0.0, len(draft.evidence) * 0.5))
                cur = conn.execute(
                    """INSERT INTO claims(
                      lineage_id, version, kind, text, context, reasoning, modality,
                      occurred_from, occurred_to, valid_from, status, episode_id,
                      refines_id, confidence, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?, 'live', ?, ?, ?, ?, ?)""",
                    (
                        lid,
                        version,
                        draft.kind,
                        tk.truncate(draft.text, T.CLAIM_MAX_CHARS),
                        draft.context,
                        draft.reasoning,
                        draft.modality,
                        draft.occurred_from,
                        draft.occurred_to,
                        ts,
                        draft.episode_id,
                        refines_id,
                        conf,
                        ts,
                        ts,
                    ),
                )
                cid = cur.lastrowid
                self._bump_df(conn, tk.content_tokens(draft.text))
                self._enqueue_embed(conn, "claim", cid)
                conn.execute("COMMIT")
                return cid
            except Exception:
                conn.execute("ROLLBACK")
                raise

        return await self._call(_create)

    async def invalidate_claim(self, claim_id: int, *, new_id: int | None = None) -> None:
        def _inv():
            self.conn.execute(
                "UPDATE claims SET status='invalidated', valid_to=?, invalidates_id=?, updated_at=? WHERE id=?",
                (now(), new_id, now(), claim_id),
            )

        await self._call(_inv)

    async def live_claims(self, *, kind: str | None = None, limit: int | None = None) -> list[sqlite3.Row]:
        def _get():
            sql = "SELECT * FROM claims WHERE status='live'"
            args: list[Any] = []
            if kind:
                sql += " AND kind=?"
                args.append(kind)
            sql += " ORDER BY updated_at DESC"
            if limit:
                sql += " LIMIT ?"
                args.append(limit)
            return self.conn.execute(sql, args).fetchall()

        return await self._call(_get)

    async def link_evidence(
        self, claim_id: int, observation_id: int, relation: str, quote: str = ""
    ) -> None:
        def _link():
            self.conn.execute(
                """INSERT INTO evidence(claim_id, observation_id, relation, quote, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(claim_id, observation_id) DO UPDATE SET
                     relation=CASE WHEN excluded.relation='supports' THEN 'supports' ELSE evidence.relation END,
                     quote=CASE WHEN excluded.relation='supports' THEN excluded.quote ELSE evidence.quote END""",
                (claim_id, observation_id, relation, quote, now()),
            )

        await self._call(_link)

    async def evidence_for_claim(self, claim_id: int) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                "SELECT * FROM evidence WHERE claim_id=? ORDER BY created_at", (claim_id,)
            ).fetchall()

        return await self._call(_get)

    async def reinforce_claim(self, claim_id: int, *, occurred_to: float | None = None) -> None:
        def _r():
            if occurred_to is not None:
                self.conn.execute(
                    "UPDATE claims SET updated_at=?, occurred_to=? WHERE id=?",
                    (now(), occurred_to, claim_id),
                )
            else:
                self.conn.execute("UPDATE claims SET updated_at=? WHERE id=?", (now(), claim_id))

        await self._call(_r)

    async def update_claim_confidence(self, claim_id: int, confidence: float) -> None:
        def _u():
            self.conn.execute("UPDATE claims SET confidence=?, updated_at=? WHERE id=?", (confidence, now(), claim_id))

        await self._call(_u)

    async def supersede_claims(self, old_ids: list[int], draft: ClaimDraft, *, derived_from: list[int]) -> int:
        def _sup():
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            ts = now()
            olds = conn.execute(
                f"SELECT * FROM claims WHERE id IN ({','.join('?'*len(old_ids))})", old_ids
            ).fetchall()
            lineage = olds[0]["lineage_id"] if olds else str(uuid4())
            version = max(r["version"] for r in olds) + 1 if olds else 1
            cur = conn.execute(
                """INSERT INTO claims(
                  lineage_id, version, kind, text, context, reasoning, modality,
                  occurred_from, occurred_to, valid_from, status, episode_id, confidence,
                  derived_from, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,'live',?,?,?,?,?,?)""",
                (
                    lineage,
                    version,
                    draft.kind,
                    draft.text,
                    draft.context,
                    draft.reasoning,
                    draft.modality,
                    draft.occurred_from,
                    draft.occurred_to,
                    ts,
                    draft.episode_id,
                    draft.evidence and 0.5 or 0.3,
                    json.dumps(derived_from),
                    ts,
                    ts,
                ),
            )
            new_id = cur.lastrowid
            for old in olds:
                conn.execute(
                    "UPDATE claims SET status='superseded', valid_to=?, superseded_by=?, updated_at=? WHERE id=?",
                    (ts, new_id, ts, old["id"]),
                )
                for link in conn.execute("SELECT * FROM evidence WHERE claim_id=?", (old["id"],)).fetchall():
                    conn.execute(
                        """INSERT INTO evidence(claim_id, observation_id, relation, quote, created_at)
                           VALUES (?,?,?,?,?)
                           ON CONFLICT(claim_id, observation_id) DO NOTHING""",
                        (new_id, link["observation_id"], link["relation"], link["quote"], ts),
                    )
            self._enqueue_embed(conn, "claim", new_id)
            conn.execute("COMMIT")
            return new_id

        return await self._call(_sup)

    async def similar_claim_clusters(self, sim_threshold: float, max_size: int) -> list[list[sqlite3.Row]]:
        rows = await self.live_claims()
        clusters: list[list[sqlite3.Row]] = []
        used: set[int] = set()
        vecs = {r["id"]: await self.get_embedding("claim", r["id"]) for r in rows}
        for r in rows:
            if r["id"] in used or r["kind"] == "event":
                continue
            cluster = [r]
            used.add(r["id"])
            va = vecs.get(r["id"])
            if not va:
                continue
            for r2 in rows:
                if r2["id"] in used or r2["kind"] != r["kind"]:
                    continue
                vb = vecs.get(r2["id"])
                if vb and tk.cosine(va, vb) >= sim_threshold:
                    cluster.append(r2)
                    used.add(r2["id"])
                if len(cluster) >= max_size:
                    break
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters

    async def apply_consolidation(self, cluster: list[sqlite3.Row], outputs: list[dict]) -> None:
        old_ids = [r["id"] for r in cluster]
        for out in outputs[: len(cluster) + 2]:
            draft = ClaimDraft(
                kind=str(out.get("kind") or cluster[0]["kind"]),
                text=str(out.get("text") or "")[: T.CLAIM_MAX_CHARS],
                episode_id=cluster[0]["episode_id"],
            )
            inputs = [int(x) for x in (out.get("input_ids") or old_ids)]
            await self.supersede_claims(old_ids, draft, derived_from=inputs)
            break

    async def supersede_unsupported(self, ttl_days: float) -> int:
        cutoff = now() - ttl_days * 86400

        def _run():
            rows = self.conn.execute(
                """SELECT c.id FROM claims c
                   LEFT JOIN evidence e ON e.claim_id=c.id AND e.relation='supports'
                   WHERE c.status='live' AND c.created_at < ?
                   GROUP BY c.id HAVING COUNT(e.claim_id)=0""",
                (cutoff,),
            ).fetchall()
            ts = now()
            for r in rows:
                self.conn.execute(
                    "UPDATE claims SET status='superseded', valid_to=?, flag='unsupported', updated_at=? WHERE id=?",
                    (ts, ts, r[0]),
                )
            return len(rows)

        return await self._call(_run)

    # ---------------------------------------------------------------- entities

    async def upsert_entity(self, name: str, kind: str = "unknown") -> int:
        def _upsert():
            conn = self.conn
            nk = tk.norm_key(name)
            row = conn.execute("SELECT entity_id FROM entity_aliases WHERE alias_norm=?", (nk,)).fetchone()
            if row:
                return row[0]
            ts = now()
            cur = conn.execute(
                "INSERT INTO entities(name, kind, created_at, updated_at) VALUES (?,?,?,?)",
                (name.strip(), kind, ts, ts),
            )
            eid = cur.lastrowid
            conn.execute(
                "INSERT INTO entity_aliases(alias_norm, entity_id, alias, norm_key) VALUES (?,?,?,?)",
                (nk, eid, name.strip(), nk),
            )
            conn.execute(
                "INSERT INTO entities_fts(rowid, name, aliases, summary) VALUES (?,?,?,?)",
                (eid, name.strip(), name.strip(), ""),
            )
            self._enqueue_embed(conn, "entity", eid)
            return eid

        return await self._call(_upsert)

    async def record_mention(self, entity_id: int, ref_kind: str, ref_id: int) -> None:
        def _rec():
            try:
                self.conn.execute(
                    "INSERT INTO mentions(entity_id, ref_kind, ref_id, created_at) VALUES (?,?,?,?)",
                    (entity_id, ref_kind, ref_id, now()),
                )
            except sqlite3.IntegrityError:
                pass

        await self._call(_rec)

    async def entity_candidates(self) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute("SELECT * FROM entity_candidates").fetchall()

        return await self._call(_get)

    async def bump_entity_candidate(self, name: str, episode_id: int) -> None:
        def _bump():
            nk = tk.norm_key(name)
            ts = now()
            row = self.conn.execute(
                "SELECT episodes FROM entity_candidates WHERE name_norm=?", (nk,)
            ).fetchone()
            if row:
                eps = set(tk.parse_json_list(row[0]))
                eps.add(episode_id)
                self.conn.execute(
                    "UPDATE entity_candidates SET episodes=?, last_seen=? WHERE name_norm=?",
                    (json.dumps(sorted(eps)), ts, nk),
                )
            else:
                self.conn.execute(
                    "INSERT INTO entity_candidates(name_norm, name, episodes, first_seen, last_seen) VALUES (?,?,?,?,?)",
                    (nk, name.strip(), json.dumps([episode_id]), ts, ts),
                )

        await self._call(_bump)

    async def find_entity_by_alias(self, alias_norm: str) -> int | None:
        def _get():
            row = self.conn.execute(
                "SELECT entity_id FROM entity_aliases WHERE alias_norm=?", (alias_norm,)
            ).fetchone()
            return row[0] if row else None

        return await self._call(_get)

    async def all_entity_aliases(self) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute("SELECT alias_norm, entity_id, alias FROM entity_aliases").fetchall()

        return await self._call(_get)

    async def get_embedding_for_name(self, name: str) -> list[float] | None:
        from .llm import EXPECTED_ERRORS, embed

        try:
            return (await embed([name]))[0]
        except EXPECTED_ERRORS:
            return None

    async def nearest_entity(self, vector: list[float]) -> tuple[int | None, float]:
        best_id, best_sim = None, -1.0
        for rid, vec in await self.all_embeddings("entity"):
            sim = tk.cosine(vector, vec)
            if sim > best_sim:
                best_id, best_sim = rid, sim
        return best_id, best_sim

    async def update_entity_page(self, entity_id: int, summary: str) -> None:
        def _u():
            ts = now()
            self.conn.execute(
                "UPDATE entities SET summary=?, page_updated_at=?, updated_at=? WHERE id=?",
                (summary, ts, ts, entity_id),
            )
            row = self.conn.execute("SELECT name FROM entities WHERE id=?", (entity_id,)).fetchone()
            if row:
                self.conn.execute("DELETE FROM entities_fts WHERE rowid=?", (entity_id,))
                self.conn.execute(
                    "INSERT INTO entities_fts(rowid, name, aliases, summary) VALUES (?,?,?,?)",
                    (entity_id, row[0], row[0], summary),
                )
            self._enqueue_embed(self.conn, "entity", entity_id)

        await self._call(_u)

    async def entities_needing_pages(self, limit: int) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                """SELECT * FROM entities WHERE merged_into IS NULL
                   AND (summary='' OR page_updated_at < updated_at)
                   ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return await self._call(_get)

    async def claims_for_entity(self, entity_id: int) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                """SELECT c.* FROM claims c
                   JOIN mentions m ON m.ref_kind='claim' AND m.ref_id=c.id
                   WHERE m.entity_id=? AND c.status='live' ORDER BY c.updated_at DESC LIMIT 30""",
                (entity_id,),
            ).fetchall()

        return await self._call(_get)

    async def aggregate_entities(self, kind_filter: str, start: float | None, end: float | None) -> list[dict]:
        def _get():
            sql = """SELECT e.id, e.name, e.kind, e.summary, COUNT(DISTINCT m.ref_id) AS mentions
                     FROM entities e JOIN mentions m ON m.entity_id=e.id
                     WHERE e.merged_into IS NULL"""
            args: list[Any] = []
            if kind_filter and kind_filter != "unknown":
                sql += " AND e.kind=?"
                args.append(kind_filter)
            sql += " GROUP BY e.id ORDER BY mentions DESC LIMIT 50"
            rows = self.conn.execute(sql, args).fetchall()
            return [dict(r) for r in rows]

        return await self._call(_get)

    async def reconcile_index_coverage(self) -> int:
        def _run():
            n = 0
            specs = [
                ("observation", "observations", ""),
                ("claim", "claims", " AND status='live'"),
                ("episode", "episodes", ""),
                ("entity", "entities", " AND merged_into IS NULL"),
            ]
            for kind, table, extra in specs:
                rows = self.conn.execute(
                    f"""SELECT id FROM {table} t
                        WHERE NOT EXISTS (
                          SELECT 1 FROM embeddings e WHERE e.ref_kind=? AND e.ref_id=t.id
                        ){extra}""",
                    (kind,),
                ).fetchall()
                for r in rows:
                    self._enqueue_embed(self.conn, kind, r[0])
                    n += 1
            return n

        return await self._call(_run)

    async def save_digest(self, day: str, text: str) -> None:
        def _s():
            self.conn.execute(
                "INSERT OR REPLACE INTO digests(day, text, created_at) VALUES (?,?,?)",
                (day, text, now()),
            )

        await self._call(_s)

    async def get_digest(self, day: str) -> str | None:
        def _g():
            row = self.conn.execute("SELECT text FROM digests WHERE day=?", (day,)).fetchone()
            return row[0] if row else None

        return await self._call(_g)

    async def purge_old_observations(self, *, ttl_days: float) -> int:
        if ttl_days <= 0:
            return 0

        def _p():
            cutoff = now() - ttl_days * 86400
            rows = self.conn.execute(
                "SELECT id FROM observations WHERE started_at < ? AND episode_id IS NOT NULL", (cutoff,)
            ).fetchall()
            ids = [r[0] for r in rows]
            if not ids:
                return 0
            ph = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM observations WHERE id IN ({ph})", ids)
            return len(ids)

        return await self._call(_p)

    # -------------------------------------------------------------- embeddings

    async def outbox_batch(self, *, limit: int = 64) -> list[sqlite3.Row]:
        def _get():
            return self.conn.execute(
                "SELECT * FROM index_outbox ORDER BY enqueued_at ASC LIMIT ?", (limit,)
            ).fetchall()

        return await self._call(_get)

    async def save_embedding(self, ref_kind: str, ref_id: int, vector: list[float], text_hash: str) -> None:
        def _save():
            conn = self.conn
            model = embedding_endpoint().model or "unknown"
            conn.execute(
                """INSERT INTO embeddings(ref_kind, ref_id, model, dim, vector, text_hash, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(ref_kind, ref_id) DO UPDATE SET
                     model=excluded.model, dim=excluded.dim, vector=excluded.vector,
                     text_hash=excluded.text_hash, created_at=excluded.created_at""",
                (ref_kind, ref_id, model, len(vector), np_bytes(vector), text_hash, now()),
            )
            conn.execute("DELETE FROM index_outbox WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id))

        await self._call(_save)

    async def get_embedding(self, ref_kind: str, ref_id: int) -> list[float] | None:
        def _get():
            row = self.conn.execute(
                "SELECT vector FROM embeddings WHERE ref_kind=? AND ref_id=?", (ref_kind, ref_id)
            ).fetchone()
            return np_from_bytes(row[0]) if row else None

        return await self._call(_get)

    async def all_embeddings(self, ref_kind: str) -> list[tuple[int, list[float]]]:
        def _all():
            rows = self.conn.execute(
                "SELECT ref_id, vector FROM embeddings WHERE ref_kind=?", (ref_kind,)
            ).fetchall()
            return [(r[0], np_from_bytes(r[1])) for r in rows]

        return await self._call(_all)

    # ------------------------------------------------------------------- gaps

    async def record_gap(self, question: str) -> int:
        def _rec():
            cur = self.conn.execute(
                "INSERT INTO gaps(question, status, created_at) VALUES (?, 'open', ?)",
                (question, now()),
            )
            return cur.lastrowid

        return await self._call(_rec)

    async def record_retrieval_miss(self, query: str, missed: str, stage: str) -> None:
        def _rec():
            self.conn.execute(
                "INSERT INTO retrieval_misses(query, missed, stage, created_at) VALUES (?,?,?,?)",
                (query, missed, stage, now()),
            )

        await self._call(_rec)

    async def answer_gap(self, gap_id: int, answer: str) -> sqlite3.Row | None:
        def _ans():
            row = self.conn.execute("SELECT * FROM gaps WHERE id=? AND status='open'", (gap_id,)).fetchone()
            if not row:
                return None
            ts = now()
            self.conn.execute(
                "UPDATE gaps SET status='answered', answer=?, answered_at=? WHERE id=?",
                (answer.strip(), ts, gap_id),
            )
            return self.conn.execute("SELECT * FROM gaps WHERE id=?", (gap_id,)).fetchone()

        return await self._call(_ans)

    async def open_gaps(self) -> list[sqlite3.Row]:
        def _g():
            return self.conn.execute("SELECT * FROM gaps WHERE status='open' ORDER BY id").fetchall()

        return await self._call(_g)


def np_bytes(vec: list[float]) -> bytes:
    import numpy as np

    return np.array(vec, dtype=np.float32).tobytes()


def np_from_bytes(blob: bytes) -> list[float]:
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32).tolist()
