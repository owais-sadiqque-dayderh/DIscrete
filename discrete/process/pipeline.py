"""Episode drafting and online merge (planS2 §5-6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, complete_json
from ..process.confidence import compute_confidence
from ..process.entities import extract_from_text, resolve_entity
from ..process.validation import atomic, injection_drop, modality_from_obs, subject_ok, verify_quote
from ..retrieval.search import fts_hits, fuse_rankings
from ..store import ClaimDraft, Store
from ..timeutil import day_key
from .prompts import DRAFT_EPISODE_PROMPT, RELATE_PROMPT

log = logging.getLogger("discrete.process.pipeline")


@dataclass
class BatchOutcome:
    episode_id: int
    claims_added: int = 0
    claims_reinforced: int = 0
    claims_invalidated: int = 0


class ProcessPipeline:
    def __init__(self, store: Store):
        self.store = store
        self.ep = insight_endpoint()

    async def draft_episode(self, episode_id: int) -> BatchOutcome | None:
        obs_rows = await self._observations_for_episode(episode_id)
        if not obs_rows:
            return None
        outcome = BatchOutcome(episode_id=episode_id)
        if not self.ep.configured:
            apps = sorted({r["app"] for r in obs_rows if r["app"]})
            sites = sorted({r["site"] for r in obs_rows if r["site"]})
            await self.store.close_episode(episode_id, title="Untitled", summary="", apps=apps, sites=sites)
            return outcome

        draft = await self._draft_episode(obs_rows)
        if not draft:
            return outcome
        apps = sorted({r["app"] for r in obs_rows if r["app"]})
        sites = sorted({r["site"] for r in obs_rows if r["site"]})
        await self.store.close_episode(
            episode_id,
            title=str(draft.get("title") or "")[:80],
            summary=str(draft.get("summary") or "")[:600],
            apps=apps,
            sites=sites,
        )
        t0 = min(r["started_at"] for r in obs_rows)
        t1 = max(r["ended_at"] for r in obs_rows)
        by_id = {r["id"]: r for r in obs_rows}

        for kind_key, claim_kind in (("events", "event"), ("facts", "fact"), ("preferences", "preference")):
            for item in (draft.get(kind_key) or [])[: T.DRAFT_MAX_EVENTS if claim_kind == "event" else T.DRAFT_MAX_DURABLE]:
                cd = self._validated_claim(item, claim_kind, by_id, episode_id, t0, t1)
                if not cd:
                    continue
                if claim_kind == "event":
                    cid = await self.store.create_claim(cd, confidence=compute_confidence([cd.modality], 1))
                    await self._link_evidence(cid, cd, by_id)
                    await self._attach_entities(cid, cd, obs_rows)
                    outcome.claims_added += 1
                else:
                    action = await self._merge_or_add(cd, by_id)
                    if action == "add":
                        outcome.claims_added += 1
                    elif action == "reinforce":
                        outcome.claims_reinforced += 1
                    elif action == "invalidate":
                        outcome.claims_invalidated += 1

        for obs in obs_rows:
            for name in extract_from_text(obs["screen_text"] + " " + obs["caption"]):
                await self.store.bump_entity_candidate(name, episode_id)
        return outcome

    async def _observations_for_episode(self, episode_id: int) -> list:
        def _get():
            return self.store.conn.execute(
                "SELECT * FROM observations WHERE episode_id=? ORDER BY started_at", (episode_id,)
            ).fetchall()

        return await self.store._call(_get)

    async def _draft_episode(self, obs_rows) -> dict | None:
        lines = []
        seen_excerpt: set[str] = set()
        for r in obs_rows[-T.DRAFT_MAX_OBS :]:
            excerpt = tk.truncate(r["screen_text"], T.DRAFT_TEXT_CHARS_PER_OBS)
            if excerpt in seen_excerpt:
                excerpt = ""
            else:
                seen_excerpt.add(excerpt)
            lines.append(
                f"#{r['id']} t={r['started_at']} app={r['app']} site={r['site']}\n"
                f"caption: {r['caption']}\n"
                f"interaction: keys={r['keys']} clicks={r['clicks']}\n"
                f"<<<DATA>>>\n{excerpt}\n<<<END>>>"
            )
        prompt = DRAFT_EPISODE_PROMPT.format(observations="\n---\n".join(lines))
        try:
            data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=2500)
            return data if isinstance(data, dict) else None
        except EXPECTED_ERRORS as exc:
            log.warning("episode draft failed: %s", exc)
            return None

    def _validated_claim(self, item, kind, by_id, episode_id, t0, t1) -> ClaimDraft | None:
        text = str(item.get("text") or "").strip()
        entities = [str(e) for e in (item.get("entities") or [])]
        if not atomic(text, entities) or not subject_ok(text):
            return None
        oid = int(item.get("quote_obs_id") or item.get("observation_id") or 0)
        quote = str(item.get("quote") or "")
        obs = by_id.get(oid)
        evidence = []
        modality = "passive"
        if obs:
            quote, ok = verify_quote(quote, obs["screen_text"], obs["caption"])
            if not ok:
                return None
            modality = modality_from_obs(obs)
            evidence = [{"observation_id": oid, "quote": quote}]
            ev_texts = [obs["screen_text"]]
            if injection_drop(text, ev_texts):
                return None
        return ClaimDraft(
            kind=kind,
            text=text,
            context=str(item.get("context") or ""),
            reasoning=str(item.get("reasoning") or ""),
            modality=modality,
            episode_id=episode_id,
            occurred_from=t0,
            occurred_to=t1,
            entities=entities,
            evidence=evidence,
        )

    async def _merge_or_add(self, draft: ClaimDraft, by_id) -> str:
        candidates = await self._candidates(draft)
        for row in candidates:
            if row["kind"] != draft.kind:
                continue
            rel = await self._relate(draft, row)
            if rel == "same":
                await self.store.reinforce_claim(row["id"], occurred_to=draft.occurred_to)
                await self._link_evidence(row["id"], draft, by_id)
                conf = compute_confidence([draft.modality], len({day_key(draft.occurred_from)}))
                await self.store.update_claim_confidence(row["id"], conf)
                return "reinforce"
            if rel == "contradicts":
                if draft.modality == "passive" and row["modality"] in ("engaged", "stated"):
                    cid = await self.store.create_claim(draft)
                    await self._link_evidence(cid, draft, by_id)
                    return "add"
                new_id = await self.store.create_claim(draft)
                await self._link_evidence(new_id, draft, by_id)
                await self.store.invalidate_claim(row["id"], new_id=new_id)
                return "invalidate"
            if rel == "refines":
                cid = await self.store.create_claim(draft, refines_id=row["id"])
                await self._link_evidence(cid, draft, by_id)
                return "add"
        cid = await self.store.create_claim(draft, confidence=compute_confidence([draft.modality], 1))
        await self._link_evidence(cid, draft, by_id)
        await self._attach_entities(cid, draft, list(by_id.values()))
        return "add"

    async def _candidates(self, draft: ClaimDraft):
        bm25 = await fts_hits(self.store, "claim", draft.text, limit=T.CANDIDATE_LIMIT)
        dense_ids: list[int] = []
        try:
            from ..llm import embed

            qv = (await embed([draft.text]))[0]
            for rid, vec in await self.store.all_embeddings("claim"):
                if tk.cosine(qv, vec) >= T.DENSE_HIT_THRESHOLD:
                    dense_ids.append(rid)
        except EXPECTED_ERRORS:
            pass
        fused = fuse_rankings([bm25, dense_ids], limit=T.CANDIDATE_LIMIT)
        live = await self.store.live_claims(kind=draft.kind)
        by_id = {r["id"]: r for r in live}
        return [by_id[i] for i in fused if i in by_id]

    async def _relate(self, draft: ClaimDraft, existing) -> str:
        prompt = RELATE_PROMPT.format(
            new_kind=draft.kind,
            new_text=draft.text,
            old_kind=existing["kind"],
            old_text=existing["text"],
        )
        try:
            data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=100)
            rel = str((data or {}).get("relation") or "unrelated")
            return rel if rel in ("same", "contradicts", "refines", "unrelated") else "unrelated"
        except EXPECTED_ERRORS:
            return "unrelated"

    async def _link_evidence(self, claim_id: int, draft: ClaimDraft, by_id) -> None:
        supports = 0
        for ev in draft.evidence:
            oid = int(ev.get("observation_id") or 0)
            quote = str(ev.get("quote") or "")
            obs = by_id.get(oid)
            if not obs:
                continue
            relation = "supports" if quote else "cooccurs"
            await self.store.link_evidence(claim_id, oid, relation, quote)
            if relation == "supports":
                supports += 1
                if supports >= T.EVIDENCE_MAX_SUPPORTS:
                    break

    async def _attach_entities(self, claim_id: int, draft: ClaimDraft, obs_rows) -> None:
        names = set(draft.entities)
        for obs in obs_rows:
            names.update(extract_from_text(obs["screen_text"]))
        for name in names:
            if not name.strip():
                continue
            eid = await resolve_entity(self.store, name)
            await self.store.record_mention(eid, "claim", claim_id)
