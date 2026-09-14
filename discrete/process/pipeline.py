"""Episode drafting and online merge (planS2 §5-6)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, NotConfigured, complete_json, embed
from ..store import ClaimDraft, Store
from ..timeutil import to_local
from .entities import attach_claim_mentions, note_candidates, structural_candidates
from .prompts import DRAFT_EPISODE_PROMPT, RELATE_PROMPT
from .validation import atomic, injection_drop, subject_ok, verify_evidence

log = logging.getLogger("discrete.process.pipeline")

_KINDS = ("event", "fact", "preference")


@dataclass
class DraftOutcome:
    episode_id: int
    added: int = 0
    reinforced: int = 0
    invalidated: int = 0
    rejected: dict = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


class ProcessPipeline:
    def __init__(self, store: Store):
        self.store = store
        self.ep = insight_endpoint()

    async def process_ready(self, *, limit: int = 5) -> list[DraftOutcome]:
        out = []
        for ep in await self.store.episodes_ready_for_draft(limit=limit):
            res = await self.draft_episode(ep["id"])
            if res:
                out.append(res)
        return out

    async def draft_episode(self, episode_id: int) -> DraftOutcome | None:
        if not await self.store.begin_draft(episode_id):
            return None  # someone else is drafting it, or it is not ready
        try:
            return await self._draft(episode_id)
        except EXPECTED_ERRORS as exc:
            log.warning("episode %s draft failed: %s", episode_id, exc)
            await self.store.draft_failed(episode_id)
            await self.store.bump_metric("draft_failed")
            return None
        except BaseException:
            await self.store.draft_failed(episode_id)
            raise

    async def _draft(self, episode_id: int) -> DraftOutcome:
        outcome = DraftOutcome(episode_id=episode_id)
        rows = await self.store.observations_for_episode(episode_id)
        apps = sorted({r["app"] for r in rows if r["app"]})
        sites = sorted({r["site"] for r in rows if r["site"]})
        engaged = sum(1 for r in rows if r["keys"] or r["clicks"] or r["clipboard"]) / max(1, len(rows))
        if not rows or not self.ep.configured:
            title = " / ".join(apps)[:80] or "(no user activity)"
            await self.store.close_episode(episode_id, title=title, summary="", apps=apps, sites=sites,
                                           engaged_fraction=engaged)
            return outcome

        data = await self._call_draft(rows)
        by_id = {r["id"]: r for r in rows}
        await self.store.close_episode(
            episode_id, title=tk.truncate(str(data.get("title") or ""), 80),
            summary=tk.truncate(str(data.get("summary") or ""), 600), apps=apps, sites=sites,
            engaged_fraction=engaged,
        )
        t0 = min(r["started_at"] for r in rows)
        t1 = max(r["ended_at"] for r in rows)
        counts = {"event": 0, "durable": 0}
        names_seen: list[str] = []
        for item in data.get("claims") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            if kind not in _KINDS:
                outcome.reject("bad_kind")
                continue
            bucket = "event" if kind == "event" else "durable"
            cap = T.DRAFT_MAX_EVENTS if bucket == "event" else T.DRAFT_MAX_DURABLE
            if counts[bucket] >= cap:
                outcome.reject("over_cap")
                continue
            cd = self._validate(item, kind, by_id, episode_id, t0, t1, outcome)
            if cd is None:
                continue
            counts[bucket] += 1
            names_seen += cd.entities
            if kind == "event":
                cid = await self.store.create_claim(cd)
                await self._mentions(cid, cd)
                outcome.added += 1
            else:
                op = await self._merge(cd)
                setattr(outcome, op, getattr(outcome, op) + 1)

        # Entity candidates (planS2 §10.2): VLM working entities, claim entities, the
        # structural pass on AX text, and ground-truth app/site where the user acted.
        for r in rows:
            names_seen += tk.parse_json_list(r["entities"])
            if r["text_source"] == "ax":
                names_seen += structural_candidates(r["screen_text"])
            if r["keys"] or r["clicks"] or r["clipboard"]:
                names_seen += [x for x in (r["app"], r["site"]) if x]
        await note_candidates(self.store, names_seen, episode_id)
        for reason, n in outcome.rejected.items():
            await self.store.bump_metric(f"claims_rejected_{reason}", n)
        return outcome

    async def _call_draft(self, rows) -> dict:
        seen_lines: set[str] = set()
        blocks = []
        for r in rows[-T.DRAFT_MAX_OBS:]:
            kept = []
            for line in (r["screen_text"] or "").splitlines():
                k = tk.norm_key(line)
                if k and k not in seen_lines:
                    seen_lines.add(k)
                    kept.append(line)
            excerpt = tk.truncate("\n".join(kept), T.DRAFT_TEXT_CHARS_PER_OBS)
            acted = "acted" if (r["keys"] or r["clicks"] or r["clipboard"]) else "viewed only"
            blocks.append(
                f"[observation_id={r['id']}] {to_local(r['started_at']):%H:%M} app={r['app']} site={r['site']} "
                f"window={r['window_title']!r} ({acted}: keys={r['keys']} clicks={r['clicks']} "
                f"scrolls={r['scrolls']} copies={r['clipboard']})\n"
                f"caption: {r['caption']}\n<<<DATA>>>\n{excerpt}\n<<<END>>>"
            )
        prompt = DRAFT_EPISODE_PROMPT.format(
            observations="\n\n".join(blocks), max_events=T.DRAFT_MAX_EVENTS, max_durable=T.DRAFT_MAX_DURABLE
        )
        data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=2500)
        if not isinstance(data, dict):
            from ..llm import LLMError

            raise LLMError("draft: not an object")
        return data

    def _validate(self, item: dict, kind: str, by_id: dict, episode_id: int, t0: float, t1: float,
                  outcome: DraftOutcome) -> ClaimDraft | None:
        text = str(item.get("text") or "").strip()
        entities = [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
        if not atomic(text, entities):
            outcome.reject("not_atomic")
            return None
        if not subject_ok(text):
            outcome.reject("third_party_subject")
            return None
        evidence, cited = [], []
        for ev in item.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            try:
                oid = int(ev.get("observation_id"))
            except (TypeError, ValueError):
                continue
            obs = by_id.get(oid)
            if obs is None:
                continue
            quote, ok = verify_evidence(text, str(ev.get("quote") or ""), obs)
            if ok:
                evidence.append({"observation_id": oid, "quote": quote})
                cited.append(obs)
        if not evidence:
            outcome.reject("unsupported")
            return None
        if injection_drop(text, [o["screen_text"] for o in cited]):
            outcome.reject("injection")
            return None
        engaged = any(o["keys"] or o["clicks"] or o["clipboard"] for o in cited)
        times = [o["started_at"] for o in cited]
        return ClaimDraft(
            kind=kind, text=text, context=str(item.get("context") or "")[:200],
            modality="engaged" if engaged else "passive", episode_id=episode_id,
            occurred_from=t0 if kind == "event" else min(times),
            occurred_to=t1 if kind == "event" else max(o["ended_at"] for o in cited),
            entities=entities, evidence=evidence,
        )

    async def _mentions(self, claim_id: int, cd: ClaimDraft) -> None:
        await attach_claim_mentions(self.store, claim_id, cd.text, [e["quote"] for e in cd.evidence], cd.entities)

    # ----------------------------------------------------------- online merge

    async def _candidates(self, cd: ClaimDraft) -> list:
        from ..retrieval.search import fts_ids

        bm25 = await fts_ids(self.store, "claim", cd.text, limit=T.CANDIDATE_LIMIT * 3)
        rankings = [bm25]
        try:
            qv = (await embed([cd.text]))[0]
            rankings.append([i for i, _ in await self.store.dense_search("claim", qv, limit=T.CANDIDATE_LIMIT * 3)])
        except (NotConfigured, *EXPECTED_ERRORS):
            pass
        fused = sorted(tk.rrf(rankings, k=T.RRF_K).items(), key=lambda x: -x[1])
        rows = {r["id"]: r for r in await self.store.get_claims([i for i, _ in fused])}
        out = [rows[i] for i, _ in fused if i in rows and rows[i]["status"] == "live" and rows[i]["kind"] == cd.kind]
        return out[: T.RELATE_CANDIDATES]

    async def _relate(self, cd: ClaimDraft, candidates: list) -> dict[int, str]:
        listing = "\n".join(f"[id={c['id']}] {c['text']}" for c in candidates)
        prompt = RELATE_PROMPT.format(kind=cd.kind, text=cd.text, candidates=listing)
        try:
            data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=300)
        except EXPECTED_ERRORS:
            return {}
        rels: dict[int, str] = {}
        for r in (data or {}).get("relations") or []:
            try:
                rid, rel = int(r.get("id")), str(r.get("relation") or "")
            except (TypeError, ValueError, AttributeError):
                continue
            if rel in ("same", "refines", "contradicts", "unrelated"):
                rels[rid] = rel
        return rels

    async def _merge(self, cd: ClaimDraft) -> str:
        """planS2 §6.2: ADD / REINFORCE / INVALIDATE. Never rewrites claim text."""
        candidates = await self._candidates(cd)
        rels = await self._relate(cd, candidates) if candidates else {}
        by_rel = {rel: [c for c in candidates if rels.get(c["id"]) == rel] for rel in ("same", "contradicts", "refines")}
        if by_rel["same"]:
            target = by_rel["same"][0]
            await self.store.reinforce_claim(target["id"], cd)
            return "reinforced"
        if by_rel["contradicts"]:
            old = by_rel["contradicts"][0]
            if cd.modality == "passive" and old["modality"] in ("engaged", "stated"):
                cid = await self.store.create_claim(cd)
                await self.store.flag_claims([cid, old["id"]], "contested")
                await self._mentions(cid, cd)
                return "added"
            cid = await self.store.create_claim(cd, invalidates_id=old["id"])
            await self.store.invalidate_claim(old["id"], by_id=cid, at=cd.occurred_from)
            await self._mentions(cid, cd)
            return "invalidated"
        refines = by_rel["refines"][0]["id"] if by_rel["refines"] else None
        cid = await self.store.create_claim(cd, refines_id=refines)
        await self._mentions(cid, cd)
        return "added"


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
