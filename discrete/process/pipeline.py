"""Episode drafting and online merge (planS2 §5-6)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, LLMError, NotConfigured, complete_json, embed
from ..models import ClaimDraft
from ..perf import PERF
from ..store import Store
from ..timeutil import to_local
from .embed_worker import claim_text
from .entities import attach_claim_mentions, note_candidates, structural_candidates
from .prompts import DRAFT_EPISODE_PROMPT, RELATE_BLOCK, RELATE_PROMPT
from .validation import atomic, injection_drop, subject_ok, verify_evidence

log = logging.getLogger("discrete.process.pipeline")

_KINDS = ("event", "fact", "preference")
_RELATIONS = ("same", "refines", "contradicts", "unrelated")


@dataclass
class DraftOutcome:
    episode_id: int
    added: int = 0
    reinforced: int = 0
    invalidated: int = 0
    rejected: dict = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


@dataclass
class _Durable:
    """A validated fact/preference on its way through the merge."""

    cd: ClaimDraft
    text: str  # embedding text (claim text + context)
    vec: list[float] | None = None
    candidates: list = field(default_factory=list)
    relations: dict[int, str] = field(default_factory=dict)


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
            with PERF.timer("draft.episode", ref_id=episode_id) as t:
                res = await self._draft(episode_id)
                t.n = res.added + res.reinforced
            return res
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
        durables: list[_Durable] = []
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
                durables.append(_Durable(cd=cd, text=claim_text(cd.text, cd.context)))
        for op in await self._merge(durables):
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
            page = f" page={r['page_title']!r}" if r["page_title"] else ""
            blocks.append(
                f"[observation_id={r['id']}] {to_local(r['started_at']):%H:%M} app={r['app']} site={r['site']} "
                f"window={r['window_title']!r}{page} ({acted}: keys={r['keys']} clicks={r['clicks']} "
                f"scrolls={r['scrolls']} copies={r['clipboard']})\n"
                f"caption: {r['caption']}\n<<<DATA>>>\n{excerpt}\n<<<END>>>"
            )
        prompt = DRAFT_EPISODE_PROMPT.format(
            observations="\n\n".join(blocks), max_events=T.DRAFT_MAX_EVENTS, max_durable=T.DRAFT_MAX_DURABLE
        )
        data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=2500)
        if not isinstance(data, dict):
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
                oid = int(ev.get("observation_id"))  # type: ignore[arg-type]  # None → TypeError
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

    async def _merge(self, durables: list[_Durable]) -> list[str]:
        """planS2 §6.2: ADD / REINFORCE / INVALIDATE for an episode's durable claims.
        One embeddings call and one relation call per episode, however many claims;
        each new claim's vector is stored with it, so it is never embedded again."""
        if not durables:
            return []
        try:
            vecs = await embed([d.text for d in durables])
        except (NotConfigured, *EXPECTED_ERRORS):
            vecs = [None] * len(durables)
        for d, v in zip(durables, vecs, strict=True):
            d.vec = v
            d.candidates = await self._candidates(d)
        await self._relate(durables)
        return [await self._apply(d) for d in durables]

    async def _candidates(self, d: _Durable) -> list:
        rankings = [await self.store.fts_ids("claim", d.cd.text, limit=T.CANDIDATE_LIMIT * 3)]
        if d.vec is not None:
            rankings.append([i for i, _ in await self.store.dense_search("claim", d.vec, limit=T.CANDIDATE_LIMIT * 3)])
        fused = sorted(tk.rrf(rankings, k=T.RRF_K).items(), key=lambda x: -x[1])
        rows = {r["id"]: r for r in await self.store.get_claims([i for i, _ in fused])}
        out = [rows[i] for i, _ in fused if i in rows and rows[i]["status"] == "live" and rows[i]["kind"] == d.cd.kind]
        return out[: T.RELATE_CANDIDATES]

    async def _relate(self, durables: list[_Durable]) -> None:
        asked = [(n, d) for n, d in enumerate(durables) if d.candidates]
        if not asked:
            return
        blocks = "\n\n".join(
            RELATE_BLOCK.format(n=n, kind=d.cd.kind, text=d.cd.text,
                                candidates="\n".join(f"[id={c['id']}] {c['text']}" for c in d.candidates))
            for n, d in asked
        )
        try:
            data = await complete_json(self.ep, [{"role": "user", "content": RELATE_PROMPT.format(blocks=blocks)}],
                                       max_tokens=150 + 60 * sum(len(d.candidates) for _, d in asked))
        except EXPECTED_ERRORS:
            return
        for r in (data or {}).get("relations") or []:
            try:
                n, rid, rel = int(r.get("new")), int(r.get("id")), str(r.get("relation") or "")
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= n < len(durables) and rel in _RELATIONS and any(c["id"] == rid for c in durables[n].candidates):
                durables[n].relations[rid] = rel

    async def _apply(self, d: _Durable) -> str:
        """Never rewrites claim text."""
        cd = d.cd
        vector = (d.vec, tk.content_hash(d.text)) if d.vec is not None else None
        by_rel = {rel: [c for c in d.candidates if d.relations.get(c["id"]) == rel]
                  for rel in ("same", "contradicts", "refines")}
        if by_rel["same"]:
            await self.store.reinforce_claim(by_rel["same"][0]["id"], cd)
            return "reinforced"
        if by_rel["contradicts"]:
            old = by_rel["contradicts"][0]
            if cd.modality == "passive" and old["modality"] in ("engaged", "stated"):
                cid = await self.store.create_claim(cd, vector=vector)
                await self.store.flag_claims([cid, old["id"]], "contested")
                await self._mentions(cid, cd)
                return "added"
            cid = await self.store.create_claim(cd, invalidates_id=old["id"], vector=vector)
            await self.store.invalidate_claim(old["id"], by_id=cid, at=cd.occurred_from)
            await self._mentions(cid, cd)
            return "invalidated"
        refines = by_rel["refines"][0]["id"] if by_rel["refines"] else None
        cid = await self.store.create_claim(cd, refines_id=refines, vector=vector)
        await self._mentions(cid, cd)
        return "added"
