"""Episode drafting and online merge (planS2 §5-6)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field

from .. import textkit as tk
from ..config import T, insight_endpoint
from ..llm import EXPECTED_ERRORS, LLMError, NotConfigured, complete_json, embed
from ..models import ClaimDraft, ClaimWrite
from ..perf import PERF
from ..store import DraftConflict, Store
from ..timeutil import to_local
from .actions import detect, hints
from .embed_worker import claim_text
from .entities import promotable, resolve_existing, structural_candidates
from .prompts import DRAFT_EPISODE_PROMPT, RELATE_BLOCK, RELATE_PROMPT
from .validation import atomic, injection_drop, subject_ok, verify_evidence
from .verification import verify_claims

log = logging.getLogger("discrete.process.pipeline")

_KINDS = ("event", "fact", "preference")
_RELATIONS = ("same", "refines", "contradicts", "unrelated")

# Prompts cover the model input; these fields cover subsequent validation,
# verification, modality/times, and capture-derived entity candidates. Capture
# bookkeeping (dup_count, updated_at, etc.) must not discard successful calls.
_DRAFT_ROW_FIELDS = (
    "id", "started_at", "ended_at", "app", "site", "window_title", "screen_text",
    "caption", "keys", "clicks", "clipboard", "entities", "text_source",
)


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


ACTION_VERBS = frozenset(
    "downloaded installed uninstalled purchased bought ordered booked paid sent posted replied published created "
    "saved exported uploaded signed_up registered submitted subscribed unsubscribed deleted shared "
    "scheduled accepted declined merged committed pushed deployed chose".split())


def action_verb(item: dict) -> str:
    """The drafted action verb, normalised, when it names a completed action we track."""
    verb = str(item.get("action") or "").strip().lower().replace(" ", "_")
    return verb if verb in ACTION_VERBS else ""


def _thing_tokens(text: str) -> set[str]:
    """What an action is about: its words minus "the user", the verb and filler."""
    return {t for t in tk.content_tokens(text, min_len=3) if t not in ACTION_VERBS and t not in ("user", "users")}


def max_events(span_min: float) -> int:
    """Event room for an episode of this length (long episodes do several things)."""
    extra = int(max(0.0, span_min - 5) // max(T.DRAFT_EVENT_MINUTES, 0.5))
    return max(T.DRAFT_MAX_EVENTS, min(T.DRAFT_MAX_EVENTS_CAP, T.DRAFT_MAX_EVENTS + extra))


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
        except (*EXPECTED_ERRORS, DraftConflict) as exc:
            log.warning("episode %s draft failed: %s", episode_id, exc)
            await self.store.draft_failed(episode_id)
            await self.store.bump_metric("draft_failed")
            await self._count_abandoned(episode_id)
            return None
        except BaseException as exc:
            await self.store.draft_failed(episode_id)
            await self.store.bump_metric("draft_cancelled" if isinstance(exc, asyncio.CancelledError) else "draft_failed")
            await self._count_abandoned(episode_id)
            raise
        # Outside the draft lifecycle: even cancellation here must not reopen a
        # committed episode. The maintenance pass resumes unfinished graph work.
        from .provenance import build_provenance

        try:
            await build_provenance(self.store)
        except Exception:
            log.exception("attention provenance failed after episode %s", episode_id)
            await self.store.bump_metric("provenance_failed")
        return res

    async def _count_abandoned(self, episode_id: int) -> None:
        episode = await self.store.get_episode(episode_id)
        if episode is not None and episode["status"] == "failed":
            await self.store.bump_metric("draft_abandoned")

    async def _draft(self, episode_id: int) -> DraftOutcome:
        outcome = DraftOutcome(episode_id=episode_id)
        rows = await self.store.observations_for_episode(episode_id)
        apps = sorted({r["app"] for r in rows if r["app"]})
        sites = sorted({r["site"] for r in rows if r["site"]})
        engaged = sum(1 for r in rows if r["keys"] or r["clicks"] or r["clipboard"]) / max(1, len(rows))
        if not rows or not self.ep.configured:
            if rows and not self.ep.configured:
                await self.store.bump_metric("draft_skipped_unconfigured", len(rows))
            title = " / ".join(apps)[:80] or "(no user activity)"
            await self.store.close_episode(episode_id, title=title, summary="", apps=apps, sites=sites,
                                           engaged_fraction=engaged)
            return outcome

        # Every observation is read: an episode longer than one call (recovered, or from
        # before EPISODE_MAX_OBS) is drafted in consecutive chunks, never truncated.
        chunks = [rows[i:i + T.DRAFT_MAX_OBS] for i in range(0, len(rows), T.DRAFT_MAX_OBS)]
        reached = {r["id"] for chunk in chunks for r in chunk}
        if dropped := {r["id"] for r in rows} - reached:
            await self.store.bump_metric("draft_window_dropped", len(dropped))
            raise DraftConflict("draft window omitted observations")
        chrome = await self._chrome(rows)
        input_hash = self._draft_input_hash(rows, [self._draft_prompt(chunk, chrome) for chunk in chunks])
        cached = await self.store.draft_chunks(episode_id, input_hash)
        parts = []
        for index, chunk in enumerate(chunks):
            if index in cached:
                part = cached[index]
            else:
                part = await self._call_draft(chunk, chrome)
                await self.store.save_draft_chunk(episode_id, index, input_hash, part)
            await self.store.record_draft_coverage(episode_id, input_hash, index, [r["id"] for r in chunk])
            parts.append(part)
        data = parts[0] if len(parts) == 1 else {
            "title": " / ".join(dict.fromkeys(str(p.get("title") or "") for p in parts if p.get("title"))),
            "summary": " ".join(str(p.get("summary") or "") for p in parts),
            "claims": [c for p in parts for c in (p.get("claims") or [])],
        }
        by_id = {r["id"]: r for r in rows}
        t0 = min(r["started_at"] for r in rows)
        t1 = max(r["ended_at"] for r in rows)
        counts = {"event": 0, "durable": 0, "action": 0}
        span_min = (t1 - t0) / 60
        caps = {"event": max_events(span_min) * len(chunks), "durable": T.DRAFT_MAX_DURABLE * len(chunks),
                "action": T.DRAFT_MAX_ACTIONS * len(chunks)}
        names_seen: list[str] = []
        drafts: list[ClaimDraft] = []
        seen: set[tuple[str, str]] = set()
        for item in data.get("claims") or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            if kind not in _KINDS:
                outcome.reject("bad_kind")
                continue
            action = action_verb(item) if kind == "event" else ""
            bucket = "action" if action else ("event" if kind == "event" else "durable")
            if counts[bucket] >= caps[bucket]:
                outcome.reject("over_cap")
                continue
            cd = self._validate(item, kind, by_id, episode_id, t0, t1, outcome)
            if cd is None:
                continue
            cd.action = action
            key = (cd.kind, tk.norm_key(cd.text))
            if key in seen:
                outcome.reject("duplicate")
                continue
            seen.add(key)
            counts[bucket] += 1
            drafts.append(cd)
        drafts = await self._verify(drafts, by_id, outcome)
        kept = []
        for cd in drafts:
            if await self.store.suppressed(cd.kind, cd.text):
                outcome.reject("forgotten")  # the user told us to forget this one
            else:
                kept.append(cd)
        drafts = kept
        writes = [await self._action_write(cd) for cd in drafts if cd.kind == "event"]
        durables = [_Durable(cd=cd, text=claim_text(cd.text, cd.context)) for cd in drafts if cd.kind != "event"]
        writes += await self._merge(durables)
        for w in writes:
            names_seen += w.draft.entities
            haystack = "\n".join([w.draft.text, *(e["quote"] for e in w.draft.evidence)])
            for name in sorted(set(w.draft.entities)):
                if tk.contains_phrase(haystack, name):
                    eid = await resolve_existing(self.store, name)
                    if eid is not None:
                        w.mentions.append(eid)

        # Entity candidates (planS2 §10.2): VLM working entities, claim entities, the
        # structural pass on AX text, and ground-truth app/site where the user acted.
        for r in rows:
            names_seen += tk.parse_json_list(r["entities"])
            if r["text_source"] == "ax":
                names_seen += structural_candidates(r["screen_text"])
            if r["keys"] or r["clicks"] or r["clipboard"]:
                names_seen += [x for x in (r["app"], r["site"]) if x]
        ops = await self.store.commit_episode_draft(
            episode_id, title=tk.truncate(str(data.get("title") or ""), 80),
            summary=tk.truncate(str(data.get("summary") or ""), 600), apps=apps, sites=sites,
            engaged_fraction=engaged, writes=writes,
            candidates=sorted({n.strip() for n in names_seen if n and promotable(n)}), rejected=outcome.rejected,
        )
        for op in ops:
            setattr(outcome, op, getattr(outcome, op) + 1)
        return outcome

    async def _chrome(self, rows) -> dict[tuple[str, str], set[str]]:
        """Per (app, site): lines that are navigation/toolbar, not content (textkit.chrome_lines)."""
        if T.CHROME_SHARE <= 0:
            return {}
        out = {}
        for app, site in {(r["app"], r["site"]) for r in rows}:
            texts = await self.store.recent_texts(app, site, before=max(r["ended_at"] for r in rows),
                                                  limit=T.CHROME_WINDOW)
            out[(app, site)] = tk.chrome_lines(texts, share=T.CHROME_SHARE, min_texts=T.CHROME_MIN_CAPTURES)
        return out

    async def _action_write(self, cd: ClaimDraft) -> ClaimWrite:
        """An action already remembered (the same verb on the same thing, close in time:
        a download page and then the file in Finder, or two episodes cut from one task)
        reinforces the earlier claim instead of adding a duplicate. Other events are new."""
        if not cd.action:
            return ClaimWrite(cd)
        mine = _thing_tokens(cd.text)
        for row in await self.store.q(
                "SELECT id, text FROM claims WHERE status='live' AND kind='event' AND action=? "
                "AND occurred_from BETWEEN ? AND ?",
                (cd.action, cd.occurred_from - T.ACTION_DEDUP_WINDOW_S, cd.occurred_to + T.ACTION_DEDUP_WINDOW_S)):
            theirs = _thing_tokens(row["text"])
            if mine and theirs and len(mine & theirs) / len(mine | theirs) >= T.ACTION_DEDUP_JACCARD:
                return ClaimWrite(cd, operation="reinforced", target_id=row["id"])
        return ClaimWrite(cd)

    async def _call_draft(self, rows, chrome: dict[tuple[str, str], set[str]] | None = None) -> dict:
        prompt = self._draft_prompt(rows, chrome, record_losses=True)
        data = await complete_json(self.ep, [{"role": "user", "content": prompt}], max_tokens=2500)
        if not isinstance(data, dict):
            raise LLMError("draft: not an object")
        return data

    @staticmethod
    def _draft_input_hash(rows, prompts: list[str]) -> str:
        """Invalidate all chunks when meaningful input anywhere in the episode changes."""
        return hashlib.sha256(json.dumps({
            "observations": [{key: r[key] for key in _DRAFT_ROW_FIELDS} for r in rows],
            "prompts": prompts,
        }, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _draft_prompt(self, rows, chrome: dict[tuple[str, str], set[str]] | None = None,
                      *, record_losses: bool = False) -> str:
        seen_lines: set[str] = set()
        blocks = []
        for r in rows:
            kept = []
            text = tk.without_lines(r["screen_text"] or "", (chrome or {}).get((r["app"], r["site"]), set()))
            for line in text.splitlines():
                k = tk.norm_key(line)
                if k and k not in seen_lines:
                    seen_lines.add(k)
                    kept.append(line)
            content = "\n".join(kept)
            if record_losses and len(content) > T.DRAFT_TEXT_CHARS_PER_OBS:
                PERF.count("health.draft_text_truncated")
            excerpt = tk.truncate(content, T.DRAFT_TEXT_CHARS_PER_OBS)
            acted = "acted" if (r["keys"] or r["clicks"] or r["clipboard"]) else "viewed only"
            page = f" page={r['page_title']!r}" if r["page_title"] else ""
            blocks.append(
                f"[observation_id={r['id']}] {to_local(r['started_at']):%H:%M} app={r['app']} site={r['site']} "
                f"window={r['window_title']!r}{page} ({acted}: keys={r['keys']} clicks={r['clicks']} "
                f"scrolls={r['scrolls']} copies={r['clipboard']})\n"
                f"caption: {r['caption']}\n<<<DATA>>>\n{excerpt}\n<<<END>>>"
            )
        span_min = (max(r["ended_at"] for r in rows) - min(r["started_at"] for r in rows)) / 60
        prompt = DRAFT_EPISODE_PROMPT.format(
            observations="\n\n".join(blocks), max_events=max_events(span_min), max_durable=T.DRAFT_MAX_DURABLE
        )
        # Deterministic detectors (process/actions.py) point the drafter at completed
        # actions and choices it might skim past; it still has to quote and cite them.
        if found := hints(detect(list(rows))):
            prompt += f"\n{found}\n"
        return prompt

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
            quote, ok, found_in = verify_evidence(text, str(ev.get("quote") or ""), obs)
            if ok:
                evidence.append({"observation_id": oid, "quote": quote, "field": found_in})
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
            # When it happened: the cited moments, not the whole episode. "Why did I…"
            # orders an action against what came just before it (trails.py).
            occurred_from=min(times),
            occurred_to=max(max(o["ended_at"], o["started_at"]) for o in cited),
            entities=entities, evidence=evidence,
        )

    async def _verify(self, drafts: list[ClaimDraft], by_id: dict, outcome: DraftOutcome) -> list[ClaimDraft]:
        """One independent support check per episode, before any claim is persisted."""
        if not drafts:
            return []
        with PERF.timer("draft.verify", n=len(drafts)):
            decisions = await verify_claims(drafts, by_id)
        accepted = []
        for i, cd in enumerate(drafts):
            if decisions[i]:
                accepted.append(cd)
            else:
                outcome.reject("not_entailed")
        return accepted

    # ----------------------------------------------------------- online merge

    async def _merge(self, durables: list[_Durable]) -> list[ClaimWrite]:
        """planS2 §6.2: ADD / REINFORCE / INVALIDATE for an episode's durable claims.
        One embeddings call and one relation call per episode, however many claims;
        each new claim's vector is stored with it, so it is never embedded again."""
        if not durables:
            return []
        try:
            vecs = await embed([d.text for d in durables])
        except (NotConfigured, *EXPECTED_ERRORS):
            vecs = [None] * len(durables)
        kept = []
        for d, v in zip(durables, vecs, strict=True):
            d.vec = v
            if v is not None and await self.store.suppressed(d.cd.kind, d.cd.text, v):
                continue  # close to a forgotten claim: not re-learned
            d.candidates = await self._candidates(d)
            kept.append(d)
        durables = kept
        await self._relate(durables)
        return [self._operation(d) for d in durables]

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
        data = await complete_json(self.ep, [{"role": "user", "content": RELATE_PROMPT.format(blocks=blocks)}],
                                   max_tokens=150 + 60 * sum(len(d.candidates) for _, d in asked))
        if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
            raise LLMError("relations: not a complete object")
        expected = {(n, c["id"]) for n, d in asked for c in d.candidates}
        decisions = {}
        for r in data["relations"]:
            if not isinstance(r, dict):
                raise LLMError("relations: invalid decision")
            n, rid, rel = r.get("new"), r.get("id"), r.get("relation")
            if type(n) is not int or type(rid) is not int or rel not in _RELATIONS:
                raise LLMError("relations: invalid decision")
            key = (n, rid)
            if key not in expected or key in decisions:
                raise LLMError("relations: duplicate or unknown pair")
            decisions[key] = rel
        if set(decisions) != expected:
            raise LLMError("relations: incomplete decisions")
        for (n, rid), rel in decisions.items():
            durables[n].relations[rid] = rel

    def _operation(self, d: _Durable) -> ClaimWrite:
        """Prepare a mutation without writing. The store commits all mutations together."""
        cd = d.cd
        w = ClaimWrite(cd, vector=d.vec, text_hash=tk.content_hash(d.text))
        by_rel = {rel: [c for c in d.candidates if d.relations.get(c["id"]) == rel]
                  for rel in ("same", "contradicts", "refines")}
        if by_rel["same"]:
            w.operation, w.target_id = "reinforced", by_rel["same"][0]["id"]
        elif by_rel["contradicts"]:
            old = by_rel["contradicts"][0]
            if old["modality"] == "stated" or (cd.modality == "passive" and old["modality"] == "engaged"):
                w.contested_id = old["id"]
            else:
                w.operation, w.target_id = "invalidated", old["id"]
        elif by_rel["refines"]:
            w.refines_id = by_rel["refines"][0]["id"]
        return w
