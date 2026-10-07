"""Model-free effect localization and bounded backward attention provenance walks."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from .. import textkit as tk
from ..config import T
from ..models import AttentionMoment, Hit, ProvenanceEdge, QueryPlan, WhyChain
from ..process.provenance import (
    distinctive,
    find_quote,
    infer_edges,
    is_origin,
    lexical_terms,
    rules_key,
    search_intent,
    shared_source,
    valid_edge,
)
from ..store import Store
from ..timeutil import to_local
from .query import fallback_plan, is_why

_CLOCK = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\s*(am|pm)?\b", re.I)
_SCAFFOLD = frozenset("why made led prompted caused did download downloaded browser open opened look looked "
                      "reading read start started browsing end ended page profile account around tried try "
                      "data when say import computer get got".split())
_ACTION = re.compile(r"\b(download|install|send|sent|buy|bought|create|delete|choose|chose)\w*\b", re.I)


def effect_terms(question: str) -> list[str]:
    text = _CLOCK.sub("", question)
    return sorted(distinctive(text) - _SCAFFOLD)


def _clock_distance(moment: AttentionMoment, question: str) -> float | None:
    m = _CLOCK.search(question)
    if not m:
        return None
    hour, minute = int(m[1]), int(m[2])
    if m[3]:
        hour = hour % 12 + (12 if m[3].lower() == "pm" else 0)
    local = to_local(moment.at)
    difference = abs((local.hour * 60 + local.minute) * 60 + local.second - (hour * 60 + minute) * 60)
    return min(difference, 86400 - difference)


async def locate_effect(store: Store, question: str, plan: QueryPlan, hits: list[Hit]) -> AttentionMoment | None:
    terms = effect_terms(question)
    if not terms:
        return None
    anchors = []
    action = _ACTION.search(question)
    for h in hits:
        if h.ref_kind == "observation" and not h.via:
            anchors.append(h.ref_id)
        elif h.ref_kind == "claim" and h.claim_kind == "event":
            anchors += [e["observation_id"] for e in await store.evidence_for_claim(h.ref_id)]
        elif h.ref_kind == "episode":
            anchors += [r["id"] for r in await store.observations_for_episode(h.ref_id)]
    action_anchors = set()
    if action:
        # A completed action's citation is the effect, rather than a product's home
        # page. Live rows only: retired claims never anchor a chain.
        claims = await store.fts_ids("claim", " ".join(terms), limit=T.WHY_EFFECT_POOL)
        for claim in await store.live_rows("claim", claims):
            if claim["action"] and action[0].lower()[:4] == claim["action"].lower()[:4]:
                action_anchors.update(e["observation_id"] for e in await store.evidence_for_claim(claim["id"]))
        anchors += list(action_anchors)
    moments = await store.effect_moments(terms, observation_ids=anchors, interval=plan.interval, limit=T.WHY_EFFECT_POOL)
    if not moments:
        return None
    df, n = await store.df_lookup(terms)
    weights = {t: tk.bm25_idf(df.get(t, 0), n) + 1 for t in terms}
    quoted = re.findall(r'["“]([^"”]{4,200})["”]', question)
    domains = re.findall(r"\b[a-z0-9-]+\.[a-z]{2,}\b", question, re.I)
    query_words = lexical_terms(" ".join(terms))
    ranked = []
    wants_profile = not quoted and bool(re.search(r"\bprofiles?\b", question, re.I))
    wants_search = bool(re.search(r"\bsearch(?:ed|ing)?\b", question, re.I))
    for moment in moments:
        distance = _clock_distance(moment, question)
        if distance is not None and distance > max(0, T.WHY_TIME_TOLERANCE_S):
            continue
        title = distinctive(moment.title + " " + moment.app + " " + moment.fields.get("window_title", "") +
                            " " + moment.fields.get("url", ""))
        body = distinctive(moment.text)
        present = set(terms) & body
        identity = lexical_terms(moment.title + " " + moment.app + " " + moment.fields.get("window_title", "") +
                                 " " + moment.fields.get("url", ""))
        normalized_body = lexical_terms(moment.text)
        topical = query_words & normalized_body
        if not topical or (domains and not any(d.casefold() in moment.fields.get("url", "").casefold() for d in domains)):
            continue
        if quoted and not any(find_quote(moment, q) for q in quoted):
            continue
        if wants_profile and not re.search(r"\bprofile\b", moment.title + " " + moment.fields.get("caption", ""), re.I):
            continue
        if wants_search and not search_intent(moment):
            continue
        if not action and not quoted and not (identity & query_words):
            continue
        if len(topical) / max(1, len(query_words)) < 0.65:
            continue
        # At least one distinctive subject must match. A dialog's own quoted text
        # takes precedence over contextual mentions of its application elsewhere.
        score = sum(weights[t] * (3 if t in title else 0.4) for t in present)
        score += 12 * sum(bool(find_quote(moment, q)) for q in quoted)
        if moment.keys and re.search(r"\b(?:what is|explain|all about)\b", moment.text, re.I):
            # A typed request shortly after arrival exposes the user's actual intent.
            urls = set(re.findall(r"https?://[^\s<>\"']+", moment.fields.get("caption", "") + " " +
                                  "\n".join(line for line in moment.fields.get("screen_text", "").splitlines()
                                            if re.search(r"\b(?:what is|explain|all about)\b", line, re.I))))
            score += min(2, len(urls)) * 3
        if action:
            score += 8 if moment.observation_id in action_anchors else 0
            score += 3 if re.search(re.escape(action[0][:4]), moment.title + " " +
                                   moment.fields.get("caption", ""), re.I) else 0
        if distance is not None:
            score += 4 * (1 - distance / max(1, T.WHY_TIME_TOLERANCE_S))
        ranked.append((score, -moment.at, moment))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not action and not quoted and not _CLOCK.search(question) and ranked:
        # Navigation/attention begins at the first relevant visit, provided all
        # requested subject words are represented. Later headings can be richer.
        complete = [r for r in ranked if query_words <= lexical_terms(r[2].text) and (
            re.search(r"\bstart(?:ed)?\b", question, re.I) or query_words <= lexical_terms(
                r[2].title + " " + r[2].fields.get("window_title", "") + " " + r[2].fields.get("url", "")))]
        if complete:
            return min(complete, key=lambda r: r[2].at)[2]
    return ranked[0][2] if ranked else None


def walk_back(effect: AttentionMoment, moments: list[AttentionMoment], edges: list[ProvenanceEdge]) -> WhyChain:
    """Prefer high confidence; stop at a real origin, reject cycles/future/stale quotes.

    Usually a single path; multiple clipboard sources are retained because a typed
    request may contain two independently copied references.
    """
    by_id = {m.sighting_id: m for m in moments}
    incoming = defaultdict(list)
    start = effect.at - max(0, T.WHY_CHAIN_WINDOW_S)
    for edge in edges:
        if valid_edge(edge, by_id) and by_id[edge.source].at >= start and edge.confidence >= T.WHY_CHAIN_MIN_CONFIDENCE:
            incoming[edge.target].append(edge)
    selected = {}
    visited = {effect.sighting_id}

    def walk(node: AttentionMoment, depth: int):
        if depth >= max(0, T.WHY_CHAIN_DEPTH) or (node is not effect and is_origin(node)):
            return
        parents = sorted(incoming[node.sighting_id], key=lambda e: (-e.confidence, e.gap_s, -e.source))
        # A post/media is an origin unless another message actually shares it.
        # A same-site visit transition alone must not carry unrelated earlier work
        # backward through a feed into a new subject.
        if node is not effect and shared_source(node) and not search_intent(node):
            parents = [e for e in parents if e.kind == "message_share" and is_origin(by_id[e.source])]
        if not parents:
            return
        # Small confidence differences should not skip immediate, quoted steps.
        preferred = [e for e in parents if e.confidence >= parents[0].confidence - 0.05]
        candidates = [min(preferred, key=lambda e: e.gap_s)]
        if parents[0].kind == "clipboard":
            candidates = [e for e in parents if e.kind == "clipboard"][:3]
        state = [e for e in parents if e.kind == "state_context"]
        if state:
            selection = next((e for e in parents if e.kind == "state_selection"), None)
            candidates = [*([selection] if selection else []), state[0]]
        # Reserve immediate sources before following their ancestors, so one
        # branch cannot absorb another independently copied reference.
        fresh_sources = {e.source for e in candidates if e.source not in visited}
        visited.update(e.source for e in candidates)
        for edge in candidates:
            selected[(edge.source, edge.target)] = edge
            if edge.source in fresh_sources and edge.kind not in ("state_context", "state_selection", "clipboard"):
                walk(by_id[edge.source], depth + 1)

    walk(effect, 0)
    kept = sorted((by_id[i] for i in visited), key=lambda m: (m.at, m.sighting_id))
    ordered_edges = sorted(selected.values(), key=lambda e: (by_id[e.source].at, by_id[e.target].at))
    return WhyChain(effect, kept, ordered_edges, reason="observed provenance" if selected else "why not known")


async def resolve_why(store: Store, question: str, *, plan: QueryPlan | None = None, hits=()) -> WhyChain:
    """Read-only, no embeddings/planner/verifier/writer calls. Old stores work too."""
    plan = plan or fallback_plan(question)
    if not T.PROVENANCE or not is_why(question, plan):
        return WhyChain(reason="not a why question")
    effect = await locate_effect(store, question, plan, list(hits))
    if effect is None:
        return WhyChain()
    start = effect.at - max(0, T.WHY_CHAIN_WINDOW_S)
    moments = await store.attention_moments(start, effect.at)
    stored = await store.provenance_edges(start, effect.at, rules_key=rules_key())
    # Fill missing/stale destinations without writes. This makes migration/backfill
    # optional for a read-only diagnosis and handles captures waiting for maintenance.
    by_id = {m.sighting_id: m for m in moments}
    edges = [e for e in stored if valid_edge(e, by_id)]
    have = {e.target for e in edges}
    for moment in moments:
        if moment.sighting_id not in have:
            neighborhood = [m for m in moments if moment.at - T.PROVENANCE_WINDOW_S <= m.at <= moment.at]
            edges += infer_edges(neighborhood, target_id=moment.sighting_id)
    chain = walk_back(effect, moments, edges)
    # Existing independently verified action trails complement the visit graph.
    claim_ids = [h.ref_id for h in hits if h.ref_kind == "claim" and h.action]
    if not claim_ids:
        claims = await store.fts_ids("claim", " ".join(effect_terms(question)), limit=T.WHY_TARGETS)
        claim_ids = [r["id"] for r in await store.live_rows("claim", claims) if r["action"] and
                     any(e["observation_id"] == effect.observation_id for e in await store.evidence_for_claim(r["id"]))]
    for link in await store.links_into("claim", claim_ids):
        if link["src_id"] not in {m.observation_id for m in moments}:
            continue
        evidence = json.loads(link["evidence"] or "[]")
        source = next(m for m in moments if m.observation_id == link["src_id"])
        if evidence and find_quote(source, evidence[0].get("quote", "")):
            chain.trails.append(dict(link))
    return chain


def chain_hits(chain: WhyChain) -> list[Hit]:
    """Compact ordered raw records for the regular support verifier and writer."""
    if chain.effect is None or not chain.edges:
        return []
    out = []
    by_id = {m.sighting_id: m for m in chain.moments}
    for moment in chain.moments:
        quotes = []
        context = []
        for edge in chain.edges:
            if moment.sighting_id == edge.source:
                quotes.append(f"> [{edge.source_field}] {edge.source_quote}")
                target = by_id[edge.target]
                context.append(f"{edge.kind} → sighting {edge.target} ({target.title[:100]}) at "
                               f"{to_local(target.at):%H:%M:%S}; confidence {edge.confidence:.2f}; gap {edge.gap_s:.0f}s")
            if moment.sighting_id == edge.target:
                quotes.append(f"> [{edge.target_field}] {edge.target_quote}")
        via = "attention provenance (inferred, requires support check)"
        if moment.sighting_id == chain.effect.sighting_id:
            via += "; effect"
        if context:
            via += "; " + "; ".join(context)
        out.append(Hit("observation", moment.observation_id, f"{moment.app} — {moment.title}",
                       "\n".join(dict.fromkeys(quotes)), occurred_at=moment.at, occurred_to=moment.ended_at,
                       url=moment.fields.get("url", ""), untrusted=True, via=via, confidence=chain.confidence))
    return out


def render_chain(chain: WhyChain, *, details: bool = True) -> str:
    if chain.effect is None:
        return chain.reason
    by_id = {m.sighting_id: m for m in chain.moments}
    lines = [f"{chain.reason}; confidence={chain.confidence:.2f}",
             f"effect: observation {chain.effect.observation_id} / sighting {chain.effect.sighting_id} "
             f"{to_local(chain.effect.at):%Y-%m-%d %H:%M:%S} {chain.effect.title}"]
    for edge in chain.edges:
        source, target = by_id[edge.source], by_id[edge.target]
        lines.append(f"{to_local(source.at):%H:%M:%S} obs {source.observation_id} [{source.title}] "
                     f"--{edge.kind} {edge.confidence:.2f}, {edge.gap_s:.0f}s--> "
                     f"{to_local(target.at):%H:%M:%S} obs {target.observation_id} [{target.title}]")
        if details:
            lines += [f"  source [{edge.source_field}]: {edge.source_quote}",
                      f"  target [{edge.target_field}]: {edge.target_quote}"]
    for trail in chain.trails:
        lines.append(f"verified action trail: obs {trail['src_id']} {trail['relation']} claim {trail['dst_id']}: {trail['reason']}")
    return "\n".join(lines)
