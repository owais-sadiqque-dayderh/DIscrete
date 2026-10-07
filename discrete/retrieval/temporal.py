"""Model-free event anchors and visit-level relative timelines."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import textkit as tk
from ..config import _typed
from ..models import AttentionMoment, Hit, RetrievalResult
from ..page_identity import canonical_url
from ..store import Store
from ..timeutil import to_local
from .query import EventQueryPlan, RelativeEvent

_ANCHOR = "relative event anchor"
_ACTION = re.compile(r"\b(?:typed|ran|runs|executed|entered|invoked)\b", re.I)
_COMPLETION = re.compile(r"^(?:finish(?:ed|ing)?|complet(?:ed|ing|e)|done with)\s+", re.I)


@dataclass
class EventAnchor:
    hit: Hit
    score: float
    session: str = ""
    identity: str = ""


def _terms(phrase: str) -> set[str]:
    # A small morphological fold matches action inflections without aliases for
    # particular products or commands. Explicit commands retain their exact form.
    return {_stem(t) for t in tk.content_tokens(_COMPLETION.sub("", phrase)) if t not in {"with", "using"}}


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith(("ed", "ing")):
        token = re.sub(r"(?:ing|ed)$", "", token)
        if len(token) > 2 and token[-1] == token[-2] and token[-1] not in "aeiouls":
            token = token[:-1]
    return token


def _command(phrase: str) -> str:
    quoted = re.findall(r"[`\"']([^`\"']+)[`\"']", phrase)
    if quoted:
        return quoted[-1].strip()
    m = re.search(r"\b(?:with|using|command|typed|ran|running|executed|executing)\s+(.+)$", phrase, re.I)
    return m[1].strip() if m else ""


def _overlap(terms: set[str], text: str) -> float:
    words = {_stem(t) for t in tk.tokens(text)}
    return len(terms & words) / len(terms) if terms else 0.0


def _contains(phrase: str, text: str) -> bool:
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(" ".join(tk.tokens(phrase))) + r"(?!\w)",
                                    " ".join(tk.tokens(text)), re.I))


def _identity(moment: AttentionMoment) -> str:
    fields = moment.fields
    return tk.norm_key(moment.app) + "|" + (_url(fields.get("url", "")) or
                                              tk.norm_key(moment.title))


def _url(raw: str) -> str:
    # Browser settings/onboarding URLs identify pages too. The shared canonical
    # helper intentionally accepts only web/filesystem URLs.
    return canonical_url(raw) or (raw.split("#", 1)[0].rstrip("/")
                                 if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I) else "")


def _hit(moment: AttentionMoment, *, via: str, terms: list[str] | None = None) -> Hit:
    from .hybrid import source_windows

    fields = moment.fields
    title = f"{moment.app} — {moment.title}" if moment.title != moment.app else moment.app
    # The event's own lines, wherever they are on the screen, so the verifier can see
    # the anchor evidence; not the whole screen (it goes to a remote verifier).
    screen, spans = source_windows(fields.get("screen_text", "") or "", terms or [])
    body = "\n".join(p for p in (fields.get("caption", ""), screen) if p)
    return Hit("observation", moment.observation_id, title[:120], body, strong=True,
               occurred_at=moment.at, occurred_to=moment.ended_at, untrusted=True,
               url=fields.get("url", ""), via=via, source_spans=spans)


def _moment_score(moment: AttentionMoment, event: RelativeEvent) -> float:
    terms = _terms(event.phrase)
    command = _command(event.phrase)
    text = moment.text
    overlap = _overlap(terms, text)
    if command:
        if not _contains(command, text):
            return 0.0
    elif not terms or overlap < 0.66 or (len(terms) > 1 and overlap * len(terms) < 2):
        return 0.0
    identity = " ".join(moment.fields.get(f, "") for f in ("window_title", "page_title", "url"))
    score = 8 * overlap + 12 * _overlap(terms, identity)
    if not event.last and not command:
        # An unquoted event phrase can itself be a shell command. Recognize it
        # from the captured prompt rather than maintaining a CLI-name vocabulary.
        screen = moment.fields.get("screen_text", "")
        if moment.keys and any(re.search(r"[%$>]\s*" + re.escape(event.phrase), line, re.I)
                               for line in screen.splitlines()):
            score += 40
        if moment.keys and _ACTION.search(moment.fields.get("caption", "")) and overlap == 1:
            score += 10
    if command:
        score += 30
        # Command help and scrollback mention the same words as actual input.
        # Require a shell prompt or an attributable action in the caption before
        # awarding the input bonus; a keyboard count alone is insufficient.
        screen = moment.fields.get("screen_text", "")
        prompted = any(re.search(r"[%$>]\s*" + re.escape(command), line, re.I) for line in screen.splitlines())
        caption = moment.fields.get("caption", "")
        acted = bool(_ACTION.search(caption) and _contains(command, caption))
        if not (prompted or acted) and re.search(r"\b(?:help|usage|available commands)\b", text, re.I):
            return 0.0
        if moment.keys and prompted:
            score += 40
        if acted:
            score += 20
    return score


async def resolve_anchor(store: Store, plan: EventQueryPlan) -> EventAnchor | None:
    event = plan.anchor
    if event is None:
        return None
    terms = list(_terms(event.phrase))
    # FTS + structured URL/title search supplies candidates; summaries and event
    # claims also supply observation IDs when the screen itself lacks the terms.
    ids: set[int] = set()
    summaries: list[Hit] = []
    from .hybrid import materialize

    for kind in ("episode", "claim"):
        rows = await store.live_rows(kind, await store.fts_ids(kind, " ".join(terms), limit=100))
        for row in rows:
            if kind == "claim" and row["kind"] != "event":
                continue
            text = f"{row['title']} {row['summary']}" if kind == "episode" else row["text"]
            if _overlap(set(terms), text) < 0.66:
                continue
            if kind == "episode":
                ids.update(r["id"] for r in await store.observations_for_episode(row["id"]))
            else:
                ids.update(r["observation_id"] for r in await store.evidence_for_claim(row["id"]))
            summaries += list((await materialize(store, kind, [row["id"]])).values())
    moments = await store.effect_moments(terms, observation_ids=ids, interval=plan.interval, limit=1000)
    words = list(dict.fromkeys(terms + tk.content_tokens(event.phrase)))  # stems and the words as typed
    candidates = [EventAnchor(_hit(m, via=_ANCHOR, terms=words), score, m.session_id, _identity(m))
                  for m in moments if (score := _moment_score(m, event)) > 0]
    if candidates:
        best = max(a.score for a in candidates)
        # Completion means the last matching visit of the activity, not the first
        # high-BM25 screen. Identity evidence outranks later logs about that activity.
        if event.last:
            candidates = [a for a in candidates if a.score >= best - 2]
            return max(candidates, key=lambda a: (a.hit.occurred_at, a.hit.occurred_to))
        return max(candidates, key=lambda a: (a.score, a.hit.occurred_at))
    summaries = [h for h in summaries if h.in_interval(plan.interval) and h.occurred_at and
                 (not _command(event.phrase) or _contains(_command(event.phrase), h.text))]
    if not summaries:
        return None
    hit = max(summaries, key=lambda h: (h.occurred_to if event.last else _overlap(set(terms), h.text), h.occurred_at))
    hit.via, hit.strong = _ANCHOR, True
    return EventAnchor(hit, 1)


async def relative_timeline(store: Store, result: RetrievalResult) -> bool:
    plan = result.plan
    if not isinstance(plan, EventQueryPlan) or plan.anchor is None:
        return False
    anchor = await resolve_anchor(store, plan)
    if anchor is None:
        return False
    after = plan.anchor.direction == "after"
    boundary = anchor.hit.occurred_to if after else anchor.hit.occurred_at
    span = max(0.0, _typed("TEMPORAL_AFTER_S" if after else "TEMPORAL_BEFORE_S", 300.0))
    limit = max(1, _typed("TEMPORAL_MAX_MOMENTS", 5))
    # Capture end/start pairs can overlap by a fraction of a millisecond.
    epsilon = 0.001
    moments = await store.attention_moments(boundary - epsilon if after else boundary - span,
                                            boundary + span if after else boundary + epsilon)
    if not after:
        moments.reverse()
    seen = {anchor.identity} if anchor.identity else set()
    seen_names: dict[str, str] = {}
    hits = []
    for moment in moments:
        if (after and moment.at < boundary - epsilon) or (not after and moment.ended_at > boundary + epsilon):
            continue
        if anchor.session and moment.session_id and anchor.session != moment.session_id:
            continue
        identity = _identity(moment)
        name = tk.norm_key(moment.app) + "|" + tk.norm_key(moment.title)
        url = _url(moment.fields.get("url", ""))
        if url and not canonical_url(url):
            url = ""  # internal/stale URLs cannot bridge a later real web URL
        # A navigation capture can have the previous internal URL, then the real
        # URL a moment later. Same title bridges that missing URL, but two known
        # different URLs remain distinct even when their page titles are identical.
        if name in seen_names and (not url or not seen_names[name] or url == seen_names[name]):
            continue
        if identity in seen or (moment.observation_id == anchor.hit.ref_id and anchor.hit.ref_kind == "observation"):
            continue
        seen.add(identity)
        seen_names[name] = url
        delta = moment.at - boundary if after else moment.ended_at - boundary
        via = (f"right {plan.anchor.direction} {anchor.hit.title} at {to_local(boundary):%H:%M:%S} "
               f"({delta:+.0f}s)")
        hits.append(_hit(moment, via=via))
        if len(hits) == limit:
            break
    # The anchor remains first in the evidence pack; neighboring visits follow in
    # temporal proximity order (nearest preceding visit first for 'before').
    result.hits = [anchor.hit, *hits]
    return True


def temporal_answer(result: RetrievalResult) -> None:
    """Build plain output only after support verification kept the anchor and visits."""
    if not isinstance(result.plan, EventQueryPlan) or not result.hits:
        return
    if result.hits[0].via != _ANCHOR or not result.hits[0].eligible:
        # A verified neighbor without the event does not establish this boundary.
        result.hits = []
        return
    neighbors = [h for h in result.hits[1:] if h.via.startswith("right ") and h.eligible]
    if neighbors:
        result.page_answer = "\n".join(f"- {h.title} ({to_local(h.occurred_at):%H:%M:%S})" for h in neighbors)
