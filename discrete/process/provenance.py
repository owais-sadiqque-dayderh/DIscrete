"""Deterministic attention provenance. These are observed associations, not motives.

Every edge quotes both endpoints. Counts establish interaction, never clipboard
contents or process liveness. Captions/referrer fields are already stored model
evidence and receive lower confidence than verbatim URL/text matches.
"""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import parse_qs, unquote, urlsplit

from .. import textkit as tk
from ..config import T
from ..models import AttentionMoment, ProvenanceEdge
from ..store import Store
from ..timeutil import now

_GENERIC = frozenset("app apps browser profile page post posts search results google site website video image photo "
                     "model models user reads reading views viewing opened open tab new space home menu file edit "
                     "view window tabs tools links link download downloaded downloading install installed importing "
                     "import data query typed typing text input address bar account click clicked message messages https http www com "
                     "notifications share sent shows showing software using use about around start browsing end up".split())
_SHARE = re.compile(r"\b(?:direct messages?|direct-message|dm|sent a post|shared|repost|quoted post|post|tweet|notification)\b", re.I)
_MEDIA = re.compile(r"\b(?:video|demo|caption|embedded|image|photo|screenshot|media)\b", re.I)
_COPY = re.compile(r"\b(?:copied|copy|clipboard|paste[ds]?)\b", re.I)
_TYPED = re.compile(r"\b(?:typ(?:ed|ing|es)|enter(?:ed|ing|s)|search(?:ed|ing|es)? for|query)\b", re.I)
_ERROR = re.compile(r"\b(?:still in use|profile.{0,25}(?:locked|in use)|already running|close the browser|close .{0,30}first)\b", re.I)
_URL = re.compile(r"(?:https?://|\bwww\.)[^\s<>\"']+|\b[a-z0-9][a-z0-9-]*\.(?:[a-z]{2,})(?:/[^\s<>\"']*)?", re.I)


def distinctive(text: str) -> set[str]:
    return {t for t in tk.content_tokens(text, min_len=3) if t not in _GENERIC and not t.isdigit()}


def shared_source(moment: AttentionMoment) -> bool:
    return bool(_SHARE.search(moment.title + " " + moment.fields.get("caption", "")) or
                re.search(r"\b(?:sent a post|shared a (?:post|link)|direct-message conversation)\b", moment.text, re.I))


def lexical_terms(text: str) -> set[str]:
    """Word inflections and URL hosts for effect identity; excludes URL schemes."""
    terms = distinctive(text)
    for match in _URL.finditer(text):
        url = match[0]
        host = urlsplit(url if "://" in url else "https://" + url).hostname or ""
        terms.update(host.removeprefix("www.").split(".")[:-1])
    return {t[:-1] if len(t) > 5 and t.endswith("s") else t for t in terms if t not in _GENERIC}


def find_quote(moment: AttentionMoment, value: str, *, fields=None) -> tuple[str, str] | None:
    """Return a literal stored substring, with nearby context; never synthesize a quote."""
    if not value.strip():
        return None
    pattern = re.compile(r"\s+".join(re.escape(p) for p in value.split()), re.I)
    for field in fields or moment.fields:
        text = moment.fields.get(field, "")
        if m := pattern.search(text):
            return field, text[max(0, m.start() - 90):min(len(text), m.end() + 130)]
    return None


def _url_key(url: str) -> str:
    return unquote(re.sub(r"^(?:https?://)?(?:www\.)?", "", url.strip(), flags=re.I)).rstrip("/.,)").casefold()


def url_quote(moment: AttentionMoment, url: str) -> tuple[str, str] | None:
    wanted = _url_key(url)
    for field in ("screen_text", "caption"):
        text = moment.fields.get(field, "")
        for match in _URL.finditer(text):
            if _url_key(match[0]) == wanted:
                return field, text[max(0, match.start() - 90):min(len(text), match.end() + 130)]
    return None


def search_intent(moment: AttentionMoment) -> tuple[str, tuple[str, str]] | None:
    url = moment.fields.get("url", "")
    params = parse_qs(urlsplit(url).query)
    for key in ("q", "query", "search", "text"):
        if params.get(key):
            return params[key][0], ("url", url)
    for field in ("window_title", "page_title"):
        text = moment.fields.get(field, "")
        if m := re.match(r"(.+?)\s*[-–|]\s*.*\bSearch\b", text, re.I):
            return m[1], (field, text)
    caption = moment.fields.get("caption", "")
    if moment.keys and _TYPED.search(caption) and re.search(r"\b(?:command|address|search) bar|search(?:ing|ed)? for", caption, re.I):
        if m := re.search(r'["“\']([^"”\']{3,250})["”\']', caption):
            return m[1], ("caption", caption)
        if m := _URL.search(caption):
            return m[0], ("caption", caption)
    return None


def is_origin(moment: AttentionMoment) -> bool:
    # An explicitly stated intent is an origin; a bare search is a step whose
    # provenance may itself be a post or embedded media mention.
    return bool(re.search(r"\b(?:direct.messages?|direct-message|dm|notification)\b", moment.title + " " +
                         moment.fields.get("caption", ""), re.I) or
                (moment.keys and (re.search(r"\b(?:what is|explain|help me|i want|i need|all about)\b",
                                           moment.fields.get("caption", ""), re.I) or
                                  re.search(r"^\s*(?:what is|i want|i need|help me)\b",
                                            moment.fields.get("screen_text", ""), re.I | re.M))))


def valid_edge(edge: ProvenanceEdge, by_id: dict[int, AttentionMoment]) -> bool:
    a, b = by_id.get(edge.source), by_id.get(edge.target)
    return bool(a and b and edge.source != edge.target and (a.at, a.sighting_id) < (b.at, b.sighting_id)
                and (not a.session_id or not b.session_id or a.session_id == b.session_id)
                and 0 <= b.at - a.at <= max(0, T.PROVENANCE_WINDOW_S)
                and abs(edge.gap_s - (b.at - a.at)) < 0.001
                and 0 <= edge.confidence <= 1
                and edge.source_quote and edge.source_quote in a.fields.get(edge.source_field, "")
                and edge.target_quote and edge.target_quote in b.fields.get(edge.target_field, ""))


def infer_edges(moments: list[AttentionMoment], *, target_id: int | None = None) -> list[ProvenanceEdge]:
    """Bounded temporal neighborhood; no edge from temporal adjacency alone."""
    ordered = sorted(moments, key=lambda m: (m.at, m.sighting_id))
    df = Counter(t for m in ordered for t in distinctive(m.text))
    edges = []
    for i, b in enumerate(ordered):
        if target_id is not None and b.sighting_id != target_id:
            continue
        parents = []
        intent_b = search_intent(b)
        for a in reversed(ordered[:i]):
            gap = b.at - a.at
            if gap > max(0, T.PROVENANCE_WINDOW_S):
                break
            if a.observation_id == b.observation_id or (a.session_id and b.session_id and a.session_id != b.session_id):
                continue
            same_page = ((a.fields.get("url") and _url_key(a.fields["url"]) == _url_key(b.fields.get("url", ""))) or
                         (a.app == b.app and tk.norm_key(a.title) == tk.norm_key(b.title)))
            proposed: list[ProvenanceEdge] = []

            def add(kind, confidence, qa, qb, proposed=proposed, a=a, b=b, gap=gap):
                if qa and qb:
                    # Caption/referrer assertions are stored evidence, not browser telemetry.
                    if "caption" in (qa[0], qb[0]):
                        confidence = min(confidence, 0.78)
                    proposed.append(ProvenanceEdge(a.sighting_id, b.sighting_id, kind, confidence,
                                                  *qa, *qb, gap))

            # Explicit referrer name/URL, checked against a real earlier page.
            came = b.fields.get("navigated_from", "")
            if came and not same_page:
                for field in ("url", "page_title", "window_title"):
                    value = a.fields.get(field, "")
                    if distinctive(value) and (tk.norm_key(value) in tk.norm_key(came) or
                                               tk.norm_key(came) in tk.norm_key(value)):
                        add("navigated_from", 0.88, (field, value), ("navigated_from", came))
                        break

            # A copy must have a payload match; the clipboard counter alone is insufficient.
            if a.clipboard or re.search(r"\b(?:copied|copies|copying|pasted)\b", a.fields.get("caption", ""), re.I) or (
                re.search(r"\bCopied (?:Current URL|to clipboard)\b", a.fields.get("screen_text", ""), re.I)):
                values = [a.fields.get("url", "")]
                copied = re.search(r"\bcopied\s+[\"“']([^\"”']+)[\"”']", a.fields.get("caption", ""), re.I)
                if copied:
                    values.append(copied[1])
                if b.keys or re.search(r"\b(?:pasted|pastes|pasting)\b", b.fields.get("caption", ""), re.I):
                    for value in values:
                        qb = find_quote(b, value, fields=("caption",)) if value else None
                        if qb is None and value:
                            # The URL must occur in a content/input line, not merely
                            # a browser sidebar containing already open tabs.
                            for line in b.fields.get("screen_text", "").splitlines():
                                if value.casefold() in line.casefold() and (re.search(
                                    r"\b(?:what|explain|about|summar|compare|read|paste|copied)\b", line, re.I) or b.clipboard):
                                    qb = ("screen_text", line)
                                    break
                            if qb is None and re.search(r"\b(?:what is|explain|all about)\b", b.text, re.I):
                                qb = find_quote(b, value, fields=("screen_text",))
                        if value and qb:
                            add("clipboard", 0.96, find_quote(a, value), qb)
                            break

            # A real link or distinctive page label present on the source screen.
            if not same_page:
                values = [b.fields.get("url", ""), b.fields.get("page_title", ""), b.fields.get("window_title", "")]
                for value in values:
                    if not distinctive(value) or (value != b.fields.get("url") and len(distinctive(value)) < 2):
                        continue
                    qa = (url_quote(a, value) if value == b.fields.get("url") else
                          find_quote(a, value, fields=("screen_text", "caption")))
                    qb = find_quote(b, value)
                    if qa and qb:
                        add("search" if search_intent(a) else "message_share" if shared_source(a) else "navigation",
                            0.9, qa, qb)
                        break

                # Successive pages on the same site supply a weaker visit transition
                # when no clicked link/referrer was captured. Quotes are the real URLs.
                ua, ub = a.fields.get("url", ""), b.fields.get("url", "")
                if ua and ub and gap <= 90 and urlsplit(ua).hostname == urlsplit(ub).hostname and (
                    urlsplit(ua).hostname and _url_key(ua) != _url_key(ub) and not search_intent(a)):
                    add("navigation", 0.64, ("url", ua), ("url", ub))
                    label = unquote(urlsplit(ub).path.rstrip("/").rsplit("/", 1)[-1]).replace("-", " ")
                    if len(label) >= 4 and not label.isdigit() and label.casefold() not in ("home", "index", "profile"):
                        add("navigation", 0.87, find_quote(a, label, fields=("screen_text",)), ("url", ub))

            intent_a = search_intent(a)
            if intent_a and not same_page:
                terms = lexical_terms(intent_a[0])
                matched = terms & lexical_terms(b.title + " " + b.fields.get("window_title", "") + " " +
                                                b.fields.get("url", ""))
                if matched:
                    term = min(matched, key=lambda t: df[t])
                    add("search", 0.86, intent_a[1], find_quote(b, term))

            # Product/brand named inside embedded media, then searched; also supports
            # a typed destination with shared subject terms (weaker evidence).
            destination = intent_b[0] if intent_b else b.title
            common = distinctive(destination) & distinctive(a.fields.get("screen_text", "") + " " +
                                                                        a.fields.get("caption", ""))
            rare = {t for t in common if df[t] <= max(3, len(ordered) * 0.15)}
            if rare:
                term = min(rare, key=lambda t: (df[t], -len(t), t))
                qa, qb = find_quote(a, term), find_quote(b, term)
                if _MEDIA.search(a.fields.get("caption", "")) and intent_b:
                    add("media_mention", 0.76, qa, qb)
                elif shared_source(a) and not same_page:
                    add("message_share", 0.72, qa, qb)
                elif a.app != b.app:
                    add("app_switch", 0.58, qa, qb)

            # An omnibox entry can have a different name from the motivating post
            # (a model announcement → a model registry). Preserve the current topic
            # only when the destination's stored caption explicitly shows that topic
            # alongside its typed intent, and it was also present on the source.
            if intent_b and b.keys and not intent_a:
                context_terms = distinctive(b.fields.get("caption", "")) & distinctive(
                    a.title + " " + a.fields.get("caption", ""))
                context_terms -= distinctive(intent_b[0])
                contextual = {t for t in context_terms if df[t] <= max(3, len(ordered) * 0.15) or len(t) >= 8}
                if contextual and shared_source(a):
                    term = min(contextual, key=lambda t: (df[t], -len(t), t))
                    add("typed_navigation", 0.68, find_quote(a, term), find_quote(b, term, fields=("caption",)))

            # State explanation: recent foreground evidence is only *possibly* still
            # running. Require its app to be named by an import/setup screen, or by
            # the dialog itself. Never assert actual background process liveness.
            if _ERROR.search(b.text) and a.app and a.app != b.app:
                setup = next((m for m in reversed(ordered[:i]) if b.at - m.at <= T.PROVENANCE_WINDOW_S
                              and m.app == b.app and re.search(r"\bimport", m.text, re.I)
                              and find_quote(m, a.app)), None)
                if find_quote(b, a.app) or setup:
                    add("state_context", 0.6, ("app", a.app), find_quote(b, _ERROR.search(b.text)[0]))
            elif _ERROR.search(b.text) and a.app == b.app and re.search(r"\bimport", a.text, re.I):
                add("state_selection", 0.7, find_quote(a, "import"), find_quote(b, _ERROR.search(b.text)[0]))

            if proposed:
                best = max(proposed, key=lambda e: e.confidence)
                if valid_edge(best, {a.sighting_id: a, b.sighting_id: b}):
                    parents.append(best)
        # Preserve alternative origins, especially two copied URLs; at most one
        # parent of each kind/page except clipboard sources with separate payloads.
        parents.sort(key=lambda e: (-e.confidence, e.gap_s, -e.source))
        unique = []
        seen = set()
        by_id = {m.sighting_id: m for m in ordered}
        for edge in parents:
            source = by_id[edge.source]
            identity = (edge.kind, source.app) if edge.kind == "state_context" else (edge.kind, source.observation_id)
            if identity not in seen:
                unique.append(edge)
                seen.add(identity)
        edges += unique[:max(0, T.PROVENANCE_MAX_PARENTS)]
    return edges


def rules_key() -> str:
    return f"v1:{T.PROVENANCE_WINDOW_S}:{T.PROVENANCE_MAX_PARENTS}"


async def build_provenance(store: Store, *, backfill: bool = False) -> int:
    """Incremental and restartable, also usable without a configured model."""
    if not T.PROVENANCE:
        return 0
    if backfill:
        await store.reset_provenance()
    made = 0
    while pending := await store.pending_provenance(rules_key(), limit=max(1, T.PROVENANCE_BATCH),
                                                   window_s=max(0, T.PROVENANCE_WINDOW_S)):
        for moment in pending:
            snapshot_at = now()
            rows = await store.attention_moments(moment.at - T.PROVENANCE_WINDOW_S, moment.at)
            edges = infer_edges(rows, target_id=moment.sighting_id)
            checkpoint_at = max([snapshot_at, *(m.updated_at for m in rows)])
            made += await store.save_provenance(moment, edges, rules_key(), checkpoint_at=checkpoint_at)
        if not backfill:
            break
    return made
