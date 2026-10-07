"""Deterministic attention provenance. These are observed associations, not motives.

Every edge quotes both endpoints. Counts establish interaction, never clipboard
contents or process liveness. Captions/referrer fields are already stored model
evidence and receive lower confidence than verbatim URL/text matches.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from urllib.parse import parse_qs, unquote, urlsplit

from .. import textkit as tk
from ..capture.redact import redact, redact_url
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
_NAVIGATION = re.compile(r"\b(?:click(?:ed|ing)?|follow(?:ed|ing)?|open(?:ed|ing)?)\b", re.I)
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
            quote = text[max(0, m.start() - 90):min(len(text), m.end() + 130)]
            if safe_text(quote) == quote:
                return field, quote
    return None


def _url_key(url: str) -> str:
    return unquote(re.sub(r"^(?:https?://)?(?:www\.)?", "", url.strip(), flags=re.I)).rstrip("/.,)").casefold()


def url_quote(moment: AttentionMoment, url: str) -> tuple[str, str] | None:
    wanted = _url_key(url)
    for field in ("screen_text", "caption"):
        text = moment.fields.get(field, "")
        for match in _URL.finditer(text):
            if _url_key(match[0]) == wanted:
                quote = text[max(0, match.start() - 90):min(len(text), match.end() + 130)]
                if safe_text(quote) == quote:
                    return field, quote
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


def safe_text(text: str) -> str:
    """Always scrub graph evidence, including legacy rows and flag-off captures."""
    text = redact(text).text
    return re.sub(r"https?://[^\s<>\"']+", lambda m: redact_url(m[0]).text, text)


def safe_moment(moment: AttentionMoment) -> AttentionMoment:
    return replace(moment, fields={k: safe_text(v) for k, v in moment.fields.items()})


def valid_edge(edge: ProvenanceEdge, by_id: dict[int, AttentionMoment]) -> bool:
    a, b = by_id.get(edge.source), by_id.get(edge.target)
    return bool(a and b and edge.source != edge.target and (a.at, a.sighting_id) < (b.at, b.sighting_id)
                and (not a.session_id or not b.session_id or a.session_id == b.session_id)
                and 0 <= b.at - a.at <= max(0, T.PROVENANCE_WINDOW_S)
                and abs(edge.gap_s - (b.at - a.at)) < 0.001
                and 0 <= edge.confidence <= 1
                and edge.source_quote and safe_text(edge.source_quote) == edge.source_quote
                and safe_text(edge.target_quote) == edge.target_quote
                and edge.source_quote in a.fields.get(edge.source_field, "")
                and edge.target_quote and edge.target_quote in b.fields.get(edge.target_field, ""))


@dataclass
class MomentFeatures:
    moment: AttentionMoment
    full_terms: set[str]
    body_terms: set[str]
    context_terms: set[str]
    caption_terms: set[str]
    identity_terms: set[str]
    labels: set[str]
    intent: tuple[str, tuple[str, str]] | None
    intent_terms: set[str]
    intent_distinctive: set[str]
    destination_terms: set[str]
    url_key: str
    host: str
    shared: bool
    copied: bool
    updated_window: float = 0.0


def window_freshness(moments: list[AttentionMoment]) -> dict[int, float]:
    """Maximum source update per temporal window, including equal-time siblings."""
    maxima = deque()
    result = {}
    right = 0
    for moment in moments:
        while right < len(moments) and moments[right].at <= moment.at:
            while maxima and moments[maxima[-1]].updated_at <= moments[right].updated_at:
                maxima.pop()
            maxima.append(right)
            right += 1
        while maxima and moments[maxima[0]].at < moment.at - max(0, T.PROVENANCE_WINDOW_S):
            maxima.popleft()
        result[moment.sighting_id] = moments[maxima[0]].updated_at
    return result


class EdgeInference:
    """Extract once; indexed candidate joins within each destination's sliding window.

    URL/title/subject indexes avoid comparing every pair of unrelated visits, even
    when thousands of captures fall inside the same temporal window. Term document
    frequencies are window-local, so batch boundaries cannot change confidence.
    """

    def __init__(self, moments: list[AttentionMoment]):
        self.ordered = sorted((safe_moment(m) for m in moments), key=lambda m: (m.at, m.sighting_id))
        self.raw_by_id = {m.sighting_id: m for m in moments}
        self.by_id = {m.sighting_id: m for m in self.ordered}
        self.positions = {m.sighting_id: i for i, m in enumerate(self.ordered)}
        self.times = [m.at for m in self.ordered]
        self.features = []
        self.terms = defaultdict(list)
        self.identities = defaultdict(list)
        self.searches = defaultdict(list)
        self.links = defaultdict(list)
        self.path_labels = defaultdict(list)
        self.copies = defaultdict(list)
        self.hosts = defaultdict(list)
        self.apps = defaultdict(list)
        self.imports = defaultdict(list)
        freshness = window_freshness(self.ordered)
        for i, m in enumerate(self.ordered):
            f = m.fields
            body = f.get("screen_text", "") + " " + f.get("caption", "")
            identity = m.title + " " + f.get("window_title", "") + " " + f.get("url", "")
            intent = search_intent(m)
            copied = bool(m.clipboard or re.search(r"\b(?:copied|copies|copying|pasted)\b", f.get("caption", ""), re.I)
                          or re.search(r"\bCopied (?:Current URL|to clipboard)\b", f.get("screen_text", ""), re.I))
            feature = MomentFeatures(m, distinctive(m.text), distinctive(body),
                                     distinctive(m.title + " " + f.get("caption", "")), distinctive(f.get("caption", "")),
                                     lexical_terms(identity), distinctive(m.title + " " + f.get("window_title", "")),
                                     intent, lexical_terms(intent[0]) if intent else set(),
                                     distinctive(intent[0]) if intent else set(),
                                     distinctive(intent[0] if intent else m.title), _url_key(f.get("url", "")),
                                     urlsplit(f.get("url", "")).hostname or "", shared_source(m), copied)
            feature.updated_window = freshness[m.sighting_id]
            self.features.append(feature)
            for term in feature.full_terms:
                self.terms[term].append(i)
            for term in distinctive(identity):
                self.identities[term].append(i)
            for term in feature.intent_terms:
                self.searches[term].append(i)
            for match in _URL.finditer(body):
                self.links[_url_key(match[0])].append(i)
            if copied:
                for term in distinctive(f.get("url", "") + " " + f.get("caption", "")):
                    self.copies[term].append(i)
            if feature.host:
                self.hosts[feature.host].append(i)
                for term in set(tk.tokens(f.get("screen_text", ""))):
                    self.path_labels[(feature.host, term)].append(i)
            self.apps[m.app].append(i)
            if re.search(r"\bimport", m.text, re.I):
                self.imports[m.app].append(i)

    def _candidates(self, i: int, low: int) -> list[int]:
        b, fb = self.ordered[i], self.features[i]
        candidates = set()

        def collect(index, terms):
            for term in terms:
                postings = index.get(term, [])
                candidates.update(postings[bisect_left(postings, low):bisect_left(postings, i)])

        collect(self.terms, fb.labels | fb.destination_terms | (fb.caption_terms if fb.intent and b.keys else set()))
        collect(self.identities, distinctive(b.fields.get("navigated_from", "")))
        collect(self.searches, fb.identity_terms)
        collect(self.copies, fb.full_terms)
        collect(self.links, [fb.url_key] if fb.url_key else [])
        # A visible same-site path label can precede intervening unrelated pages.
        label = unquote(urlsplit(b.fields.get("url", "")).path.rstrip("/").rsplit("/", 1)[-1]).replace("-", " ")
        if len(label) >= 4 and not label.isdigit() and label.casefold() not in ("home", "index", "profile"):
            collect(self.path_labels, [(fb.host, t) for t in tk.tokens(label)])
        # Bare host succession contributes only the nearest contextual predecessor.
        host = self.hosts.get(fb.host, [])
        previous = bisect_left(host, i) - 1
        if previous >= 0 and host[previous] >= low:
            candidates.add(host[previous])
        if _ERROR.search(b.text):
            collect(self.imports, [b.app])
            for visits in self.apps.values():
                previous = bisect_left(visits, i) - 1
                if previous >= 0 and visits[previous] >= low:
                    candidates.add(visits[previous])
        return sorted(candidates, reverse=True)

    def edges_for(self, target_id: int) -> list[ProvenanceEdge]:
        i = self.positions[target_id]
        ordered = self.ordered
        b, fb = ordered[i], self.features[i]
        low = bisect_left(self.times, b.at - max(0, T.PROVENANCE_WINDOW_S))
        upper = bisect_right(self.times, b.at)
        # Includes equal-time siblings just as an independently loaded window does.
        class Frequency:
            def __getitem__(_, term):
                postings = self.terms.get(term, [])
                return bisect_left(postings, upper) - bisect_left(postings, low)

        df = Frequency()
        window_n = upper - low
        parents = []
        intent_b = fb.intent
        for j in self._candidates(i, low):
            a, fa = ordered[j], self.features[j]
            gap = b.at - a.at
            if a.observation_id == b.observation_id or (a.session_id and b.session_id and a.session_id != b.session_id):
                continue
            same_page = ((a.fields.get("url") and _url_key(a.fields["url"]) == _url_key(b.fields.get("url", ""))) or
                         (a.app == b.app and tk.norm_key(a.title) == tk.norm_key(b.title)))
            proposed: list[ProvenanceEdge] = []

            def add(kind, confidence, qa, qb, proposed=proposed, a=a, b=b, gap=gap):
                if qa and qb:
                    # Caption/referrer assertions are stored evidence, not browser telemetry.
                    if {"caption", "navigated_from"} & {qa[0], qb[0]}:
                        confidence = min(confidence, 0.78)
                    if safe_text(qa[1]) != qa[1] or safe_text(qb[1]) != qb[1]:
                        return
                    proposed.append(ProvenanceEdge(a.sighting_id, b.sighting_id, kind, confidence,
                                                  *qa, *qb, gap))

            # A model referrer must match an earlier page and be corroborated.
            came = b.fields.get("navigated_from", "")
            if came and not same_page:
                for field in ("url", "page_title", "window_title"):
                    value = a.fields.get(field, "")
                    if distinctive(value) and (tk.norm_key(value) in tk.norm_key(came) or
                                               tk.norm_key(came) in tk.norm_key(value)):
                        # Counts alone cannot identify a navigation's payload.
                        interaction = bool((a.clicks or b.clicks) and _NAVIGATION.search(
                            a.fields.get("caption", "")) and find_quote(
                                a, b.fields.get("url", "") or b.title, fields=("caption",)))
                        corroborated = bool(interaction or url_quote(a, b.fields.get("url", "")) or
                                            find_quote(a, b.title, fields=("screen_text",)) or
                                            any(_url_key(ref) == fa.url_key for key, refs in
                                                parse_qs(urlsplit(b.fields.get("url", "")).query).items()
                                                if key in ("ref", "referrer", "from") for ref in refs))
                        add("navigated_from", 0.78 if corroborated else 0.35,
                            (field, value), ("navigated_from", came))
                        break

            # A copy must have a payload match; the clipboard counter alone is insufficient.
            if fa.copied:
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
                        add("search" if fa.intent else "message_share" if fa.shared else "navigation",
                            0.9, qa, qb)
                        break

                # Bare host succession is contextual; a visible path label can
                # independently establish navigation. Both endpoints stay quoted.
                ua, ub = a.fields.get("url", ""), b.fields.get("url", "")
                if ua and ub and gap <= 90 and fa.host == fb.host and (
                    fa.host and fa.url_key != fb.url_key and not fa.intent):
                    add("same_site_context", 0.3, ("url", ua), ("url", ub))
                    label = unquote(urlsplit(ub).path.rstrip("/").rsplit("/", 1)[-1]).replace("-", " ")
                    if len(label) >= 4 and not label.isdigit() and label.casefold() not in ("home", "index", "profile"):
                        add("navigation", 0.87, find_quote(a, label, fields=("screen_text",)), ("url", ub))

            intent_a = fa.intent
            if intent_a and not same_page:
                terms = fa.intent_terms
                matched = terms & fb.identity_terms
                if matched:
                    term = min(matched, key=lambda t: df[t])
                    add("search", 0.86, intent_a[1], find_quote(b, term))

            # Product/brand named inside embedded media, then searched; also supports
            # a typed destination with shared subject terms (weaker evidence).
            common = fb.destination_terms & fa.body_terms
            rare = {t for t in common if df[t] <= max(3, window_n * 0.15)}
            if rare:
                term = min(rare, key=lambda t: (df[t], -len(t), t))
                qa, qb = find_quote(a, term), find_quote(b, term)
                if _MEDIA.search(a.fields.get("caption", "")) and intent_b:
                    add("media_mention", 0.76, qa, qb)
                elif fa.shared and not same_page:
                    add("message_share", 0.72, qa, qb)
                elif a.app != b.app:
                    add("app_switch", 0.58, qa, qb)

            # An omnibox entry can have a different name from the motivating post
            # (a model announcement → a model registry). Preserve the current topic
            # only when the destination's stored caption explicitly shows that topic
            # alongside its typed intent, and it was also present on the source.
            if intent_b and b.keys and not intent_a:
                context_terms = fb.caption_terms & fa.context_terms
                context_terms -= fb.intent_distinctive
                contextual = {t for t in context_terms if df[t] <= max(3, window_n * 0.15) or len(t) >= 8}
                if contextual and fa.shared:
                    term = min(contextual, key=lambda t: (df[t], -len(t), t))
                    add("typed_navigation", 0.68, find_quote(a, term), find_quote(b, term, fields=("caption",)))

            # State explanation: recent foreground evidence is only *possibly* still
            # running. Require its app to be named by an import/setup screen, or by
            # the dialog itself. Never assert actual background process liveness.
            if _ERROR.search(b.text) and a.app and a.app != b.app:
                setup = next((ordered[j] for j in reversed(self.imports.get(b.app, []))
                              if low <= j < i and find_quote(ordered[j], a.app)), None)
                if find_quote(b, a.app) or setup:
                    add("state_context", 0.6, ("app", a.app), find_quote(b, _ERROR.search(b.text)[0]))
            elif _ERROR.search(b.text) and a.app == b.app and re.search(r"\bimport", a.text, re.I):
                add("state_selection", 0.7, find_quote(a, "import"), find_quote(b, _ERROR.search(b.text)[0]))

            if proposed:
                best = max(proposed, key=lambda e: e.confidence)
                if valid_edge(best, self.raw_by_id):
                    parents.append(best)
        # Preserve alternative origins, especially two copied URLs; at most one
        # parent of each kind/page except clipboard sources with separate payloads.
        parents.sort(key=lambda e: (-e.confidence, e.gap_s, -e.source))
        unique = []
        seen = set()
        by_id = self.by_id
        for edge in parents:
            source = by_id[edge.source]
            identity = (edge.kind, source.app) if edge.kind == "state_context" else (edge.kind, source.observation_id)
            if identity not in seen:
                unique.append(edge)
                seen.add(identity)
        return unique[:max(0, T.PROVENANCE_MAX_PARENTS)]


def infer_edges(moments: list[AttentionMoment], *, target_id: int | None = None,
                target_ids: set[int] | None = None, inference: EdgeInference | None = None) -> list[ProvenanceEdge]:
    inference = inference or EdgeInference(moments)
    ids = {target_id} if target_id is not None else target_ids
    return [edge for m in inference.ordered if ids is None or m.sighting_id in ids
            for edge in inference.edges_for(m.sighting_id)]


def rules_key() -> str:
    return f"v2:{T.PROVENANCE_WINDOW_S}:{T.PROVENANCE_MAX_PARENTS}"


async def build_provenance(store: Store, *, backfill: bool = False) -> int:
    """Incremental and restartable, also usable without a configured model."""
    if not T.PROVENANCE:
        return 0
    if backfill:
        await store.reset_provenance()
    batch_size = max(1, T.PROVENANCE_BATCH)
    pending = await store.pending_provenance(rules_key(), limit=-1 if backfill else batch_size,
                                             window_s=max(0, T.PROVENANCE_WINDOW_S))
    if not pending:
        return 0
    snapshot_at = now()
    rows = await store.attention_moments(pending[0].at - max(0, T.PROVENANCE_WINDOW_S), pending[-1].at)
    inference = EdgeInference(rows)
    made = 0
    for offset in range(0, len(pending), batch_size):
        batches = []
        for moment in pending[offset:offset + batch_size]:
            feature = inference.features[inference.positions[moment.sighting_id]]
            checkpoint_at = max(snapshot_at, feature.updated_window)
            batches.append((moment, inference.edges_for(moment.sighting_id), checkpoint_at))
        made += await store.save_provenance_batch(batches, rules_key())
    return made
