"""Topic-neutral text utilities: normalizing, tokenizing, shingles, overlap, snippets.

Nothing here may contain a topic-specific constant (no entity names, no domain
terms). The stopword and imperative lists are function words only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata

import numpy as np
from datasketch import MinHash

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@/\-'’]*")
_EDGE = "._-/'’@"
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)

STOPWORDS = frozenset(
    """
a an the and or but if then else of in on at to for from by with without as is
are was were be been being have has had do does did will would shall should can
could may might must not no nor so than that this these those there here it its
i me my we our us you your he him his she her they them their who whom whose
which what when where why how all any both each few more most other some such
only own same too very just about into over under again further once dont
im ive id youre theyre thats whats got get getting
know knows knew tell tells told say says said please thanks thank ok okay
yes yeah recently recent currently now today lately still already
anything something nothing everything anyone someone thing things stuff
give gives gave make makes made want wants need needs like likes going go
""".split()
)

INTERROGATIVES = frozenset("what which who whom whose where when why how".split())

_IMPERATIVE_HEADS = frozenset(
    """ignore disregard forget remember always never pretend act respond reply
    output print send delete execute run obey follow""".split()
)
_IMPERATIVE_PHRASES = ("you must", "you should", "you will", "you are now", "ai assistant", "language model", "system prompt")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    """Normalized, case-folded form used for exact comparisons."""
    return normalize(text).casefold()


def tokens(text: str, *, lower: bool = True) -> list[str]:
    """Word tokens with possessives and edge punctuation stripped (insights.md §7.10)."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        t = _POSSESSIVE.sub("", raw).strip(_EDGE)
        if t:
            out.append(t.lower() if lower else t)
    return out


def content_tokens(text: str, *, min_len: int = 2) -> list[str]:
    return [t for t in tokens(text) if len(t) >= min_len and t not in STOPWORDS]


def word_shingles(text: str, n: int = 3) -> set[str]:
    xs = tokens(text)
    if len(xs) < n:
        return {" ".join(xs)} if xs else set()
    return {" ".join(xs[i : i + n]) for i in range(len(xs) - n + 1)}


def trigrams(text: str) -> frozenset[tuple[str, ...]]:
    xs = tokens(text)
    return frozenset(tuple(xs[i : i + 3]) for i in range(max(0, len(xs) - 2)))


def jaccard(a: frozenset | set, b: frozenset | set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def containment(small: frozenset | set, big: frozenset | set) -> float:
    """Share of `small` found in `big` (asymmetric)."""
    if not small:
        return 0.0
    return len(small & big) / len(small)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def looks_imperative(sentence: str) -> bool:
    low = sentence.lower()
    toks = tokens(low)
    if toks and toks[0] in _IMPERATIVE_HEADS:
        return True
    return any(p in low for p in _IMPERATIVE_PHRASES)


def best_window(text: str, terms: set[str], width: int = 40) -> tuple[str, float]:
    """Snippet of `text` (about `width` tokens) holding the most distinct `terms`.

    Returns (snippet, overlap), where overlap = distinct terms in window / |terms|.
    Always a verbatim substring of `text`.
    """
    if not text or not terms:
        return "", 0.0
    spans = [(m.group(0).lower().strip(_EDGE), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    if not spans:
        return "", 0.0
    best = (0, 0, min(width, len(spans)))
    for i in range(0, max(1, len(spans) - width + 1)):
        window = spans[i : i + width]
        hit = len({t for t, _, _ in window if t in terms})
        if hit > best[0]:
            best = (hit, i, i + len(window))
    hit, i, j = best
    j = max(j, i + 1)
    snippet = text[spans[i][1] : spans[min(j, len(spans)) - 1][2]]
    return snippet, hit / len(terms)


def truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


def content_hash(text: str) -> str:
    return hashlib.sha1(norm_key(text).encode("utf-8")).hexdigest()


def stable_obs_key(*parts: str) -> str:
    """Dedup key from stable fields only (planS2 §2). Caption excluded."""
    return norm_key("\n".join(p for p in parts if p))


def minhash(text: str, *, permutations: int | None = None) -> MinHash:
    permutations = permutations or 128
    m = MinHash(num_perm=permutations)
    xs = word_shingles(text, 3)
    if len(xs) < 30:
        xs = xs | {t for t in tokens(text) if len(t) >= 2}
    for sh in xs:
        m.update(sh.encode("utf-8"))
    return m


def minhash_jaccard(a: MinHash, b: MinHash) -> float:
    return float(a.jaccard(b))


def bm25_idf(df: int, n_docs: int) -> float:
    if n_docs <= 0 or df <= 0:
        return 0.0
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def cosine(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rrf(rankings: list[list[int]], *, k: float = 60.0) -> dict[int, float]:
    out: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            out[doc_id] = out.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return out


def fts_query(terms: list[str], *, mode: str = "OR") -> str:
    safe = []
    for t in terms:
        t = re.sub(r"[^\w\s.@/'-]", " ", t).strip()
        if not t:
            continue
        safe.append('"' + t.replace('"', "") + '"' if " " in t else f'"{t}"')
    if not safe:
        return ""
    joiner = " OR " if mode.upper() == "OR" else " AND "
    return joiner.join(safe)


def parse_json_list(raw: str) -> list:
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


_DOMAINISH = re.compile(r"^(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/:?#].*)?$")
_FILE_EXT = re.compile(r"\.(?:py|js|ts|tsx|jsx|md|json|ya?ml|txt|csv|pdf|png|jpe?g|svg|ai|psd|fig)$")
_GENERIC_SUFFIXES = frozenset("font fonts app inc ltd llc".split())


def entity_key(name: str) -> str:
    """Normalized identity key (planS2 §10.4 step 2): web addresses reduce to the
    registrable domain's label, file extensions and generic suffixes drop, then
    only alphanumerics remain."""
    s = norm_key(name).strip(_EDGE + " \"()[]{}")
    if _DOMAINISH.match(s):
        try:
            import tldextract

            label = tldextract.extract(s).domain
            if label:
                s = label
        except Exception:
            pass
    s = _FILE_EXT.sub("", s)
    parts = re.findall(r"[a-z0-9]+", s)
    while len(parts) > 1 and parts[-1] in _GENERIC_SUFFIXES:
        parts.pop()
    return "".join(parts)


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Normalized, word-bounded containment."""
    p = norm_key(phrase)
    if not p:
        return False
    return re.search(r"(?<![\w])" + re.escape(p) + r"(?![\w])", norm_key(haystack)) is not None


def site_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        import tldextract

        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    return ""


_GENERIC_TITLE = re.compile(
    r"^(?:space\s+\d+|untitled(?:\s+document)?|new\s+tab|document|window)$", re.I
)
_LISTING_URL = re.compile(r"news\.ycombinator\.com/(?:news|newest|shownew|ask|show|jobs)(?:\?|$)", re.I)
_NAV_CAPTION = re.compile(
    r"\b(?:switched from|clicked on a link|clicked on a specific item|opened a new tab|navigated to|"
    r"reading an article on|viewing the github repository|github repository page for|focused on the)\b",
    re.I,
)
_DEST_CAPTION = re.compile(
    r"(?:switched from .+? to (?:the )?|clicked on a link to (?:the )?|reading an article on (?:the )?|"
    r"github repository page for ['\"]?|viewing the github repository for ['\"]?|post regarding )"
    r"(.+?)(?:\s+website|\s+page|\s+about|\s+project|\s+repository|,|\.|$)",
    re.I,
)
_HN_ITEM = re.compile(r"\d+\.\s*(?:[▲A►●4]\s*)?(.+?)\s+\([\w.-]+\)", re.I)


def is_generic_title(title: str) -> bool:
    t = normalize(title)
    return not t or bool(_GENERIC_TITLE.match(t))


def is_listing_url(url: str) -> bool:
    return bool(url and _LISTING_URL.search(url))


def navigation_caption(caption: str) -> bool:
    return bool(caption and _NAV_CAPTION.search(caption))


def destination_from_caption(caption: str) -> str:
    if not caption:
        return ""
    m = re.search(r"github repository page for ['\"]?([^'\".,]+)", caption, re.I)
    if m:
        return f"{normalize(m.group(1))} (GitHub)"[:120]
    m = re.search(r"repository (?:page )?for ['\"]?([^'\".,]+)", caption, re.I)
    if m:
        return f"{normalize(m.group(1))} (GitHub)"[:120]
    m = re.search(r"post regarding (.+?)(?:\.|$)", caption, re.I)
    if m:
        return normalize(m.group(1))[:120]
    m = re.search(r"article on (?:the )?(.+?) website", caption, re.I)
    if m:
        return normalize(m.group(1))[:120]
    m = _DEST_CAPTION.search(caption)
    if m:
        dest = normalize(m.group(1).strip("'\""))
        if dest and not dest.lower().startswith(("the ", "a ", "an ")):
            return dest[:120]
    return ""


def hn_headlines(screen_text: str, *, limit: int = 12) -> list[str]:
    """Article titles visible on a Hacker News listing page (OCR-tolerant)."""
    if not screen_text:
        return []
    out: list[str] = []
    for m in _HN_ITEM.finditer(screen_text):
        title = normalize(m.group(1))
        if len(title) >= 8 and title not in out:
            out.append(title)
        if len(out) >= limit:
            break
    return out


def page_label(title: str, *, url: str = "", caption: str = "", screen_text: str = "", site: str = "") -> str:
    """Human-readable page name when the window title is generic (Arc 'Space 1', etc.)."""
    dest = destination_from_caption(caption)
    if dest:
        return dest
    if not is_generic_title(title):
        return normalize(title)
    if url and not is_listing_url(url):
        try:
            import tldextract

            ext = tldextract.extract(url)
            if ext.domain:
                host = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
                path = (url.split("?", 1)[0].split("#", 1)[0].rstrip("/").split("/")[-1] or "").replace("-", " ")
                if path and path not in ("news", "index", "html"):
                    return f"{normalize(path)} ({host})"
                return host
        except Exception:
            pass
    heads = hn_headlines(screen_text, limit=1)
    if heads:
        return heads[0]
    if site:
        return site
    return normalize(title) or "page"


def page_identity(title: str, *, url: str = "", caption: str = "", screen_text: str = "") -> str:
    """Dedup key for enumerate: URL when known, else a stable label — never bare 'Space 1'."""
    u = url.split("#")[0].rstrip("/") if url else ""
    if u and not is_listing_url(u):
        return u
    label = page_label(title, url=url, caption=caption, screen_text=screen_text)
    return f"{label}\x00{hashlib.sha1(normalize(caption)[:200].encode()).hexdigest()[:10]}"
