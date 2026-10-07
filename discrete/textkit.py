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
import tldextract
from datasketch import MinHash

from .perf import PERF

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
    if len(text) > n:
        # Count actual cuts by their limit. In particular, the caption's 4000
        # character cut is observable without coupling utilities to enrichment.
        PERF.count(f"health.text_truncated.{n}")
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


def content_hash(text: str) -> str:
    return hashlib.sha1(norm_key(text).encode("utf-8")).hexdigest()


def stable_obs_key(*parts: str) -> str:
    """Dedup key from stable fields only (planS2 §2). Caption excluded."""
    return norm_key("\n".join(p for p in parts if p))


def minhash(text: str, *, permutations: int | None = None) -> np.ndarray:
    """MinHash signature as its raw uint64 hash values (datasketch's fixed seed,
    so signatures from different runs compare)."""
    m = MinHash(num_perm=permutations or 128)
    xs = word_shingles(text, 3)
    if len(xs) < 30:
        xs = xs | {t for t in tokens(text) if len(t) >= 2}
    for sh in xs:
        m.update(sh.encode("utf-8"))
    return m.hashvalues.astype(np.uint64)


def minhash_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or not len(a):
        return 0.0
    return float(np.mean(a == b))


def minhash_bytes(sig: np.ndarray) -> bytes:
    return np.asarray(sig, dtype="<u8").tobytes()


def minhash_from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<u8")


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


def looks_like_domain(text: str) -> bool:
    return bool(_DOMAINISH.match(norm_key(text)))


def entity_key(name: str) -> str:
    """Normalized identity key (planS2 §10.4 step 2): web addresses reduce to the
    registrable domain's label, file extensions and generic suffixes drop, then
    only alphanumerics remain."""
    s = norm_key(name).strip(_EDGE + " \"()[]{}")
    if _DOMAINISH.match(s):
        try:
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
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass
    return ""


_CHROME_MAX_WORDS = 3


def chrome_lines(texts: list[str], *, share: float, min_texts: int) -> set[str]:
    """Normalized label-like lines recurring in more than `share` of `texts` (captures of
    one app and site): navigation and toolbar labels. Fewer than `min_texts` captures say
    nothing reliable, so then nothing is chrome. Recurrence alone is not chrome: content
    that stays on screen (a chat transcript, a sentence, a total) is longer or
    carries digits, so it stays. Callers drop chrome only where a repeat adds nothing
    (the drafter and the caption prompt), never from what is stored."""
    if len(texts) < min_texts:
        return set()
    counts: dict[str, int] = {}
    for text in texts:
        for key in {norm_key(line) for line in (text or "").splitlines() if line.strip()}:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n / len(texts) > share and _label_like(k)}


def _label_like(key: str) -> bool:
    return len(key.split()) <= _CHROME_MAX_WORDS and not any(c.isdigit() for c in key)


def without_lines(text: str, drop: set[str]) -> str:
    if not drop:
        return text
    return "\n".join(line for line in (text or "").splitlines() if norm_key(line) not in drop)
