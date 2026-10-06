"""Secret redaction: credentials never reach the store or a model.

Ported from the Hindsight activity agent (closed Worklog, Phase 18), where a key
visible in a terminal was stored as memory evidence and sent to the LLM provider.
Screen text, window titles and URLs pass through :func:`redact` in
``ScreenObserver._capture`` before anything is stored; captions coming back from the
vision model and Bubble's field text are redacted too.

The patterns favour precision: each matches a credential *format* (a known
prefix, a PEM block, a JWT's three base64url segments) or a value assigned to a
credential-named key. Ordinary prose and identifiers stay readable, because the
memory is only useful if what the user did survives.

Text redaction cannot clean a frame, so a capture where something was redacted keeps
no frame and its caption is written from the (redacted) text alone.

OCR reads the same screen as the accessibility tree, but not exactly: it splits
``sk-proj-abc…`` at a space, or shows the tail of a password peeking out from
behind a dialog. Those copies no longer match any credential format.
:class:`SecretMemory` closes that gap: every redacted value is remembered (as
hashes of its fragments, never the value itself) for a while, and any word that
shares a fragment with one is redacted too, in this capture and later ones.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # PEM private keys, whole block.
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.DOTALL),
    # Vendor tokens with a recognisable prefix.
    re.compile(r"\b(?:sk|pk|rk)-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI, Anthropic, Stripe-style
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),  # Stripe
    re.compile(r"\bhsk_[A-Za-z0-9_\-]{16,}"),  # Hindsight
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),  # GitLab
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),  # Slack
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),  # AWS access key id
    re.compile(r"\bAIza[A-Za-z0-9_\-]{35}"),  # Google API key
    re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}"),  # Google OAuth access token
    re.compile(r"\bnpm_[A-Za-z0-9]{36}"),
    re.compile(r"\bpypi-[A-Za-z0-9_\-]{50,}"),
    # Hugging Face; OCR has been seen to read a character of one as "/".
    re.compile(r"\bhf_[A-Za-z0-9/]{20,}"),
    # JWTs: header.payload.signature, header always starts with {"  -> eyJ.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # Credentials embedded in a URL: scheme://user:password@host
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
)

# Key material no pattern names, or one OCR garbled past recognition ("şk=Xe-60a9…"
# for an OpenRouter key): one 32+ character run mixing 8+ digits with 8+ letters.
# No "/" or ".", so URL paths and file names stay; UUIDs and commit hashes go too.
# Applied after the named patterns, so a JWT or key they know is removed whole.
_KEY_MATERIAL = re.compile(
    r"(?<![A-Za-z0-9_\-+=])(?=(?:[A-Za-z_\-+=]*\d){8})(?=(?:[0-9_\-+=]*[A-Za-z]){8})[A-Za-z0-9_\-+=]{32,}"
)

# key = value / key: value / "key": "value", where the key names a credential.
# The value is replaced, the key kept, so "the user set an API key" survives. OCR can
# split a value at a space ("sk-proj- FAKE…"); a following chunk that looks like key
# material (12+ characters mixing letters and digits) still belongs to the value.
_SECRET_KEY = (
    # Optional "OPENAI_" / "db-" style prefixes. The separator is required: an optional
    # one makes this a nested quantifier that backtracks exponentially on long words.
    r"(?:[A-Za-z0-9]+[_\-])*"
    r"(?:passw(?:or)?d|passphrase|secret|token|api[_\-]?key|apikey|access[_\-]?key|private[_\-]?key"
    r"|client[_\-]?secret|auth)"
)
_CONTINUATION = r"(?:[ \t](?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{12,})*"
_ASSIGNMENT = re.compile(
    rf"(?i)(?P<key>[\"']?\b{_SECRET_KEY}\b[\"']?\s*[:=]\s*)(?P<quote>[\"']?)(?!\[REDACTED\])(?P<value>[^\s\"',;]{{6,}}{_CONTINUATION})(?P=quote)"
)
# Authorization: Bearer <token> / Basic <base64>
_AUTH_HEADER = re.compile(r"(?i)\b(?P<scheme>bearer|basic)\s+(?P<value>[A-Za-z0-9._~+/\-]{16,}=*)")

_SECRET_QUERY_KEY = re.compile(rf"(?i)^{_SECRET_KEY}$|^(?:sig|signature|code|key|session|sid|jwt)$")


@dataclass(frozen=True)
class Redaction:
    text: str
    count: int
    #: The values that were removed, for :meth:`SecretMemory.remember`. In memory only.
    secrets: tuple[str, ...] = ()


def redact(text: str) -> Redaction:
    """``text`` with every recognised secret replaced by ``[REDACTED]``."""
    if not text:
        return Redaction(text, 0)
    count = 0
    secrets: list[str] = []

    def _sub(pattern: re.Pattern[str], replace, value: str, group: int | str = 0) -> str:  # noqa: ANN001
        nonlocal count

        def _one(match: re.Match[str]) -> str:
            secrets.append(match.group(group))
            return replace(match) if callable(replace) else replace

        value, n = pattern.subn(_one, value)
        count += n
        return value

    for pattern in (*_TOKEN_PATTERNS, _KEY_MATERIAL):
        text = _sub(pattern, REDACTED, text)
    text = _sub(_AUTH_HEADER, lambda m: f"{m.group('scheme')} {REDACTED}", text, "value")
    text = _sub(_ASSIGNMENT, lambda m: f"{m.group('key')}{m.group('quote')}{REDACTED}{m.group('quote')}", text, "value")
    return Redaction(text, count, tuple(secrets))


def redact_url(url: str) -> Redaction:
    """``url`` with credential-like query values and embedded user:password removed.

    The path and the other parameters stay: which page the user was on is the point.
    """
    if not url:
        return Redaction(url, 0)
    try:
        parts = urlsplit(url)
        query = parse_qsl(parts.query, keep_blank_values=True)
    except ValueError:
        return redact(url)
    count = 0
    secrets: list[str] = []
    cleaned: list[tuple[str, str]] = []
    for key, value in query:
        if value and _SECRET_QUERY_KEY.search(key):
            cleaned.append((key, REDACTED))
            secrets.append(value)
            count += 1
        else:
            cleaned.append((key, value))
    netloc = parts.netloc
    if "@" in netloc and ":" in netloc.split("@", 1)[0]:
        secrets.append(netloc.split("@", 1)[0].split(":", 1)[1])
        netloc = f"{REDACTED}@{netloc.split('@', 1)[1]}"
        count += 1
    rebuilt = urlunsplit(
        (parts.scheme, netloc, parts.path, urlencode(cleaned, safe="[]") if count else parts.query, parts.fragment)
    )
    rest = redact(rebuilt)
    return Redaction(rest.text, count + rest.count, (*secrets, *rest.secrets))


# A fragment is this many letters/digits of a secret, lower-cased. Long enough that a
# fragment of "hunter2secret99" does not match the word "secret" in prose, short
# enough that an OCR copy with a few misread characters still shares one.
_FRAGMENT = 8
_WORD = re.compile(r"[^\s\"'`,;()<>\[\]{}|]+")


def _normalise(value: str) -> str:
    return "".join(c for c in value.lower() if c.isalnum())


def _fragment_hash(fragment: str) -> bytes:
    return hashlib.blake2b(fragment.encode(), digest_size=8).digest()


@dataclass
class SecretMemory:
    """Recently redacted secrets, kept as fragment hashes so near-copies are caught.

    Not thread-safe: the observer takes one snapshot at a time.
    """

    ttl_s: float = 1800.0
    clock: Callable[[], float] = time.monotonic
    _expiry: dict[bytes, float] = field(default_factory=dict, init=False, repr=False)

    def remember(self, secrets: tuple[str, ...]) -> None:
        until = self.clock() + self.ttl_s
        for secret in secrets:
            value = _normalise(secret)
            for i in range(len(value) - _FRAGMENT + 1):
                self._expiry[_fragment_hash(value[i : i + _FRAGMENT])] = until

    def scrub(self, text: str) -> Redaction:
        """``text`` with every word that shares a fragment with a remembered secret removed."""
        now = self.clock()
        if self._expiry:
            self._expiry = {h: t for h, t in self._expiry.items() if t > now}
        if not self._expiry or not text:
            return Redaction(text, 0)
        count = 0

        def _word(match: re.Match[str]) -> str:
            nonlocal count
            word = match.group(0)
            value = _normalise(word)
            if word == REDACTED or len(value) < _FRAGMENT:
                return word
            if any(_fragment_hash(value[i : i + _FRAGMENT]) in self._expiry for i in range(len(value) - _FRAGMENT + 1)):
                count += 1
                return REDACTED
            return word

        return Redaction(_WORD.sub(_word, text), count)
