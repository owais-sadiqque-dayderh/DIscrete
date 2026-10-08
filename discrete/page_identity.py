"""Deterministic page/document identities; no caption parsing or model calls."""

from __future__ import annotations

import posixpath
import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

KINDS = ("article", "repo", "video", "product", "chat", "file", "listing", "search", "doc", "other")
_TRACKING = frozenset({"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "si", "ref_src", "ref_url"})
_DOCUMENT = re.compile(r"\.(?:md|txt|rtf|pdf|docx?|xlsx?|pptx?|pages|numbers|key|odt|tex|py|js|tsx?|csv)$", re.I)
_DOCUMENT_APPS = frozenset({"notes", "textedit", "preview", "pages", "numbers", "keynote", "microsoft word",
                            "microsoft excel", "obsidian", "visual studio code", "code", "cursor"})
_GENERIC = re.compile(r"^(?:untitled(?: document)?|space \d+|new tab|window|document)$", re.I)


def canonical_url(url: str) -> str:
    """Keep identity-bearing query params (q, v, id), strip known tracking/fragment.

    Scheme and path remain meaningful: http/https and case-sensitive paths are not
    conflated. File paths are lexical only; we never read or resolve a local file.
    """
    raw = url.strip()
    if not raw:
        return ""
    if raw.startswith(("/", "~/")):
        return "file://" + quote(posixpath.normpath(unquote(raw)), safe="/~")
    try:
        u = urlsplit(raw)
        if u.scheme.lower() == "file":
            if u.hostname not in (None, "", "localhost"):
                return urlunsplit(("file", (u.hostname or "").lower(), quote(unquote(u.path), safe="/"), "", ""))
            return "file://" + quote(posixpath.normpath(unquote(u.path)), safe="/")
        if u.scheme.lower() not in ("http", "https") or not u.hostname:
            return ""
        host = u.hostname.lower().rstrip(".").removeprefix("www.").encode("idna").decode("ascii")
        if ":" in host:
            host = f"[{host}]"
        if u.port and (u.scheme.lower(), u.port) not in (("http", 80), ("https", 443)):
            host += f":{u.port}"
        query = sorted((k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                       if not k.lower().startswith("utm_") and k.lower() not in _TRACKING)
        return urlunsplit((u.scheme.lower(), host, u.path.rstrip("/"), urlencode(query), ""))
    except (ValueError, UnicodeError):
        return ""


def document_identity(app: str, title: str) -> str:
    """Title-only documents are scoped to their app; homonymous files may differ.

    Without AXDocument's path this is an approximate identity, never a filesystem
    assertion. Generic/untitled windows are deliberately not merged into a page.
    """
    title = re.sub(r"\s+[—–-]\s+" + re.escape(app) + r"$", "", title.strip(), flags=re.I)
    title = re.sub(r"\s+[—–-]\s+Edited$", "", title, flags=re.I)
    if not title or title.casefold() == app.casefold() or _GENERIC.fullmatch(title):
        return ""
    if title.startswith(("/", "~/")):
        return canonical_url(title)
    if app.lower() in _DOCUMENT_APPS or _DOCUMENT.search(title):
        return f"doc://{quote(app.lower(), safe='')}/{quote(title, safe='')}"
    if re.fullmatch(r"[^/]+\.[a-zA-Z0-9]{1,8}", title) or title == ".env":
        return f"file-title://{quote(app.lower(), safe='')}/{quote(title, safe='')}"
    return ""


def classify_kind(url: str, *, site: str = "", is_listing: bool = False) -> str:
    u = urlsplit(url)
    host, path = (u.hostname or site).lower(), unquote(u.path).lower()
    parts = [p for p in path.split("/") if p]
    if path == "/search" or (host in {"google.com", "bing.com", "duckduckgo.com"} and "q" in dict(parse_qsl(u.query))):
        return "search"
    if is_listing or (u.scheme in ("doc", "file-title") and re.search(r"\s+[–—-]\s+\d+\s+(?:notes?|items?|files?)$", path)):
        return "listing"
    if u.scheme == "doc":
        return "doc"
    if u.scheme == "file-title":
        return "file"
    if u.scheme == "file":
        return "doc" if _DOCUMENT.search(path) else "file"
    if host in {"github.com", "gitlab.com", "bitbucket.org"}:
        if len(parts) >= 2 and parts[0] not in {"search", "topics", "explore", "settings", "orgs", "users"}:
            return "repo"
        return "listing"
    if host in {"youtube.com", "m.youtube.com", "youtu.be", "vimeo.com"}:
        return "video" if path.startswith(("/watch", "/shorts/")) or (host != "youtube.com" and parts) else "listing"
    if host in {"chatgpt.com", "chat.openai.com", "claude.ai", "web.whatsapp.com", "discord.com", "app.slack.com"}:
        return "chat"
    if host == "docs.google.com" or path.startswith(("/docs/", "/documentation/")) or host.startswith("docs."):
        return "doc"
    if re.search(r"/(?:products?|dp)/[^/]+", path):
        return "product"
    if re.search(r"/(?:articles?|posts?|blog|p)/[^/]+", path) or (host.endswith("substack.com") and parts):
        return "article"
    return "other"
