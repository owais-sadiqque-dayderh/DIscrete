"""Page identity and labels for enumeration ("list the pages I opened").

Built only from ground-truth fields (window title, URL, site) and the structured
page fields the vision call returns (page_title, is_listing, navigated_from).
Nothing here parses caption prose or knows any particular site: a change of
vision model or prompt wording cannot silently break it.
"""

from __future__ import annotations

import hashlib
import re

import tldextract

from .. import textkit as tk
from ..page_identity import canonical_url

# Window chrome that names no page (browser Spaces, blank documents).
_GENERIC_TITLE = re.compile(r"^(?:space\s+\d+|untitled(?:\s+document)?|new\s+tab|document|window)$", re.I)
_EMPTY_PATH = frozenset({"", "index", "index.html", "html", "home"})


def is_generic_title(title: str) -> bool:
    t = tk.normalize(title)
    return not t or bool(_GENERIC_TITLE.match(t))


def canon_url(url: str) -> str:
    return canonical_url(url)


def url_label(url: str) -> str:
    """'last path segment (host)', or the host alone."""
    try:
        ext = tldextract.extract(url)
    except Exception:
        return ""
    if not ext.domain:
        return ""
    host = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/").split("/")
    last = path[-1].replace("-", " ") if len(path) > 3 else ""
    return f"{tk.normalize(last)} ({host})" if last.lower() not in _EMPTY_PATH else host


def page_label(title: str, *, page_title: str = "", url: str = "", site: str = "") -> str:
    """Human-readable page name; the vision call's page title wins over window chrome."""
    if page_title.strip():
        return tk.normalize(page_title)
    if not is_generic_title(title):
        return tk.normalize(title)
    if url and (label := url_label(url)):
        return label
    return site or tk.normalize(title) or "page"


def page_identity(title: str, *, page_title: str = "", url: str = "", caption: str = "") -> str:
    """Dedup key: the URL when known, else the page's name. A generic window title
    with nothing better is never an identity on its own ('Space 1' is every page)."""
    if u := canon_url(url):
        return u
    label = page_label(title, page_title=page_title)
    if page_title.strip() or not is_generic_title(title):
        return tk.norm_key(label)
    return f"{tk.norm_key(label)}\x00{hashlib.sha1(tk.normalize(caption)[:200].encode()).hexdigest()[:10]}"


def names_overlap(a: str, b: str, *, threshold: float = 0.5) -> bool:
    """Do two page names share most of the shorter one's content words?"""
    ta, tb = set(tk.content_tokens(a)), set(tk.content_tokens(b))
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold
