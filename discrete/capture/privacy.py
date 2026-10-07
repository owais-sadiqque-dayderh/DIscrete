"""Apps and sites DIscrete never watches (DISCRETE_DENIED_APPS, DISCRETE_DENIED_SITES).

Checked before any screen text is read or any frame is saved, so a denied window
leaves nothing behind: no row, no screenshot, no model call. Bubble also refuses to
read a text field in one. Unlike SELF_APPS (the system's own windows, still stored
for audit), a denied window is not stored at all.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from ..config import T
from .browser import private_window

#: Never watched whatever .env says: their whole window is secrets. DENIED_APPS adds
#: to this list; it cannot remove from it.
DEFAULT_DENIED_APPS: tuple[str, ...] = (
    "1Password",
    "1Password 7",
    "Bitwarden",
    "Keychain Access",
    "Passwords",
    "LastPass",
    "Dashlane",
    "KeePassXC",
    "Enpass",
    "com.1password.1password",
    "com.agilebits.onepassword7",
    "com.bitwarden.desktop",
    "com.apple.keychainaccess",
    "com.apple.Passwords",
    "com.lastpass.LastPass",
    "com.dashlane.dashlanephonefinal",
    "org.keepassxc.keepassxc",
    "in.sinew.Enpass-Desktop",
)

_APPS = frozenset(a.lower() for a in (*DEFAULT_DENIED_APPS, *T.DENIED_APPS))


def _host(url: str) -> str:
    """Canonical host: lower case, no trailing dot, no "www.", IDNA-encoded."""
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".").removeprefix("www.").encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        return ""


_SITES = tuple(filter(None, (_host("https://" + s.strip().strip(".")) for s in T.DENIED_SITES)))
#: Browsers whose tab URL decides a site match. Without a URL, a denied site
#: cannot be ruled out.
_BROWSERS = frozenset((
    "safari", "google chrome", "chromium", "arc", "microsoft edge", "brave browser", "firefox", "opera",
    "vivaldi", "orion", "com.apple.safari", "com.google.chrome", "org.chromium.chromium",
    "company.thebrowser.browser", "com.microsoft.edgemac", "com.brave.browser", "org.mozilla.firefox",
    "com.operasoftware.opera", "com.vivaldi.vivaldi", "com.kagi.kagimacos",
))
# English title markers of private windows (Firefox, Edge, some Arc/Chrome builds).
_PRIVATE_TITLES = ("incognito", "private browsing", "inprivate")


def denied(app: str = "", bundle_id: str = "", url: str = "", *, window_title: str = "", window_id: int = 0) -> str:
    """Why this window must not be watched ("app", "private", "unknown_url" or "site"), or ""
    when it may be.

    An app matches by its name or bundle id (case-insensitive, whole value). A site
    matches its host and every subdomain: "google.com" covers mail.google.com, while
    "mail.google.com" covers only Gmail. Hosts compare case-insensitively, without a
    trailing dot or "www.". When sites are denied, a browser window whose web URL is
    unknown is excluded too. A private browser window is excluded when detectable:
    Chromium-family incognito via Apple Events (browser.py), else a title marker; an
    unmarked Safari or Arc private window cannot be told apart.
    """
    if (app or "").lower() in _APPS or (bundle_id or "").lower() in _APPS:
        return "app"
    host = _host(url)
    if (app or "").lower() in _BROWSERS or (bundle_id or "").lower() in _BROWSERS:
        if (window_id and private_window(window_id)) or any(m in window_title.lower() for m in _PRIVATE_TITLES):
            return "private"
        # An internal page (about:, file:, chrome:) is no site; a missing or unparsable web URL may be one.
        if _SITES and not host and (not url or url.lower().startswith("http")):
            return "unknown_url"
    if host and any(host == site or host.endswith("." + site) for site in _SITES):
        return "site"
    return ""
