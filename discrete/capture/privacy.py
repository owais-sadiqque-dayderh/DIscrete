"""Apps and sites DIscrete never watches (DISCRETE_DENIED_APPS, DISCRETE_DENIED_SITES).

Checked before any screen text is read or any frame is saved, so a denied window
leaves nothing behind: no row, no screenshot, no model call. Bubble also refuses to
read a text field in one. Unlike SELF_APPS (the system's own windows, still stored
for audit), a denied window is not stored at all.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from ..config import T

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
_SITES = tuple(s.lower().removeprefix("www.").strip(".") for s in T.DENIED_SITES)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".").removeprefix("www.").encode("idna").decode("ascii")
    except (ValueError, UnicodeError):
        return ""


def denied(app: str = "", bundle_id: str = "", url: str = "", *, window_title: str = "") -> str:
    """Why this window must not be watched ("app" or "site"), or "" when it may be.

    An app matches by its name or bundle id (case-insensitive, whole value). A site
    Unknown browser URLs and recognizable private-window titles are excluded.
    Title detection covers English incognito/private markers only; it cannot prove
    that an unmarked window is public. Browser/version validation remains necessary.
    A site matches its host and every subdomain: "google.com" covers mail.google.com, while
    "mail.google.com" covers only Gmail.
    """
    if (app or "").lower() in _APPS or (bundle_id or "").lower() in _APPS:
        return "app"
    browser = (app or "").lower() in {
        "arc", "google chrome", "chrome", "safari", "firefox", "microsoft edge", "brave browser",
    } or (bundle_id or "").lower() in {
        "company.thebrowser.browser", "com.google.chrome", "com.apple.safari", "org.mozilla.firefox",
    }
    # Simple detectable-title policy, not a claim of universal private-mode detection.
    if browser and any(marker in window_title.lower() for marker in (
        "incognito", "private browsing", "private window", "inprivate",
    )):
        return "private"
    host = _host(url)
    if browser and not host:
        return "unknown_url"
    if host and any(host == _host("https://" + site) or host.endswith("." + _host("https://" + site)) for site in _SITES):
        return "site"
    return ""
