"""DISCRETE_DENIED_APPS / DISCRETE_DENIED_SITES: a denied window leaves nothing behind."""

from __future__ import annotations

import asyncio
import time

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import privacy
from discrete.capture import screen as sc


def _deny(monkeypatch, apps=(), sites=()):
    monkeypatch.setattr(privacy, "_APPS", frozenset(a.lower() for a in (*privacy.DEFAULT_DENIED_APPS, *apps)))
    monkeypatch.setattr(privacy, "_SITES", tuple(sites))


def test_password_managers_are_never_watched(monkeypatch):
    _deny(monkeypatch)
    assert privacy.denied("1Password") == "app"
    assert privacy.denied("Some Vault", bundle_id="com.bitwarden.desktop") == "app"
    assert privacy.denied("Safari", url="https://example.org") == ""


def test_apps_match_by_name_or_bundle_id_whole_and_case_insensitive(monkeypatch):
    _deny(monkeypatch, apps=["WhatsApp", "com.t3tools.t3code"])
    assert privacy.denied("whatsapp") == "app"
    assert privacy.denied("T3 Code (Alpha)", bundle_id="com.t3tools.t3code") == "app"
    # A whole name, not a fragment: "Notes" does not deny "Sticky Notes".
    _deny(monkeypatch, apps=["Notes"])
    assert privacy.denied("Sticky Notes") == ""


def test_a_site_covers_its_subdomains_and_a_subdomain_only_itself(monkeypatch):
    _deny(monkeypatch, sites=["hdfcbank.com", "mail.google.com"])
    assert privacy.denied("Arc", url="https://netbanking.hdfcbank.com/login") == "site"
    assert privacy.denied("Arc", url="https://hdfcbank.com/") == "site"
    assert privacy.denied("Arc", url="https://mail.google.com/mail/u/0/") == "site"
    assert privacy.denied("Arc", url="https://docs.google.com/document/d/1") == ""
    assert privacy.denied("Arc", url="https://nothdfcbank.com/") == ""


async def test_a_denied_window_is_not_read_saved_or_queued(store, fake, monkeypatch):
    _deny(monkeypatch, sites=["hdfcbank.com"])
    q: asyncio.Queue = asyncio.Queue()
    obs = sc.ScreenObserver(queue=q)
    read: list[str] = []
    monkeypatch.setattr(obs, "_screen_text", lambda ctx, after, stale=False: read.append("x") or ("text", "ax"))
    ctx = ax.WindowContext(app="Arc", bundle_id="company.thebrowser.Browser", window_title="NetBanking",
                           url="https://netbanking.hdfcbank.com/", pid=7)
    pending = {"kind": "click", "started": time.time(), "counts": {"keys": 0, "clicks": 1, "scrolls": 0, "clipboard": 0}}

    await obs._capture(pending, ctx, None, Image.new("RGB", (64, 64)))

    assert q.empty() and read == []
    assert obs.stats["denied"] == 1
