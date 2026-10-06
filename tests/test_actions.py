"""Quote-backed action and choice candidates, including the Helium misses."""

from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from discrete.process.actions import Detection, detect, hints


def row(oid: int, *, app: str = "Arc", title: str = "", text: str = "",
        caption: str = "", site: str = "", url: str = "") -> dict:
    return {"id": oid, "app": app, "window_title": title, "screen_text": text,
            "caption": caption, "site": site, "url": url, "page_title": ""}


def test_helium_download_install_and_choices():
    rows = [
        row(87, title="Helium · Download", text="Thank you for downloading Helium!\nDownload again"),
        row(89, app="Finder", title="Downloads", caption="The helium_0.18.3.1_arm64-macos.dmg is highlighted.",
            text="Downloads\nDate Added\nhelium_0.18.3.1_arm64-macos.dmg\n113.2 MB\nDisk Image\nToday at 10:21 PM"),
        row(90, app="Finder", title="Helium", text="Helium\nApplications",
            caption="The Helium disk image installer is open, showing the app icon."),
        row(92, app="Helium", title="Helium Setup - Helium", text="* Helium Setup\n* helium://setup"),
        row(93, title="Helium · Download", text="Thank you for downloading Helium!"),
        row(97, app="Helium", title="Helium Setup - Helium",
            text="Choose a search engine\nDuckDuckGo\nKagi\nStartpage\nBrave Search\nEcosia",
            caption="The wizard shows DuckDuckGo selected, and the user is proceeding."),
        row(98, app="Helium", title="Helium Setup - Helium",
            text="Choose a search engine\nEcosia\nQwant\nGoogle\nMicrosoft Bing",
            caption="The user scrolls through options without selecting one."),
        row(109, app="Helium", title="Helium Setup - Helium",
            text="Ready to switch to Helium?\nYes Make Helium my default browser.\nNo Just trying Helium for now.",
            caption="'Yes, Make Helium my default browser' selected."),
    ]
    got = detect(rows)
    assert [(item.observation_id, item.verb, item.subject) for item in got] == [
        (87, "downloaded", "Helium"),
        (89, "downloaded", "helium_0.18.3.1_arm64-macos.dmg"),
        (92, "installed", "Helium"),
        (97, "chose", "DuckDuckGo"),
        (109, "chose", "Helium"),
    ]
    assert got[3].alternatives == ("Kagi", "Startpage", "Brave Search", "Ecosia")
    assert all(item.quote in next(r for r in rows if r["id"] == item.observation_id)["screen_text"]
               or item.quote in next(r for r in rows if r["id"] == item.observation_id)["caption"]
               or item.quote in next(r for r in rows if r["id"] == item.observation_id)["window_title"]
               for item in got)
    assert "obs 87 downloaded Helium" in hints(got)
    assert "verify before using" in hints(got)


@pytest.mark.parametrize("observation", [
    row(1, title="Widget Pro", text="Buy now\n$29.00\nAdd to cart"),
    row(2, title="Shopping cart", text="Widget Pro\nAdd to cart\nProceed to checkout"),
    row(3, app="Mail", title="Draft — hello", text="To: Pat\nHello Pat!\nSend"),
    row(4, title="An article about browsers", text="Read our guide to download and install apps."),
    row(5, app="Helium", title="Helium Setup - Helium",
        text="Choose a search engine\nDuckDuckGo\nKagi\nStartpage", caption="Options are visible."),
    row(6, app="Mail", title="Inbox", text="Inbox\nSent\nDrafts\nTo: Pat"),
    row(7, app="Arc", title="Password managers", text="Install Bitwarden\nInstall Proton Pass"),
    row(8, app="Arc", title="X / Home", text="Someone posted: Order placed yesterday."),
])
def test_near_misses_do_not_claim_completion(observation):
    got = detect([observation])
    if observation["id"] != 5:  # The running native setup does prove installation.
        assert not [item for item in got if item.kind == "action"]
    if observation["id"] == 5:
        assert not [item for item in got if item.kind == "choice"]


@pytest.mark.parametrize(("text", "verb", "subject"), [
    ("Order placed", "ordered", "order"),
    ("Order placed: Widget Pro", "ordered", "Widget Pro"),
    ("Thank you for your order", "ordered", "order"),
    ("Booking confirmed", "booked", "booking"),
    ("Booking confirmed: Flight to Delhi", "booked", "Flight to Delhi"),
    ("Payment successful", "paid", "payment"),
    ("Post published", "posted", "post"),
    ("Message sent", "sent", "message"),
    ("Exported to /Users/me/Documents/report.pdf", "exported", "/Users/me/Documents/report.pdf"),
    ("File saved", "saved", "document.txt"),
    ("Welcome, Ada! Your account has been created", "signed_up", "example.com"),
    ("You're subscribed", "subscribed", "example.com"),
])
def test_explicit_confirmations(text, verb, subject):
    app = "Mail" if verb == "sent" else "TextEdit" if verb == "saved" else "Arc"
    got = detect([row(20, app=app, title="document.txt", text=text, site="example.com")])
    assert [(item.verb, item.subject) for item in got] == [(verb, subject)]
    assert got[0].quote in text


def test_checkmark_and_default_browser_confirmation():
    chosen = detect([row(1, app="Example", title="Theme Settings",
                         text="Choose theme\n✓ Dark\nLight\nSystem")])
    assert chosen == [Detection("choice", "chose", "Dark", 1, "✓ Dark", 0.78,
                                ("Light", "System"))]
    default = detect([row(2, app="Helium", title="Helium Settings",
                          text="Helium is now your default browser")])
    assert [(item.kind, item.subject) for item in default] == [("choice", "Helium")]


def test_download_page_mounted_dmg_and_engine_default_marker():
    rows = [
        row(1, title="Widget · Download", text="Your download is complete"),
        row(2, app="Finder", title="Widget", text="Drag the Widget icon into Applications"),
        row(3, app="Widget", title="Widget Setup", text="Choose a search engine\nKagi\nDuckDuckGo\nDefault: Kagi"),
    ]
    got = detect(rows)
    assert [(item.observation_id, item.verb, item.subject) for item in got] == [
        (1, "downloaded", "Widget"), (3, "installed", "Widget"), (3, "chose", "Kagi"),
    ]
    mounted = detect([rows[1]])
    assert [(item.verb, item.subject) for item in mounted] == [("downloaded", "Widget")]
    # A mounted disk image does not prove that an installation finished.
    assert not [item for item in got if item.observation_id == 2 and item.verb == "installed"]


def test_sent_mail_view_and_delivered_chat_caption():
    got = detect([
        row(1, app="Mail", title="Sent - Project update",
            text="Sent\nTo: pat@example.com\nProject update\nHello Pat"),
        row(2, app="Messages", title="Pat", text="Hello Pat\nDelivered",
            caption="The user's message 'Hello Pat' is marked Delivered."),
    ])
    assert [(item.observation_id, item.verb) for item in got] == [(1, "sent"), (2, "sent")]
    assert got[0].quote == "Sent"
    assert got[1].quote.startswith("The user's message")


def test_sqlite_rows_are_supported_and_detection_is_immutable():
    with sqlite3.connect(":memory:") as db:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE observations (id INTEGER, app TEXT, window_title TEXT, screen_text TEXT)")
        db.execute("INSERT INTO observations VALUES (7, 'Arc', 'Widget · Download', "
                   "'Your download is complete')")
        result = detect(db.execute("SELECT * FROM observations").fetchall())
    assert result[0].subject == "Widget"
    with pytest.raises(FrozenInstanceError):
        result[0].subject = "Other"
    assert hints([]) == ""
