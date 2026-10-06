"""Conservative, quote-backed signals from individual screen observations.

These are candidates for the drafter, not claims about the user's intent. In
particular, a visible option is not a choice without a selection marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Detection:
    kind: Literal["action", "choice"]
    verb: str
    subject: str
    observation_id: int
    quote: str
    confidence: float
    alternatives: tuple[str, ...] = ()


_DOWNLOAD_THANKS = re.compile(r"\bThank you for downloading (?P<name>[^\n!?]{2,90})[!?]?", re.I)
_DOWNLOAD_DONE = re.compile(
    r"\b(?:Download complete(?:d)?|Downloaded successfully)\s*[:—-]\s*"
    r"(?P<name>[^\n!?]{2,90})",
    re.I,
)
_DOWNLOAD_COMPLETE_PAGE = re.compile(r"\bYour download is complete\b", re.I)
_FILE_TODAY = re.compile(
    r"(?P<name>[\w][\w .()+-]{0,100}\.(?:dmg|pkg|zip|pdf|jpe?g|png|mov|mp4|app))"
    r"\n(?:[^\n]{0,100}\n){0,3}Today(?: at [^\n]{1,30}|\b)",
    re.I,
)
_DISK_IMAGE = re.compile(r"\b(?P<name>[\w][\w .+-]{1,60}) disk image installer is open\b", re.I)
_DRAG_INSTALLER = re.compile(r"\bDrag (?:the )?(?P<name>[\w][\w .+-]{1,60}?) (?:icon )?(?:in)?to (?:your )?Applications\b", re.I)
_SETUP_TITLE = re.compile(
    r"^(?:(?P<setup>[\w][\w .+-]{1,50}) Setup|Install (?P<install>[\w][\w .+-]{1,50})"
    r"|Welcome to (?P<welcome>[\w][\w .+-]{1,50}))(?:\b|\s*[-—])",
    re.I,
)
_ORDER = re.compile(
    r"(?m)^[ \t]*(?:Order placed|Thank you for your order|Order confirmed)"
    r"(?::[ \t]*(?P<name>[^\n]{2,90}))?(?=[.!\n]|$)", re.I,
)
_BOOKING = re.compile(
    r"(?m)^[ \t]*(?:Booking confirmed|Reservation confirmed|Your booking is confirmed)"
    r"(?::[ \t]*(?P<name>[^\n]{2,90}))?(?=[.!\n]|$)", re.I,
)
_PAYMENT = re.compile(r"(?m)^[ \t]*Payment successful(?=[.!\n]|$)", re.I)
_POSTED = re.compile(r"(?m)^[ \t]*(?:Post published|Your post has been published)(?=[.!\n]|$)", re.I)
_MESSAGE = re.compile(r"(?m)^[ \t]*(?:Message sent|Your message was sent|Email sent)(?=[.!\n]|$)", re.I)
_DELIVERED_CAPTION = re.compile(
    r"\b(?:the user's|their|a sent) (?:chat )?message\b[^\n.]{0,120}\bdelivered\b",
    re.I,
)
_EXPORTED = re.compile(r"\bExported to\s+(?P<name>[^\n!?]{2,100})", re.I)
_SAVED = re.compile(r"\b(?:File saved|Saved successfully|Saved to\s+(?P<name>[^\n!?]{2,100}))\b", re.I)
_ACCOUNT = re.compile(
    r"\b(?:Welcome,?\s+[^\n.!?]{1,60}[.!]?\s+)?Your account has been created\b",
    re.I,
)
_SUBSCRIBED = re.compile(r"\b(?:You're subscribed|You are now subscribed|Subscription confirmed)\b", re.I)
_DEFAULT_DONE = re.compile(
    r"\b(?P<name>[\w][\w .+-]{1,50}) is now your default browser\b"
    r"|\bDefault browser set to (?P<set>[\w][\w .+-]{1,50})\b",
    re.I,
)
_CHECKED = re.compile(r"^\s*(?:✓|✔|☑)\s*(?P<name>[^\n]{2,50})$", re.I)
_ENGINE_NAMES = (
    "DuckDuckGo", "Kagi", "Startpage", "Brave Search", "Ecosia", "Qwant",
    "Google", "Microsoft Bing", "Bing", "Yahoo",
)
_BROWSER_APPS = frozenset({"arc", "chrome", "google chrome", "safari", "firefox", "edge", "helium"})
_EDIT_APPS = frozenset({"textedit", "notes", "preview", "pages", "numbers", "keynote", "code"})


def _value(row: object, key: str) -> str:
    try:
        value = row[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return ""
    return str(value) if value is not None else ""


def _subject(value: str) -> str:
    return value.strip(" \t\r\n'\".,!?:;—–-")[:100]


def _fields(row: object) -> tuple[str, ...]:
    return tuple(_value(row, key) for key in ("screen_text", "caption", "window_title", "page_title"))


def _signal(kind: Literal["action", "choice"], verb: str, subject: str, row: object,
            quote: str, confidence: float, alternatives: tuple[str, ...] = ()) -> Detection:
    return Detection(kind, verb, _subject(subject), int(_value(row, "id")), quote,
                     confidence, alternatives)


def _actions(row: object, fields: tuple[str, ...]) -> list[Detection]:
    screen, caption, title, _ = fields
    app = _value(row, "app").lower()
    if app == "terminal":
        return []  # Logs, command help and the app's own recall output are not actions.
    out: list[Detection] = []
    for pattern, confidence in ((_DOWNLOAD_THANKS, 0.96), (_DOWNLOAD_DONE, 0.94)):
        match = next((m for field in (screen, caption) if (m := pattern.search(field))), None)
        if match:
            out.append(_signal("action", "downloaded", match["name"], row,
                               match.group(), confidence))
            break
    else:
        match = _DOWNLOAD_COMPLETE_PAGE.search(screen)
        title_name = re.match(r"(?P<name>[^·|]{2,60})\s*[·|]\s*Download", title, re.I)
        if match and title_name:
            out.append(_signal("action", "downloaded", title_name["name"], row,
                               match.group(), 0.91))

    if app == "finder" and "downloads" in title.lower() and "Date Added" in screen:
        match = _FILE_TODAY.search(screen)
        if match:
            out.append(_signal("action", "downloaded", match["name"], row,
                               match.group(), 0.91))
    if app == "finder" and "Applications" in screen:
        match = _DISK_IMAGE.search(caption) or _DRAG_INSTALLER.search(screen)
        match = match or _DRAG_INSTALLER.search(caption)
        if match and _subject(match["name"]).casefold() == title.casefold():
            out.append(_signal("action", "downloaded", match["name"], row,
                               match.group(), 0.84))

    # A native app's own setup wizard demonstrates that it has been installed
    # and launched. A web page with an Install button does not.
    if app and not _value(row, "site"):
        match = _SETUP_TITLE.search(title) or _SETUP_TITLE.search(screen[:150])
        if match:
            name = next(part for part in match.groups() if part)
            if app in {name.casefold(), "installer", "finder"}:
                out.append(_signal("action", "installed", name, row,
                                   match.group(), 0.88))

    for pattern, verb, subject in (
        (_ORDER, "ordered", "order"), (_BOOKING, "booked", "booking"),
        (_PAYMENT, "paid", "payment"),
    ):
        match = next((m for field in (screen, caption) if (m := pattern.search(field))), None)
        if match:
            out.append(_signal("action", verb, match.groupdict().get("name") or subject,
                               row, match.group(), 0.96))

    match = next((m for field in (screen, caption) if (m := _POSTED.search(field))), None)
    if match:
        out.append(_signal("action", "posted", "post", row, match.group(), 0.94))

    chat_context = (app in {"mail", "messages", "slack", "discord", "whatsapp"}
                    or (app in _BROWSER_APPS and any(
                        part in (_value(row, "site") + " " + title + " " + _value(row, "url")).lower()
                        for part in ("mail", "message", "chat", "slack", "discord", "whatsapp"))))
    if chat_context:
        match = next((m for field in (screen, caption) if (m := _MESSAGE.search(field))), None)
        match = match or _DELIVERED_CAPTION.search(caption)
        if match:
            out.append(_signal("action", "sent", "message", row, match.group(), 0.92))
        elif (app == "mail" and title.lower().startswith(("sent -", "sent mail"))
              and re.search(r"\bTo:\s*\S+", screen) and "Sent" in screen):
            out.append(_signal("action", "sent", "email", row, "Sent", 0.85))

    match = next((m for field in (screen, caption) if (m := _EXPORTED.search(field))), None)
    if match:
        out.append(_signal("action", "exported", match["name"], row, match.group(), 0.94))
    if app in _EDIT_APPS or (app not in _BROWSER_APPS and _value(row, "site") == ""):
        match = next((m for field in (screen, caption) if (m := _SAVED.search(field))), None)
        if match:
            out.append(_signal("action", "saved", match.groupdict().get("name") or title or "file",
                               row, match.group(), 0.88))
        elif app in _EDIT_APPS and re.search(r"\.[\w]{1,8}$", title):
            match = re.search(r"(?m)^Saved$", screen)
            if match:
                out.append(_signal("action", "saved", title, row, match.group(), 0.82))

    for pattern, verb, subject in (
        (_ACCOUNT, "signed_up", _value(row, "site") or _value(row, "app") or "account"),
        (_SUBSCRIBED, "subscribed", _value(row, "site") or "subscription"),
    ):
        match = next((m for field in (screen, caption) if (m := pattern.search(field))), None)
        if match:
            out.append(_signal("action", verb, subject, row, match.group(), 0.94))
    return out


def _choices(row: object, fields: tuple[str, ...]) -> list[Detection]:
    screen, caption, title, page_title = fields
    if _value(row, "app").lower() == "terminal":
        return []
    context = " ".join((title, page_title, screen[:400], _value(row, "url"))).lower()
    if not any(word in context for word in ("setup", "settings", "preferences", "choose a ", "default browser")):
        return []
    out: list[Detection] = []
    if "search engine" in context:
        present = tuple(name for name in _ENGINE_NAMES
                        if re.search(rf"(?m)^{re.escape(name)}(?:\.|$)", screen, re.I))
        for name in present:
            escaped = re.escape(name)
            marker = re.search(rf"\b{escaped}\s+(?:is\s+)?selected\b", caption, re.I)
            marker = marker or re.search(rf"\bDefault:\s*{escaped}\b", screen, re.I)
            marker = marker or re.search(rf"(?m)^[ \t]*[✓✔☑][ \t]*{escaped}\b", screen)
            if marker and len(present) >= 2:
                out.append(_signal("choice", "chose", name, row, marker.group(), 0.91,
                                   tuple(other for other in present if other != name)))
                break

    match = re.search(r"Yes,?\s+Make\s+(?P<name>[\w .+-]{2,50})\s+my default browser[\w\s,.'’]*selected",
                      caption, re.I)
    if match and "default browser" in screen.lower() and re.search(r"\bNo\b", screen):
        out.append(_signal("choice", "chose", match["name"], row, match.group(), 0.91,
                           ("No, just trying for now",)))
    else:
        match = next((m for field in (screen, caption) if (m := _DEFAULT_DONE.search(field))), None)
        if match:
            out.append(_signal("choice", "chose", match["name"] or match["set"], row,
                               match.group(), 0.95))

    if not out and "search engine" not in context:
        lines = [line.strip() for line in screen.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            marker = _CHECKED.fullmatch(line)
            if not marker:
                continue
            name = _subject(marker["name"])
            neighbors = lines[index + 1:index + 4] or lines[max(0, index - 3):index]
            options = tuple(dict.fromkeys(
                option for option in neighbors if 2 <= len(option) <= 40
                and not any(mark in option for mark in (".", ":", "→", "✓", "✔", "☑"))
                and not option.lower().startswith(("choose ", "back", "next", "save", "cancel"))
            ))
            if options:
                out.append(_signal("choice", "chose", name, row, line, 0.78, options))
                break
    return out


def detect(rows: list[object]) -> list[Detection]:
    """Find completed actions and explicitly selected choices in observation rows.

    Repeated views of the same signal within this batch yield one detection.
    A later batch may legitimately report it again; persistence dedup belongs to
    the caller.
    """
    found: list[Detection] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        fields = _fields(row)
        for item in (*_actions(row, fields), *_choices(row, fields)):
            key = (item.kind, item.verb, item.subject.casefold())
            if item.subject and item.quote and key not in seen:
                found.append(item)
                seen.add(key)
    return found


def hints(detections: list[Detection]) -> str:
    """Render short, explicitly tentative evidence for an episode draft prompt."""
    if not detections:
        return ""
    lines = ["Detected on screen (verify before using):"]
    for item in detections:
        quote = " ".join(item.quote.split())[:140]
        lines.append(f"obs {item.observation_id} {item.verb} {item.subject} — “{quote}”")
    return "\n".join(lines)
