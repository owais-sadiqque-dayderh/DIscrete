"""Ground-truth browser URL via Apple Events (planS1 §6.3)."""

from __future__ import annotations

import collections
import logging
import subprocess

log = logging.getLogger("discrete.capture.browser")

_BROWSER_SCRIPTS = {
    "safari": 'tell application "Safari" to return URL of current tab of front window',
    "google chrome": 'tell application "Google Chrome" to return URL of active tab of front window',
    "arc": 'tell application "Arc" to return URL of active tab of front window',
    "microsoft edge": 'tell application "Microsoft Edge" to return URL of active tab of front window',
    "brave browser": 'tell application "Brave Browser" to return URL of active tab of front window',
}


# Chromium-family windows report "incognito" or "normal"; Safari and Arc expose no mode.
_MODE_SCRIPTS = {
    name: f'tell application "{app}" to return mode of front window'
    for name, app in (("google chrome", "Google Chrome"), ("microsoft edge", "Microsoft Edge"),
                      ("brave browser", "Brave Browser"))
}
# window id -> incognito; a window's mode never changes, so each is asked once.
_PRIVATE: collections.OrderedDict[int, bool] = collections.OrderedDict()


def _script_for(app_name: str, scripts: dict[str, str]) -> str | None:
    key = (app_name or "").lower()
    return next((scr for name, scr in scripts.items() if name in key), None)


def _osascript(script: str) -> str:
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2, check=False)
    return (out.stdout or "").strip()


def browser_url(app_name: str) -> str:
    script = _script_for(app_name, _BROWSER_SCRIPTS)
    if not script:
        return ""
    try:
        url = _osascript(script)
        return url if url.startswith("http") else ""
    except Exception as exc:
        log.debug("browser_url(%s): %s", app_name, exc)
        return ""


def note_private(app_name: str, window_id: int) -> None:
    """Learn whether `window_id`, the browser's front window right now, is incognito.
    Unanswered (no Automation permission) counts as normal: undetectable, like Safari."""
    script = _script_for(app_name, _MODE_SCRIPTS)
    if not script or not window_id or window_id in _PRIVATE:
        return
    try:
        mode = _osascript(script).lower()
    except Exception as exc:
        log.debug("browser mode(%s): %s", app_name, exc)
        mode = ""
    _PRIVATE[window_id] = mode == "incognito"
    while len(_PRIVATE) > 256:
        _PRIVATE.popitem(last=False)


def private_window(window_id: int) -> bool:
    """Whether `window_id` is known to be an incognito window (see note_private)."""
    return _PRIVATE.get(window_id, False)
