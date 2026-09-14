"""Ground-truth browser URL via Apple Events (planS1 §6.3)."""

from __future__ import annotations

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


def browser_url(app_name: str) -> str:
    key = (app_name or "").lower()
    script = None
    for name, scr in _BROWSER_SCRIPTS.items():
        if name in key:
            script = scr
            break
    if not script:
        return ""
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        url = (out.stdout or "").strip()
        return url if url.startswith("http") else ""
    except Exception as exc:
        log.debug("browser_url(%s): %s", app_name, exc)
        return ""
