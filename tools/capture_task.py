"""Record a CUA-operated window through the real capture path for reproducible replay.

The manifest keeps original pixels/text and actual timestamps. It uses an isolated
data directory; it never puts validation tasks into the user's normal memory store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import Quartz
from AppKit import NSRunningApplication

from discrete.capture.accessibility import WindowContext
from discrete.capture.screen import ScreenObserver
from discrete.capture.session import SessionTracker


async def record(args) -> None:
    windows = [w for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID) if w.get("kCGWindowLayer") == 0 and w.get("kCGWindowOwnerName") == args.app
                and (not args.title or w.get("kCGWindowName") == args.title)]
    if len(windows) != 1:
        raise ValueError(f"select exactly one window with --title; found {[w.get('kCGWindowName') for w in windows]}")
    w = windows[0]
    b = w["kCGWindowBounds"]
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(w["kCGWindowOwnerPID"]))
    ctx = WindowContext(app=args.app, bundle_id=str(app.bundleIdentifier() or ""),
        window_title=str(w.get("kCGWindowName") or ""), pid=int(w["kCGWindowOwnerPID"]), url=args.url or "",
        window_id=int(w["kCGWindowNumber"]), bounds=(float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])))
    queue: asyncio.Queue = asyncio.Queue()
    session = SessionTracker()
    observer = ScreenObserver(queue=queue, session=session)
    image = await asyncio.to_thread(observer.grabber.grab, ctx.bounds, max_width=1600, window_id=ctx.window_id)
    if image is None:
        raise ValueError("window capture failed")
    output = await asyncio.to_thread(Path(args.output).expanduser)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / f"{args.label}.png"
    image.save(image_path)
    pending = {"kind": "replay", "started": time.time(),
               "counts": {"keys": args.keys, "clicks": args.clicks, "scrolls": args.scrolls, "clipboard": 0}}
    await observer._capture(pending, ctx, None, image)
    capture = queue.get_nowait()
    payload = {"label": args.label, "draft": asdict(capture.draft), "image": str(image_path),
               "provenance": "CUA actions; window-isolated capture; declared interaction counts", "group": args.group or args.label}
    with (output / "captures.jsonl").open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    observer.grabber.close()
    print(json.dumps({"label": args.label, "app": args.app, "chars": len(capture.draft.screen_text),
                      "text_source": capture.draft.text_source, "image": str(image_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True)
    parser.add_argument("--title")
    parser.add_argument("--label", required=True)
    parser.add_argument("--group")
    parser.add_argument("--url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--keys", type=int, default=0)
    parser.add_argument("--clicks", type=int, default=0)
    parser.add_argument("--scrolls", type=int, default=0)
    asyncio.run(record(parser.parse_args()))
