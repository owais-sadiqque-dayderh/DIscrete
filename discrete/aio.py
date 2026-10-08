"""Background tasks that are never lost.

The event loop keeps only a weak reference to a task, so a task nobody holds can
be garbage-collected mid-flight, and an exception it raises is never seen.
`spawn` keeps a strong reference until the task finishes and logs its failure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("discrete.aio")

_live: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None,
          owner: set[asyncio.Task] | None = None,
          max_pending: int | None = None) -> asyncio.Task | None:
    """Start `coro` as a task that stays referenced until done. `owner` is an
    optional set the caller also tracks (to cancel its own tasks on shutdown)."""
    if owner is not None and max_pending is not None and len(owner) >= max_pending:
        coro.close()
        log.warning("background task admission full: %s", name)
        from .perf import PERF

        PERF.count("health.capture_finalizer_dropped")
        return None
    task = asyncio.get_running_loop().create_task(coro, name=name)
    holders = (_live,) if owner is None else (_live, owner)
    for h in holders:
        h.add(task)

    def _done(t: asyncio.Task) -> None:
        for h in holders:
            h.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("background task %s failed", t.get_name(), exc_info=t.exception())

    task.add_done_callback(_done)
    return task


def live_tasks() -> int:
    return len(_live)


# A daemon owns one data directory until every worker and connection is closed.
# Keep metadata in the locked inode; never unlink it (which would split ownership).
def _daemon_file():
    from .config import data_dir

    path = data_dir() / "runtime.lock"
    stream = open(path, "a+")
    path.chmod(0o600)
    return stream


def _process_identity(pid: int) -> str:
    import subprocess

    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


@contextmanager
def daemon_lease():
    import fcntl
    import json
    import os

    from .config import ROOT, db_path

    stream = _daemon_file()
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A daemon is already running for this data directory") from None
        record = {"pid": os.getpid(), "identity": _process_identity(os.getpid()),
                  "checkout": str(ROOT.resolve()), "db": str(db_path().resolve())}
        if not record["identity"]:
            raise RuntimeError("Cannot establish daemon process identity")
        stream.seek(0)
        stream.truncate()
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        stream.close()


def daemon_status() -> dict:
    import fcntl
    import json

    stream = _daemon_file()
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return {"running": False}
        except BlockingIOError:
            stream.seek(0)
            try:
                record = json.load(stream)
            except (ValueError, TypeError):
                raise RuntimeError("Daemon ownership is locked but its identity is unreadable") from None
            pid = record.get("pid")
            if type(pid) is not int or pid <= 0 or not record.get("identity"):
                raise RuntimeError("Invalid locked daemon identity") from None
            if _process_identity(pid) != record["identity"]:
                raise RuntimeError("Locked daemon process identity does not match") from None
            return {**record, "running": True}
    finally:
        stream.close()


def daemon_stop(*, timeout: float = 30.0) -> None:
    import os
    import signal
    import time

    from .config import ROOT

    record = daemon_status()
    if not record["running"]:
        return
    if record.get("checkout") != str(ROOT.resolve()):
        raise RuntimeError("This data directory is owned by another checkout; stop it from that checkout")
    pid, identity = record["pid"], record["identity"]
    # Check again immediately before signalling; stale metadata never authorizes kill.
    if _process_identity(pid) != identity:
        raise RuntimeError("Daemon identity changed; refusing to signal")
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = daemon_status()
        if not current["running"]:
            # Lock release proves cleanup finished. A replacement owner is never killed.
            return
        if current.get("pid") != pid or current.get("identity") != identity:
            raise RuntimeError("Daemon ownership changed during shutdown")
        time.sleep(0.05)
    raise RuntimeError(f"Daemon did not shut down within {timeout:g}s; restart aborted")
