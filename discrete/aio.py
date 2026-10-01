"""Background tasks that are never lost.

The event loop keeps only a weak reference to a task, so a task nobody holds can
be garbage-collected mid-flight, and an exception it raises is never seen.
`spawn` keeps a strong reference until the task finishes and logs its failure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger("discrete.aio")

_live: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None,
          owner: set[asyncio.Task] | None = None) -> asyncio.Task:
    """Start `coro` as a task that stays referenced until done. `owner` is an
    optional set the caller also tracks (to cancel its own tasks on shutdown)."""
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
