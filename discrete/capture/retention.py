"""Frame and screen-text retention (planS2 §12)."""

from __future__ import annotations

import logging
import time

from ..config import T, frames_dir

log = logging.getLogger("discrete.capture.retention")


def purge_frames(*, ttl_days: float | None = None) -> int:
    ttl = T.FRAME_TTL_DAYS if ttl_days is None else ttl_days
    if ttl <= 0:
        return 0
    cutoff = time.time() - ttl * 86400
    removed = 0
    for path in frames_dir().glob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


async def run_retention(store) -> dict:
    """Rows are never deleted: evidence must outlive the text (insights.md §7.21)."""
    return {
        "frames_removed": purge_frames(),
        "screen_text_cleared": await store.clear_old_screen_text(ttl_days=T.SCREEN_TEXT_TTL_DAYS),
        "perf_rows_removed": await store.prune_perf(ttl_days=T.PERF_TTL_DAYS),
    }
