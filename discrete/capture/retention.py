"""Frame and observation retention (planS2 §12)."""

from __future__ import annotations

import logging
import os
import time

from ..config import T, frames_dir

log = logging.getLogger("discrete.capture.retention")


def purge_frames(*, ttl_days: float | None = None) -> dict:
    ttl = T.FRAME_TTL_DAYS if ttl_days is None else ttl_days
    d = frames_dir()
    if ttl <= 0 or not d.is_dir():
        return {"removed": 0, "bytes": 0}
    cutoff = time.time() - ttl * 86400
    removed = freed = 0
    for path in d.glob("*.jpg"):
        try:
            st = path.stat()
            if st.st_mtime < cutoff:
                freed += st.st_size
                path.unlink()
                removed += 1
        except OSError:
            continue
    return {"removed": removed, "bytes": freed}


async def run_retention(store, *, screen_text_ttl_days: float | None = None) -> dict:
    shots = purge_frames()
    ttl = T.SCREEN_TEXT_TTL_DAYS if screen_text_ttl_days is None else screen_text_ttl_days
    purged = await store.purge_old_observations(ttl_days=ttl)
    return {"frames": shots, "observations_purged": purged}
