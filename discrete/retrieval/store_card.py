"""Cached store summary for upstream prompts."""

from __future__ import annotations

import time

from ..config import T
from ..store import Store
from ..timeutil import iso

_cache: tuple[float, str] | None = None


async def store_card(store: Store) -> str:
    global _cache
    now = time.time()
    if _cache and now - _cache[0] < T.STORE_CARD_TTL_S:
        return _cache[1]
    stats = await store.stats()
    text = (
        f"Store: {stats['observations']} observations, {stats['claims_live']} live claims, "
        f"{stats['entities']} entities, {stats['episodes']} episodes. "
        f"Coverage: {stats['first_observation'] and iso(stats['first_observation'])} → "
        f"{stats['last_observation'] and iso(stats['last_observation'])}. "
        f"Pending embeddings: {stats['outbox_pending']}."
    )
    _cache = (now, text)
    return text


def invalidate() -> None:
    global _cache
    _cache = None
