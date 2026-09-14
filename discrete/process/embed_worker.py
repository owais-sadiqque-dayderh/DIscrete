"""Drain the embedding outbox through the API (planS2 Rev 2)."""

from __future__ import annotations

import logging

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed, embedding_model
from ..store import Store, np_bytes

log = logging.getLogger("discrete.process.embed")


async def drain_outbox(store: Store, *, limit: int | None = None) -> int:
    limit = limit or T.EMBED_BATCH
    rows = await store.outbox_batch(limit=limit)
    if not rows:
        return 0
    texts: list[str] = []
    refs: list[tuple[str, int, str]] = []
    for row in rows:
        kind, rid = row["ref_kind"], row["ref_id"]
        text, th = await _text_for(store, kind, rid)
        if not text.strip():
            continue
        texts.append(tk.truncate(text, T.EMBED_TEXT_CHARS))
        refs.append((kind, rid, th))
    if not texts:
        return 0
    try:
        vectors = await embed(texts)
    except (NotConfigured, *EXPECTED_ERRORS) as exc:
        log.warning("embed batch failed: %s", exc)
        return 0
    for (kind, rid, th), vec in zip(refs, vectors):
        await store.save_embedding(kind, rid, vec, th)
    return len(vectors)


async def _text_for(store: Store, kind: str, rid: int) -> tuple[str, str]:
    if kind == "observation":
        row = await store.get_observation(rid)
        if not row:
            return "", ""
        text = "\n".join(p for p in (row["caption"], row["screen_text"], row["app"], row["window_title"]) if p)
        return text, tk.content_hash(text)
    if kind == "claim":
        live = await store.live_claims()
        row = next((r for r in live if r["id"] == rid), None)
        if not row:
            return "", ""
        text = f"{row['text']} {row['context']}"
        return text, tk.content_hash(text)
    return "", ""
