"""Drain the embedding outbox through the API (planS2 §11, Rev 2)."""

from __future__ import annotations

import logging

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed
from ..store import Store

log = logging.getLogger("discrete.process.embed")


async def text_for(store: Store, kind: str, rid: int) -> str | None:
    """Embedding text for a live row; None when the row is gone or not live."""
    if kind == "observation":
        r = await store.get_observation(rid)
        if not r or r["origin"] != "user":
            return None
        return "\n".join(p for p in (r["window_title"], r["site"], r["caption"], r["screen_text"]) if p)
    if kind == "claim":
        r = await store.get_claim(rid)
        if not r or r["status"] != "live":
            return None
        return f"{r['text']}\n{r['context']}".strip()
    if kind == "episode":
        r = await store.get_episode(rid)
        if not r or r["status"] != "closed":
            return None
        return f"{r['title']}\n{r['summary']}".strip()
    if kind == "entity":
        r = await store.get_entity(rid)
        if not r or r["merged_into"] is not None:
            return None
        return f"{r['name']}\n{r['summary']}".strip()
    return None


async def drain_once(store: Store, *, limit: int | None = None) -> int:
    rows = await store.outbox_batch(limit=limit or T.EMBED_BATCH)
    if not rows:
        return 0
    texts, refs = [], []
    for row in rows:
        kind, rid = row["ref_kind"], row["ref_id"]
        text = await text_for(store, kind, rid)
        if not text or not text.strip():
            await store.outbox_discard(kind, rid)
            continue
        texts.append(tk.truncate(text, T.EMBED_TEXT_CHARS))
        refs.append((kind, rid))
    if not texts:
        return 0
    try:
        vectors = await embed(texts)
    except (NotConfigured, *EXPECTED_ERRORS) as exc:
        log.warning("embed batch failed: %s", exc)
        await store.outbox_failed(refs)
        return 0
    await store.save_embeddings([(k, r, v, tk.content_hash(t)) for (k, r), v, t in zip(refs, vectors, texts)])
    return len(vectors)


async def drain_outbox(store: Store, *, max_batches: int = 50) -> int:
    total = 0
    for _ in range(max_batches):
        n = await drain_once(store)
        if not n:
            break
        total += n
    return total
