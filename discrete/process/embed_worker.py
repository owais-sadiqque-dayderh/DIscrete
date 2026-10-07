"""Embeddings through the API (planS2 §11, Rev 2).

Rows are embedded once. Observations are embedded as they are segmented
(`embed_row`) and durable claims as they are merged, with the vector stored
straight away; the outbox only catches what those paths missed (a failed
call, an event claim, an episode or entity page, retention edits).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed
from ..perf import PERF, since_ms
from ..store import Store
from ..store.index import embedding_text

log = logging.getLogger("discrete.process.embed")


def observation_text(row) -> str:
    return embedding_text("observation", row)


def claim_text(text: str, context: str) -> str:
    return embedding_text("claim", {"text": text, "context": context})


async def text_for(store: Store, kind: str, rid: int) -> str | None:
    """Embedding text for a live row; None when the row is gone or not live."""
    if kind == "observation":
        r = await store.get_observation(rid)
        if not r or r["origin"] != "user":
            return None
        return observation_text(r)
    if kind == "claim":
        r = await store.get_claim(rid)
        if not r or r["status"] != "live":
            return None
        return claim_text(r["text"], r["context"])
    if kind == "episode":
        r = await store.get_episode(rid)
        if not r or r["status"] != "closed":
            return None
        return embedding_text(kind, r)
    if kind == "entity":
        r = await store.get_entity(rid)
        if not r or r["merged_into"] is not None:
            return None
        return embedding_text(kind, r)
    return None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


async def embed_row(store: Store, kind: str, rid: int, text: str) -> np.ndarray | None:
    """Unit vector for a row's text, stored once. Reuses a stored vector for the same
    text; on failure returns None and leaves the row in the outbox for a retry."""
    if not text.strip():
        return None
    th = tk.content_hash(text)
    if (have := await store.get_vector(kind, rid, th)) is not None:
        return _unit(have)
    try:
        vec = (await embed([text]))[0]
    except (NotConfigured, *EXPECTED_ERRORS) as exc:
        log.debug("embed %s:%s deferred to the outbox: %s", kind, rid, exc)
        return None
    if not await store.save_embeddings([(kind, rid, vec, th)], current=True):
        return None  # the row changed or went away while the call was out
    return _unit(vec)


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
        texts.append(text)
        refs.append((kind, rid))
    if not texts:
        return 0
    try:
        vectors = await embed(texts)
    except (NotConfigured, *EXPECTED_ERRORS) as exc:
        log.warning("embed batch failed: %s", exc)
        await store.outbox_failed(refs)
        return 0
    await store.save_embeddings([(k, r, v, tk.content_hash(t)) for (k, r), v, t in zip(refs, vectors, texts, strict=True)],
                                current=True)
    return len(vectors)


async def drain_outbox(store: Store, *, max_batches: int = 50) -> int:
    total, t0 = 0, time.perf_counter()
    for _ in range(max_batches):
        n = await drain_once(store)
        if not n:
            break
        total += n
    if total:
        PERF.sample("embed.outbox", since_ms(t0), n=total)
    return total
