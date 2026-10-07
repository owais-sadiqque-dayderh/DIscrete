from discrete import textkit as tk
from discrete.models import ClaimDraft, ObservationDraft
from discrete.process import embed_worker


async def _vector_hash(store, kind, rid):
    row = await store.q1("SELECT text_hash FROM embeddings WHERE ref_kind=? AND ref_id=?", (kind, rid))
    return row[0] if row else None


async def test_late_embedding_does_not_replace_newer_vector_or_consume_newer_work(store, monkeypatch):
    oid, _ = await store.add_observation(ObservationDraft(app="Code", screen_text="old", started_at=100, ended_at=101))
    old = await embed_worker.text_for(store, "observation", oid)

    async def delayed(texts):
        # While the call is out, the row changes and its newer text is embedded first.
        await store.execute("UPDATE observations SET screen_text='new' WHERE id=?", (oid,))
        new = await embed_worker.text_for(store, "observation", oid)
        await store.save_embeddings([("observation", oid, [0., 1.], tk.content_hash(new))])
        await store.execute("INSERT INTO index_outbox(ref_kind,ref_id,enqueued_at,attempts) VALUES ('observation',?,999,0)",
                            (oid,))
        return [[1., 0.]]

    monkeypatch.setattr(embed_worker, "embed", delayed)
    assert await embed_worker.embed_row(store, "observation", oid, old) is None
    new = await embed_worker.text_for(store, "observation", oid)
    assert await _vector_hash(store, "observation", oid) == tk.content_hash(new)
    assert await store.q1("SELECT 1 FROM index_outbox WHERE ref_kind='observation' AND ref_id=?", (oid,))


async def test_stale_outbox_completion_is_requeued_not_stored(store, monkeypatch):
    oid, _ = await store.add_observation(ObservationDraft(app="Code", screen_text="old", started_at=100, ended_at=101))
    await store.execute("DELETE FROM embeddings")

    async def delayed(texts):
        await store.execute("UPDATE observations SET screen_text='new' WHERE id=?", (oid,))
        return [[1., 0.] for _ in texts]

    monkeypatch.setattr(embed_worker, "embed", delayed)
    assert await embed_worker.drain_once(store) == 1
    assert await _vector_hash(store, "observation", oid) is None
    assert await store.q1("SELECT 1 FROM index_outbox WHERE ref_kind='observation' AND ref_id=?", (oid,))


async def test_late_embedding_cannot_restore_a_forgotten_claim(store, monkeypatch):
    cid = await store.create_claim(ClaimDraft(kind="fact", text="The user uses Acme"))
    text = await embed_worker.text_for(store, "claim", cid)

    async def delayed(texts):
        await store.forget_claim(cid)
        return [[1., 0.]]

    monkeypatch.setattr(embed_worker, "embed", delayed)
    assert await embed_worker.embed_row(store, "claim", cid, text) is None
    assert await _vector_hash(store, "claim", cid) is None
