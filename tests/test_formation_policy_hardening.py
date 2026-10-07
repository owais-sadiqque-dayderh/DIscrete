import json

from discrete.capture import privacy
from discrete.models import ClaimDraft, ObservationDraft
from discrete.process import embed_worker
from discrete.process.pipeline import ProcessPipeline


async def _episode(store, *apps):
    eid = await store.open_episode("s", 100)
    oids = []
    for i, app in enumerate(apps):
        oid, _ = await store.add_observation(ObservationDraft(app=app, window_title=app, screen_text=f"{app} report {i}",
                                                              started_at=100 + i, ended_at=101 + i))
        await store.execute("UPDATE observations SET episode_id=? WHERE id=?", (eid, oid))
        oids.append(oid)
    await store.mark_episode_ready(eid)
    return eid, oids


async def test_drafting_leaves_out_history_from_a_newly_denied_app(store, fake, monkeypatch):
    eid, (denied, kept) = await _episode(store, "Vault", "Notes")
    prompts = []
    chat = fake.chat

    async def recording(ep, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        return await chat(ep, messages, **kwargs)

    monkeypatch.setattr(fake, "chat", recording)
    monkeypatch.setattr(privacy, "_APPS", privacy._APPS | {"vault"})
    assert await ProcessPipeline(store).draft_episode(eid)
    assert prompts and not any("Vault report" in p for p in prompts) and any("Notes report" in p for p in prompts)
    assert json.loads(await store.get_meta(f"draft_policy_blocked:{eid}")) == [denied]
    assert (await store.get_episode(eid))["status"] == "closed"


async def test_a_fully_denied_episode_closes_without_model_calls(store, fake, monkeypatch):
    eid, _ = await _episode(store, "Vault")
    monkeypatch.setattr(privacy, "_APPS", privacy._APPS | {"vault"})
    await ProcessPipeline(store).draft_episode(eid)
    ep = await store.get_episode(eid)
    assert ep["status"] == "closed" and "Vault" not in ep["title"] and fake.calls == []


async def test_embedding_skips_rows_from_newly_denied_sources(store, fake, monkeypatch):
    eid, (oid,) = await _episode(store, "Vault")
    cid = await store.create_claim(ClaimDraft(kind="fact", text="The user keeps a vault report",
                                              evidence=[{"observation_id": oid, "quote": "Vault report"}]))
    await store.execute("DELETE FROM embeddings")
    await store.execute("INSERT OR REPLACE INTO index_outbox(ref_kind,ref_id,enqueued_at,attempts) "
                        "VALUES ('observation',?,1,0), ('claim',?,1,0)", (oid, cid))
    monkeypatch.setattr(privacy, "_APPS", privacy._APPS | {"vault"})
    text = await embed_worker.text_for(store, "observation", oid)
    assert await embed_worker.embed_row(store, "observation", oid, text) is None
    await embed_worker.drain_once(store)
    assert fake.embedded == []
    assert await store.q("SELECT * FROM index_outbox WHERE ref_kind IN ('observation','claim')") == []
