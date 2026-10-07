from discrete import textkit as tk
from discrete.config import Endpoint, T
from discrete.models import ObservationDraft
from discrete.process.pipeline import ProcessPipeline


def test_recurring_content_is_not_chrome():
    text = "Home timeline\nThe invoice total is 314 dollars\nAlex: can you send the quarterly report today?"
    assert tk.chrome_lines([text] * 6, share=0.3, min_texts=5) == {"home timeline"}


async def _rows(store, texts):
    rows = []
    for i, text in enumerate(texts):
        oid, _ = await store.add_observation(ObservationDraft(app="Code", window_title=str(i), screen_text=text,
                                                              started_at=100 + i, ended_at=101 + i))
        rows.append(await store.get_observation(oid))
    return rows


async def test_lines_cut_by_truncation_reach_the_drafter_from_a_later_copy(store, monkeypatch):
    monkeypatch.setattr(T, "DRAFT_TEXT_CHARS_PER_OBS", 40)
    rows = await _rows(store, ["x" * 80 + "\nCritical total 314", "Critical total 314"])
    assert "Critical total 314" in ProcessPipeline(store)._draft_prompt(rows)


async def test_identical_long_captures_deliver_the_rest_once(store, monkeypatch):
    monkeypatch.setattr(T, "DRAFT_TEXT_CHARS_PER_OBS", 40)
    body = "a" * 30 + "b" * 30 + "\nCritical total 314"
    prompt = ProcessPipeline(store)._draft_prompt(await _rows(store, [body] * 4))
    assert "Critical total 314" in prompt
    assert prompt.count("b" * 20) == 1  # the tail once, not every copy


async def test_missing_model_defers_a_populated_episode_until_configured(store, fake):
    eid = await store.open_episode("s", 100)
    oid, _ = await store.add_observation(ObservationDraft(app="Code", screen_text="report", started_at=100, ended_at=101))
    await store.execute("UPDATE observations SET episode_id=? WHERE id=?", (eid, oid))
    await store.mark_episode_ready(eid)
    pipe = ProcessPipeline(store)
    configured, pipe.ep = pipe.ep, Endpoint("insight", "", "", "")
    for _ in range(T.DRAFT_MAX_ATTEMPTS + 1):
        assert await pipe.draft_episode(eid) is None
    ep = await store.get_episode(eid)
    assert ep["status"] == "ready" and ep["draft_attempts"] == 0 and "draft" not in fake.calls
    assert await pipe.process_ready() == []  # a draining loop terminates
    pipe.ep = configured
    assert await pipe.draft_episode(eid)
    assert (await store.get_episode(eid))["status"] == "closed" and fake.calls.count("draft") == 1
