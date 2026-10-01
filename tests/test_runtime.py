"""Capture → store → enrich → embed/segment (docs/C6.md §1, §3)."""

from __future__ import annotations

import asyncio
import gc
import logging
import time

from PIL import Image

from discrete.aio import live_tasks, spawn
from discrete.capture import accessibility as ax
from discrete.capture import screen as sc
from discrete.capture.enrich import Capture, Enricher, EnrichJob
from discrete.config import T
from discrete.models import ObservationDraft
from discrete.process.embed_worker import drain_outbox, observation_text

T0 = 1_789_000_000.0
TEXT = "Quarterly planning notes for the archive migration and the storage budget review"


def _ctx(title="Plan", url="https://example.org/plan"):
    return ax.WindowContext(app="Safari", bundle_id="com.apple.Safari", window_title=title, url=url, pid=7)


def _img(shade=128):
    img = Image.new("RGB", (320, 200), (shade, shade, shade))
    for x in range(0, 320, 16):  # some structure, so frames hash differently
        img.paste((255 - shade, 20, x % 255), (x, (x * 3) % 180, x + 8, (x * 3) % 180 + 20))
    return img


def _job(text=TEXT):
    return EnrichJob(frames=[_img()], app="Safari", window_title="Plan", url="https://example.org/plan",
                     screen_text=text, counts={"keys": 0, "clicks": 1, "scrolls": 0, "clipboard": 0})


def _draft(title="Plan", text=TEXT, t=T0, **kw):
    return ObservationDraft(app="Safari", window_title=title, url=f"https://example.org/{title}", screen_text=text,
                            caption=f"Safari: {title}", started_at=t, ended_at=t + 5, clicks=1, session_id="s", **kw)


# ------------------------------------------------------------------ tasks

async def test_spawned_task_survives_gc_and_its_failure_is_logged(caplog):
    done = asyncio.Event()

    async def work():
        await asyncio.sleep(0.01)
        done.set()
        raise RuntimeError("boom")

    before = live_tasks()
    spawn(work(), name="doomed")  # no reference kept by the caller
    gc.collect()
    with caplog.at_level(logging.ERROR, logger="discrete.aio"):
        await asyncio.wait_for(done.wait(), 1)
        await asyncio.sleep(0.01)
    assert live_tasks() == before
    assert any("doomed" in r.getMessage() for r in caplog.records)


# -------------------------------------------------------- capture path

async def test_capture_is_queued_without_a_model_call(store, fake, monkeypatch):
    """The capture lock no longer covers the vision call: a capture is queued at once
    with a basic caption and its frames, and the model is not called on this path."""
    q: asyncio.Queue = asyncio.Queue()
    obs = sc.ScreenObserver(queue=q)
    monkeypatch.setattr(obs, "_screen_text", lambda ctx, after, stale=False: (TEXT, "ax"))
    pending = {"kind": "click", "started": time.time(), "counts": {"keys": 0, "clicks": 1, "scrolls": 0, "clipboard": 0}}
    await obs._capture(pending, _ctx(), None, _img())
    cap: Capture = q.get_nowait()
    assert cap.job is not None and cap.draft.enrich_state == "pending"
    assert cap.draft.caption == "Safari: Plan" and cap.draft.frame_path
    assert "caption" not in fake.calls


async def test_slow_vision_call_does_not_hold_up_the_next_capture(store, fake, monkeypatch):
    fake.caption_gate = asyncio.Event()  # the model hangs until released
    q: asyncio.Queue = asyncio.Queue()
    obs = sc.ScreenObserver(queue=q)
    monkeypatch.setattr(obs, "_screen_text", lambda ctx, after, stale=False: (TEXT + ctx.window_title, "ax"))
    enricher = Enricher(store)
    counts = {"keys": 0, "clicks": 1, "scrolls": 0, "clipboard": 0}
    await obs._finalize({"kind": "click", "started": time.time(), "counts": counts, "before_img": None},
                        _ctx("A", "https://example.org/a"), _img(40), stale=False)
    first = q.get_nowait()
    oid, _ = await store.add_observation(first.draft)
    task = enricher.submit(oid, first.job)
    await asyncio.sleep(0.05)
    assert not task.done()  # the vision call is still hanging
    # A second capture completes while the first caption is outstanding.
    await asyncio.wait_for(obs._finalize({"kind": "click", "started": time.time(), "counts": counts,
                                          "before_img": None}, _ctx("B", "https://example.org/b"), _img(200),
                                         stale=False), 1)
    assert q.qsize() == 1
    fake.caption_gate.set()
    await asyncio.wait_for(task, 1)


async def test_enrichment_fills_caption_and_page_fields(store, fake):
    fake.caption = {"caption": "The user reviews the storage budget.", "page_title": "Storage budget 2027",
                    "is_listing": False, "navigated_from": "Team wiki", "entities": ["Storage"], "useful": True}
    oid, _ = await store.add_observation(_draft(enrich_state="pending"))
    await Enricher(store).submit(oid, _job())
    row = await store.get_observation(oid)
    assert row["caption"] == "The user reviews the storage budget."
    assert row["page_title"] == "Storage budget 2027" and row["navigated_from"] == "Team wiki"
    assert row["enrich_state"] == "done" and not row["is_listing"]
    assert oid in await store.fts_ids("observation", "budget 2027")  # page title is indexed


async def test_vision_retries_do_not_sleep_after_the_last_attempt(store, fake, monkeypatch):
    fake.caption = None  # every attempt fails
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def record(d, *a, **k):
        slept.append(d)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", record)
    oid, _ = await store.add_observation(_draft(enrich_state="pending"))
    await Enricher(store).submit(oid, _job())
    assert len(slept) == T.CAPTURE_MAX_RETRIES - 1
    row = await store.get_observation(oid)
    assert row["enrich_state"] == "failed" and row["source"] == "degraded"
    assert row["caption"] == "Safari: Plan"  # the basic caption stays


async def test_useless_frame_leaves_every_index(store, fake):
    fake.caption = {"caption": "", "useful": False}
    oid, _ = await store.add_observation(_draft(enrich_state="pending"))
    await Enricher(store).submit(oid, _job())
    row = await store.get_observation(oid)
    assert row["origin"] == "discarded"
    assert await store.live_rows("observation", [oid]) == []
    df, n = await store.df_lookup(["quarterly"])
    assert df.get("quarterly", 0) == 0 and n == 0
    # An identical frame later still dedups against it: no second vision call.
    _, is_new = await store.add_observation(_draft())
    assert not is_new


def test_frame_duplicate_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(T, "LAST_KEPT_MAX", 10)
    obs = sc.ScreenObserver(queue=asyncio.Queue())
    for i in range(50):
        obs._remember(("app", f"w{i}", ""), "0" * 16)
    assert len(obs._last_kept) == 10 and ("app", "w49", "") in obs._last_kept


# ------------------------------------------------------ runtime wiring

async def test_runtime_embeds_each_observation_once_after_its_caption(store, fake):
    from discrete.app import Runtime

    fake.caption = {"caption": "The user edits the plan.", "page_title": "Plan", "useful": True}
    rt = Runtime(store, bubble=False)
    tasks = [asyncio.create_task(rt._ingest_loop()), asyncio.create_task(rt._segment_loop())]
    try:
        await rt.queue.put(Capture(_draft("a", enrich_state="pending", text=TEXT + " alpha"), _job(TEXT + " alpha")))
        await rt.queue.put(Capture(_draft("b", enrich_state="pending", text=TEXT + " beta", t=T0 + 20),
                                   _job(TEXT + " beta")))
        await rt.queue.put(Capture(_draft("c", source="heartbeat", text=TEXT + " gamma", t=T0 + 40)))
        for _ in range(200):
            rows = await store.q("SELECT * FROM observations WHERE episode_id IS NOT NULL")
            if len(rows) == 3:
                break
            await asyncio.sleep(0.01)
        assert len(rows) == 3
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert await drain_outbox(store) == 0  # nothing left to embed
    for r in rows:
        assert fake.embedded.count(observation_text(r)) == 1
    enriched = [r for r in rows if r["window_title"] in ("a", "b")]
    assert all(r["caption"] == "The user edits the plan." for r in enriched)  # embedded after the caption
