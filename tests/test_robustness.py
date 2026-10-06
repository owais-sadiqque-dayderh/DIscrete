"""Failure-mode regressions: support, atomicity, overload, and shared readers."""

from __future__ import annotations

import asyncio
import threading

import pytest

from discrete import llm
from discrete.capture.enrich import Enricher
from discrete.config import T
from discrete.evaluation.harness import Case, CaseResult
from discrete.process.pipeline import ProcessPipeline
from discrete.process.validation import verify_evidence
from discrete.store import Store
from tests.test_pipeline import _one_episode
from tests.test_runtime import _draft, _job


def test_genuine_but_unrelated_quote_is_not_evidence():
    row = {"screen_text": "Editing the Dooran logo in Illustrator", "caption": "", "window_title": ""}
    assert not verify_evidence("The user owns a yacht", "Editing the Dooran logo", row)[1]


def test_contamination_fails_answerable_and_unanswerable_cases():
    assert not CaseResult(Case("q"), 1, "grounded", 0, contaminated=True).correct
    assert not CaseResult(Case("q", unanswerable=True), None, "empty", 0, contaminated=True).correct


def test_supported_negative_answer_can_refute_unanswerable_question_premise():
    result = CaseResult(Case("Which flight did I book?", unanswerable=True), None, "grounded", 0,
                        answer_supported=True, answer_correct=True)
    assert result.correct


async def test_retrieval_support_rejects_related_calculation_as_salary(fake):
    from discrete.models import Hit, QueryPlan, RetrievalResult
    from discrete.retrieval.support import verify_support
    result = RetrievalResult("What is my salary?", QueryPlan("What is my salary?"),
                             [Hit("observation", 1, "Calculator", "250 times 12 equals 3000", strong=True)])
    fake.support_indices = []
    await verify_support(result)
    assert result.hits == []


async def test_retrieval_support_fails_closed_on_invalid_indices(fake):
    from discrete.models import Hit, QueryPlan, RetrievalResult
    from discrete.retrieval.support import verify_support
    result = RetrievalResult("Who owns it?", QueryPlan("Who owns it?"), [Hit("observation", 1, "Article", "Seen")])
    fake.support_indices = [999]
    await verify_support(result)
    assert result.hits == []


async def test_consolidation_rejects_unsupported_rewrite_without_replacing_inputs(store, fake):
    from discrete.models import ClaimDraft
    from discrete.process.consolidate import run_consolidation
    from discrete.process.embed_worker import drain_outbox
    eid, (oid,) = await _one_episode(store, ["I prefer concise notes with concrete examples."])
    ids = [await store.create_claim(ClaimDraft(kind="preference", text=text, episode_id=eid,
                evidence=[{"observation_id": oid, "quote": "I prefer concise notes"}]))
           for text in ("The user prefers concise notes.", "The user prefers concise technical notes.")]
    await drain_outbox(store)
    fake.consolidation = {"claims": [{"text": "The user prefers verbose notes.", "input_ids": ids}]}
    fake.verification = {0: False}
    await run_consolidation(store)
    assert {r["id"] for r in await store.live_claims(kind="preference")} == set(ids)


async def test_semantic_verifier_rejects_negation_and_preference_inferences(store, fake):
    eid, (oid,) = await _one_episode(store, ["I did not export the report. I read about dark themes."])
    fake.draft = {"title": "Reading", "summary": "Viewed a report.", "claims": [
        {"kind": "event", "text": "The user exported the report", "entities": [],
         "evidence": [{"observation_id": oid, "quote": "I did not export the report"}]},
        {"kind": "preference", "text": "The user prefers dark themes", "entities": [],
         "evidence": [{"observation_id": oid, "quote": "dark themes"}]},
    ]}
    fake.verification = {0: False, 1: False}
    out = await ProcessPipeline(store).draft_episode(eid)
    assert out is not None and out.rejected == {"not_entailed": 2}
    assert await store.live_claims() == []


async def test_draft_transaction_rolls_back_everything_then_retries_once(store, fake, monkeypatch):
    eid, (oid,) = await _one_episode(store, ["Editing the Dooran logo and reviewing its exported report"])
    fake.draft = {"title": "Logo", "summary": "Worked on a logo.", "claims": [
        {"kind": "event", "text": "The user edited the Dooran logo", "entities": [],
         "evidence": [{"observation_id": oid, "quote": "Editing the Dooran logo"}]},
        {"kind": "event", "text": "The user reviewed the exported report", "entities": [],
         "evidence": [{"observation_id": oid, "quote": "exported report"}]},
    ]}
    original = store._insert_claim
    calls = 0

    def fail(conn, *args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise llm.LLMError("injected second write failure")
        return original(conn, *args, **kw)

    monkeypatch.setattr(store, "_insert_claim", fail)
    assert await ProcessPipeline(store).draft_episode(eid) is None
    assert await store.live_claims() == []
    assert await store.scalar("SELECT COUNT(*) FROM evidence") == 0
    assert (await store.get_episode(eid))["status"] == "ready"
    monkeypatch.setattr(store, "_insert_claim", original)
    out = await ProcessPipeline(store).draft_episode(eid)
    assert out is not None and out.added == 2
    assert len(await store.live_claims()) == 2
    await store.draft_failed(eid)  # late cancellation must not re-open a committed draft
    assert (await store.get_episode(eid))["status"] == "closed"


async def test_cancelled_sql_worker_keeps_connection_lock(store):
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        release.wait(2)
        store.conn.execute("INSERT INTO meta VALUES ('worker-completed','yes')")

    first = asyncio.create_task(store._call(blocked))
    await asyncio.to_thread(entered.wait, 1)
    first.cancel()
    second = asyncio.create_task(store.set_meta("second", "yes"))
    await asyncio.sleep(0.02)
    assert not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert await store.get_meta("worker-completed") == "yes"


def test_ax_clipped_text_is_excluded_but_visible_children_are_walked(monkeypatch):
    from discrete.capture import accessibility as ax

    nodes = {"root": {"AXRole": "AXGroup", "AXTitle": "offscreen group", "AXPosition": (0, 50),
                     "AXSize": (100, 1), "AXChildren": ["offscreen", "visible"]},
             "offscreen": {"AXRole": "AXStaticText", "AXValue": "never viewed", "AXPosition": (0, 50), "AXSize": (100, 1)},
             "visible": {"AXRole": "AXStaticText", "AXValue": "actually viewed", "AXPosition": (0, 100), "AXSize": (100, 20)}}
    monkeypatch.setattr(ax, "_ax_get", lambda el, key: nodes[el].get(key))
    monkeypatch.setattr(ax, "_ax_value", lambda raw, kind: raw)
    parts = []
    ax._walk_text("root", parts, depth=0, max_nodes=10, deadline=float("inf"), seen=[0], viewport=(0, 0, 200, 200))
    assert parts == ["actually viewed"]


def test_ocr_reads_document_even_when_ax_chrome_exceeds_threshold(tmp_path, monkeypatch):
    from discrete.capture import accessibility as ax
    from discrete.capture import ocr
    from discrete.capture.screen import ScreenObserver
    monkeypatch.setenv("DISCRETE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ax, "read_screen_text", lambda *args, **kw: "Window chrome " * 30)
    monkeypatch.setattr(ocr, "ocr_image", lambda image: "Scanned document content")
    observer = ScreenObserver(queue=asyncio.Queue())
    text, source = observer._screen_text(ax.WindowContext(), None)
    assert text.startswith("Scanned document content") and source == "ax+ocr"
    observer.grabber.close()


async def test_bubble_shutdown_waits_for_cancelled_background_work(store):
    from discrete.bubble.overlay_client import Overlay
    from discrete.bubble.service import BubbleService
    service = BubbleService(store, overlay=Overlay(enabled=False))
    finished = asyncio.Event()

    async def worker():
        try:
            await asyncio.sleep(100)
        finally:
            await store.set_meta("shutdown-worker", "finished")
            finished.set()

    service._spawn(worker())
    await asyncio.sleep(0)
    await service.stop()
    assert finished.is_set() and await store.get_meta("shutdown-worker") == "finished"


async def test_enrichment_overload_stays_bounded_and_drains_persisted_rows(store, fake, monkeypatch):
    monkeypatch.setattr(T, "ENRICH_MAX_PENDING", 2)
    fake.caption_gate = asyncio.Event()
    enricher = Enricher(store)
    ids = []
    for i in range(10):
        oid, _ = await store.add_observation(_draft(title=str(i), enrich_state="pending"))
        ids.append(oid)
        enricher.submit(oid, _job())
    assert len(enricher._tasks) == 2
    assert len(await store.pending_enrichment()) == 10
    fake.caption_gate.set()
    for _ in range(200):
        await enricher.fill_pending()
        if not await store.pending_enrichment():
            break
        await asyncio.sleep(0.005)
    assert all([(await store.get_observation(i))["enrich_state"] == "done" for i in ids])


async def test_dense_cache_sees_other_process_writes_and_model_changes(store, tmp_db, monkeypatch):
    other = await Store.open(tmp_db)
    try:
        await store.save_embeddings([("observation", 1, [1.0, 0.0], "a")])
        assert (await other.dense_index("observation"))[0].tolist() == [1]
        await store.save_embeddings([("observation", 2, [0.0, 1.0], "b")])
        assert (await other.dense_index("observation"))[0].tolist() == [1, 2]
        monkeypatch.setenv("EMBED_LM_MODEL_NAME", "replacement")
        assert (await other.dense_index("observation"))[0].tolist() == []
    finally:
        await other.close()


async def test_observer_shutdown_awaits_owned_capture_tasks(tmp_path, monkeypatch):
    from discrete.aio import spawn
    from discrete.capture.screen import ScreenObserver
    monkeypatch.setenv("DISCRETE_DATA_DIR", str(tmp_path))
    observer = ScreenObserver(queue=asyncio.Queue())
    finished = asyncio.Event()

    async def worker():
        try:
            await asyncio.sleep(100)
        finally:
            finished.set()

    spawn(worker(), owner=observer._tasks)
    await asyncio.sleep(0)
    await observer.stop()
    assert finished.is_set() and not observer._tasks
    observer.arm("keyboard")  # queued late input cannot restart a stopped observer


async def test_failed_work_retries_after_cooldown_without_reopening_closed_episodes(store):
    from discrete.timeutil import now
    eid = await store.open_episode("retry-test", now())
    await store.execute("UPDATE episodes SET status='failed', draft_attempts=? WHERE id=?",
                        (T.DRAFT_MAX_ATTEMPTS, eid))
    await store.retry_failed_drafts(older_than_s=900)
    assert (await store.get_episode(eid))["status"] == "failed"
    await store.execute("UPDATE episodes SET updated_at=? WHERE id=?", (now() - 901, eid))
    await store.retry_failed_drafts(older_than_s=900)
    row = await store.get_episode(eid)
    # One more attempt per round: attempts are kept, so the next failure is final again.
    assert row["status"] == "ready" and row["draft_attempts"] == T.DRAFT_MAX_ATTEMPTS
    # Each later round waits twice as long.
    await store.execute("UPDATE episodes SET status='failed', draft_attempts=?, updated_at=? WHERE id=?",
                        (T.DRAFT_MAX_ATTEMPTS + 2, now() - 901 * 3, eid))
    await store.retry_failed_drafts(older_than_s=900)
    assert (await store.get_episode(eid))["status"] == "failed"
    await store.execute("UPDATE episodes SET updated_at=? WHERE id=?", (now() - 901 * 4, eid))
    await store.retry_failed_drafts(older_than_s=900)
    assert (await store.get_episode(eid))["status"] == "ready"
    # `process --recover` resets attempts outright.
    await store.execute("UPDATE episodes SET status='failed', updated_at=? WHERE id=?", (now(), eid))
    await store.retry_failed_drafts()
    row = await store.get_episode(eid)
    assert row["status"] == "ready" and row["draft_attempts"] == 0
    await store.execute("UPDATE episodes SET status='closed', updated_at=0 WHERE id=?", (eid,))
    await store.retry_failed_drafts()
    assert (await store.get_episode(eid))["status"] == "closed"
    await store.execute("INSERT INTO index_outbox(ref_kind, ref_id, enqueued_at, attempts) VALUES ('episode',?, ?,?)",
                        (eid, now(), T.OUTBOX_MAX_ATTEMPTS))
    await store.retry_failed_outbox(older_than_s=900)
    assert (await store.q1("SELECT attempts FROM index_outbox"))[0] == T.OUTBOX_MAX_ATTEMPTS
    await store.execute("UPDATE index_outbox SET enqueued_at=?", (now()-901,))
    await store.retry_failed_outbox(older_than_s=900)
    assert (await store.q1("SELECT attempts FROM index_outbox"))[0] == 0


async def test_incomplete_merge_decisions_do_not_silently_add_conflicting_preferences(store, monkeypatch):
    import discrete.process.pipeline as pipeline
    from discrete.models import ClaimDraft
    from discrete.process.pipeline import _Durable
    durable = _Durable(ClaimDraft(kind="preference", text="The user prefers detailed notes."),
                       text="The user prefers detailed notes.", candidates=[{"id": 7, "text": "The user prefers concise notes."}])

    async def incomplete(*args, **kwargs):
        return {"relations": []}

    monkeypatch.setattr(pipeline, "complete_json", incomplete)
    with pytest.raises(llm.LLMError, match="incomplete decisions"):
        await ProcessPipeline(store)._relate([durable])
    assert not durable.relations


def test_failed_known_window_capture_never_substitutes_display_pixels(monkeypatch):
    from discrete.capture.frames import FrameGrabber
    grabber = FrameGrabber()
    grabber._sck = object()
    monkeypatch.setattr(grabber, "_grab_window", lambda *args: None)
    def display():
        pytest.fail("a known window failure must not capture unrelated display pixels")
    monkeypatch.setattr(grabber, "_mss", display)
    assert grabber.grab((0, 0, 100, 100), max_width=100, window_id=7) is None
    grabber.close()


async def test_all_required_expectations_work_without_expect_any(store, monkeypatch):
    import discrete.evaluation.harness as harness
    from discrete.models import Hit, QueryPlan, RetrievalResult

    async def retrieved(*args, **kwargs):
        return RetrievalResult("q", QueryPlan("q"),
                               [Hit("observation", 1, "Compare search quality; evaluate PostgreSQL", "", strong=True)])

    monkeypatch.setattr(harness, "retrieve", retrieved)
    result = await harness.run_case(store, Case("q", expect_all=["search quality", "PostgreSQL"]))
    assert result.rank == 1 and not result.missing_required and result.correct


# ------------------------------------------------- local adaptations (event-driven runtime)

async def test_segmentation_waits_for_the_oldest_caption_and_resumes_backlog_from_disk(store, fake, monkeypatch):
    from discrete.app import Runtime
    from discrete.capture.enrich import Capture
    from tests.test_runtime import T0, TEXT

    monkeypatch.setattr(T, "ENRICH_MAX_PENDING", 1)
    fake.caption = {"caption": "The user edits the plan.", "page_title": "Plan", "useful": True}
    fake.caption_gate = asyncio.Event()
    rt = Runtime(store, bubble=False)
    tasks = [asyncio.create_task(rt._ingest_loop()), asyncio.create_task(rt._segment_loop())]
    try:
        await rt.queue.put(Capture(_draft("a", enrich_state="pending", text=TEXT + " alpha"), _job(TEXT + " alpha")))
        await rt.queue.put(Capture(_draft("b", enrich_state="pending", text=TEXT + " beta", t=T0 + 20),
                                   _job(TEXT + " beta")))
        await rt.queue.put(Capture(_draft("c", source="heartbeat", text=TEXT + " gamma", t=T0 + 40)))
        await asyncio.sleep(0.1)
        # The pool holds one caption; "b" waits on disk, and nothing overtakes "a".
        assert len(rt.enricher._active) == 1
        assert await store.q("SELECT id FROM observations WHERE episode_id IS NOT NULL") == []
        fake.caption_gate.set()
        for _ in range(300):
            rows = await store.q("SELECT * FROM observations WHERE episode_id IS NOT NULL ORDER BY id")
            if len(rows) == 3:
                break
            await asyncio.sleep(0.01)
        assert [r["window_title"] for r in rows] == ["a", "b", "c"]
        assert all(r["caption"] == "The user edits the plan." for r in rows[:2])
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_shutdown_leaves_caption_pending_and_next_run_finishes_it(store, fake):
    fake.caption = {"caption": "The user edits the plan.", "page_title": "Plan", "useful": True}
    fake.caption_gate = asyncio.Event()
    oid, _ = await store.add_observation(_draft("a", enrich_state="pending"))
    first = Enricher(store)
    assert first.submit(oid, _job()) is not None
    await asyncio.sleep(0)
    await first.cancel_all()
    assert (await store.get_observation(oid))["enrich_state"] == "pending"
    assert first.submit(oid, _job()) is None  # no new model calls once shutting down
    fake.caption_gate.set()
    nxt = Enricher(store)
    assert await nxt.fill_pending() == 1
    await asyncio.gather(*nxt._tasks)
    assert (await store.get_observation(oid))["enrich_state"] == "done"


async def test_unexpected_enrichment_error_never_leaves_a_row_pending(store, fake, monkeypatch):
    oid, _ = await store.add_observation(_draft("a", enrich_state="pending"))
    enricher = Enricher(store)

    async def broken(*_a, **_k):
        raise RuntimeError("bug")

    monkeypatch.setattr(enricher, "_apply", broken)
    await enricher.submit(oid, _job())
    assert (await store.get_observation(oid))["enrich_state"] == "failed"


async def test_day_digest_reaches_the_writer_only_when_it_supports_the_answer(fake, monkeypatch):
    from discrete.models import Hit, QueryPlan, RetrievalResult
    from discrete.retrieval.support import verify_support

    def result():
        return RetrievalResult("What did I do on Monday?", QueryPlan("What did I do on Monday?", intent="temporal"),
                               [Hit("episode", 1, "10:00 Plan", "Edited the plan")], digest="Monday: planning, salary maths")

    r = result()
    fake.support_indices = [0]
    await verify_support(r)
    assert [h.ref_id for h in r.hits] == [1] and r.hits[0].strong and r.digest == ""
    r = result()
    fake.support_indices = None  # everything supports, the digest (index 1) included
    await verify_support(r)
    assert r.digest
    monkeypatch.setattr(T, "SUPPORT_CHECK", False)
    r = result()
    fake.support_indices = []
    await verify_support(r)
    assert len(r.hits) == 1 and r.digest


def test_only_ocr_lines_missing_from_ax_text_are_added():
    from discrete.capture.screen import _novel_lines

    ax_text = "SQLite FTS5 Extension\nFTS5 is an SQLite virtual table module"
    assert _novel_lines("SQLite FTS5 Extension\nFTS5 is an SQLite virtual table\nScanned invoice total", ax_text) == [
        "Scanned invoice total"]


async def test_slow_support_check_leaves_results_unverified_instead_of_stalling(fake, monkeypatch):
    import discrete.retrieval.support as support
    from discrete.models import Hit, QueryPlan, RetrievalResult

    async def slow(*_a, **_k):
        await asyncio.sleep(10)

    monkeypatch.setattr(T, "SUPPORT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(support, "_supported", slow)
    r = RetrievalResult("q", QueryPlan("q"), [Hit("observation", 1, "Notes", "Budget review")])
    await support.verify_support(r)
    assert [h.ref_id for h in r.hits] == [1] and not r.hits[0].strong


async def test_a_provider_glitch_is_retried_but_a_wrong_model_is_not(fake, monkeypatch):
    import httpx
    import openai

    from discrete import llm
    from discrete.config import T, insight_endpoint

    monkeypatch.setattr(T, "LLM_PROVIDER_RETRY_S", 0.0)
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def err(body: str) -> openai.NotFoundError:
        return openai.NotFoundError(body, response=httpx.Response(404, request=req), body=None)

    glitch = err("Error code: 404 - {'error': {'message': 'Provider returned error', 'code': 404, "
                 "'metadata': {'raw': 'model_not_found', 'provider_name': 'Meta'}}}")
    calls = []

    async def flaky(ep, messages, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise glitch
        return "ok"

    monkeypatch.setattr(fake, "chat", flaky)
    assert await llm.complete(insight_endpoint(), [{"role": "user", "content": "x"}]) == "ok"
    assert len(calls) == 3

    async def wrong(ep, messages, **kw):
        calls.append(1)
        raise err("Error code: 404 - {'error': {'message': 'No endpoints found for meta/does-not-exist'}}")

    calls.clear()
    monkeypatch.setattr(fake, "chat", wrong)
    try:
        await llm.complete(insight_endpoint(), [{"role": "user", "content": "x"}])
    except openai.NotFoundError:
        pass
    assert len(calls) == 1


def test_discretes_own_output_is_not_the_users_activity():
    from discrete.capture.suppress import classify_origin

    assert classify_origin("Terminal", 'jarvis ask "which browser did I use?" — zsh', "Helium") == "agent"
    assert classify_origin("DB Browser for SQLite", "DB Browser for SQLite - /Users/x/.cache/discrete/discrete.db",
                           "claims") == "agent"
    records = "[claim 32 · event · engaged · Mon]\\nThe user ...\\n[screen 95 · Mon 22:22 · Helium Setup]"
    assert classify_origin("Terminal", "zsh", records) == "agent"
    # The user's own project named Jarvis is ordinary activity.
    assert classify_origin("Arc", "Jarvis roadmap - Google Docs", "Jarvis plan for October") == "user"
