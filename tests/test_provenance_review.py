"""Ports of the 669a298 review repros; synthetic data, fake models and keys only."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from discrete import encryption as enc
from discrete.capture import accessibility as ax
from discrete.capture import privacy, screen
from discrete.capture.enrich import Enricher, EnrichJob, backfill_page_fields
from discrete.config import T
from discrete.models import ObservationDraft, ProvenanceEdge, RetrievalResult
from discrete.process.provenance import build_provenance, infer_edges, valid_edge
from discrete.retrieval.grounding import evaluate
from discrete.retrieval.hybrid import expand_why
from discrete.retrieval.query import fallback_plan
from discrete.retrieval.why import render_chain, resolve_why
from discrete.store import Store

T0 = 1_790_000_000.0
SECRET = "sk-proj-" + "A" * 24


async def add(store, offset, title, **kwargs):
    return (await store.add_observation(ObservationDraft(
        started_at=T0 + offset, ended_at=T0 + offset + 0.01, app="Browser",
        window_title=title, page_title=title, caption=title, session_id="s", **kwargs)))[0]


def context(url):
    return ax.WindowContext(app="Arc", bundle_id="arc", window_title=url, url=url, pid=7)


async def test_denied_frame_is_never_grabbed_or_cached(store, monkeypatch):
    monkeypatch.setattr(privacy, "_SITES", ("denied.test",))
    monkeypatch.setattr(T, "FRAME_CACHE_S", 0.001)
    monkeypatch.setattr(ax, "window_context", lambda: context("https://denied.test/"))
    observer = screen.ScreenObserver(queue=asyncio.Queue())
    grabs = []
    monkeypatch.setattr(observer.grabber, "grab", lambda *a, **k: grabs.append(1) or Image.new("RGB", (64, 64), "red"))
    task = asyncio.create_task(observer._frame_loop())
    try:
        await asyncio.sleep(0.03)
        assert observer._latest is None and not observer._recent and grabs == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        observer.grabber.close()


async def test_denied_cached_before_is_not_forwarded_to_allowed_vision_job(store, monkeypatch):
    monkeypatch.setattr(privacy, "_SITES", ("denied.test",))
    queue = asyncio.Queue()
    observer = screen.ScreenObserver(queue=queue)
    observer.ep = SimpleNamespace(configured=True)
    monkeypatch.setattr(observer, "_save", lambda *args: "")
    monkeypatch.setattr(observer, "_screen_text", lambda *args: ("Allowed article content with enough safe words.", "ax"))
    denied = context("https://denied.test/")
    before, after = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
    pending = {"kind": "keyboard", "started": time.time(), "before_ctx": denied,
               "counts": {"keys": 1, "clicks": 0, "scrolls": 0, "clipboard": 0}}
    try:
        await observer._capture(pending, context("https://allowed.test/"), before, after)
        assert queue.get_nowait().job.frames == [after]
    finally:
        observer.grabber.close()


async def test_arm_discards_denied_cache_after_policy_change(store, monkeypatch):
    observer = screen.ScreenObserver(queue=asyncio.Queue())
    observer._loop = asyncio.get_running_loop()
    observer._latest = (Image.new("RGB", (64, 64)), context("https://denied.test/"))
    monkeypatch.setattr(privacy, "_SITES", ("denied.test",))
    monkeypatch.setattr(ax, "window_context", lambda: context("https://allowed.test/"))
    try:
        observer.arm("keyboard")
        assert observer._pending["before_img"] is None
    finally:
        await observer.stop()


async def test_all_model_fields_are_redacted_before_storage_and_inference(store):
    ids = [await add(store, dt, title, screen_text="Safe screen content sufficient for an observation.", enrich_state="pending")
           for dt, title in [(0, "First window"), (20, "Second window")]]
    job = EnrichJob([], "Browser", "First window", "", "safe screen text" * 4,
                    {"keys": 0, "clicks": 0, "scrolls": 0, "clipboard": 0})
    enricher = Enricher(store)
    await enricher._apply(ids[0], job, {"caption": f"Shown {SECRET}", "page_title": f"Launch {SECRET}",
                                      "entities": [f"Entity {SECRET}"], "useful": True})
    await enricher._apply(ids[1], job, {"caption": "Second safe caption", "page_title": "Second page",
                                      "navigated_from": f"Launch {SECRET}", "useful": True})
    rows = [dict(await store.get_observation(oid)) for oid in ids]
    assert SECRET not in str(rows)
    edges = infer_edges(await store.attention_moments(T0, T0 + 30))
    assert SECRET not in str(edges)
    assert SECRET not in render_chain(await resolve_why(store, "why did I open Second page?"))


async def test_page_field_backfill_redacts_model_fields(store, fake):
    oid, _ = await store.add_observation(ObservationDraft(
        screen_text="Safe content", caption="A safe caption", app="Browser", window_title="First window", session_id="s"))
    fake.caption = {"page_title": f"Launch {SECRET}", "navigated_from": f"Source {SECRET}"}
    assert (await backfill_page_fields(store))["filled"] == 1
    assert SECRET not in str(dict(await store.get_observation(oid)))


async def test_legacy_unredacted_provenance_quotes_are_rejected(store):
    await add(store, 0, "Reading list", screen_text=f"Second page {SECRET}")
    await add(store, 20, "Second page")
    moments = await store.attention_moments(T0, T0 + 30)
    edges = infer_edges(moments)
    assert SECRET not in str(edges)
    unsafe = ProvenanceEdge(moments[0].sighting_id, moments[1].sighting_id, "navigation", 0.9,
                            "screen_text", moments[0].fields["screen_text"], "page_title", "Second page", 20)
    assert not valid_edge(unsafe, {m.sighting_id: m for m in moments})


async def test_custom_encrypted_store_frames_follow_owner_with_flag_off(tmp_path, monkeypatch, fake):
    pytest.importorskip("sqlcipher3")
    from discrete.app import Runtime

    class FakeKeys:
        def __init__(self):
            self.keys = {}

        def get(self, key_id):
            return self.keys[key_id]

        def put(self, key_id, key):
            self.keys[key_id] = key

    monkeypatch.setattr(enc, "_keychain", FakeKeys())
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "true")
    path = tmp_path / "custom.db"
    store = await Store.open(str(path))
    await store.close()
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "false")
    store = await Store.open(str(path))
    runtime = Runtime(store, bubble=False)
    try:
        frame = runtime.observer._save(Image.new("RGB", (64, 64), "red"), "after")
        assert (await asyncio.to_thread(Path(frame).read_bytes)).startswith(enc.FRAME_MAGIC)
        with enc.open_frame(frame) as image:
            assert image.size == (64, 64)
        assert not (tmp_path / "data" / "discrete.db").exists()
    finally:
        runtime.observer.grabber.close()
        await store.close()


def test_frame_write_without_owner_fails_closed(tmp_path):
    frame = tmp_path / "unowned.jpg"
    with pytest.raises(enc.EncryptionError, match="owning store"):
        enc.write_frame(frame, Image.new("RGB", (4, 4)), quality=65)
    assert not frame.exists()


async def test_same_host_succession_is_context_below_causal_threshold(store):
    for dt, title, path in [(0, "Payroll dashboard", "payroll"), (20, "Private health report", "health"),
                            (40, "Billing settings", "billing")]:
        await add(store, dt, title, url=f"https://example.test/{path}")
    edges = infer_edges(await store.attention_moments(T0, T0 + 41))
    assert all(e.kind == "same_site_context" and e.confidence < T.WHY_CHAIN_MIN_CONFIDENCE for e in edges)
    assert not (await resolve_why(store, "why did I open Billing settings?")).edges


async def test_model_referrer_requires_independent_corroboration(store):
    await add(store, 0, "Payroll dashboard", url="https://payroll.test/", screen_text="Screen showing Payroll dashboard")
    await add(store, 20, "Cooking article", url="https://recipes.test/soup", navigated_from="Payroll dashboard")
    edges = infer_edges(await store.attention_moments(T0, T0 + 21))
    assert all(e.confidence < T.WHY_CHAIN_MIN_CONFIDENCE for e in edges)
    assert not (await resolve_why(store, "why did I open Cooking article?")).edges


@pytest.mark.parametrize("signal", ["source_link", "referrer_url", "matched_click"])
async def test_corroborated_navigation_retains_a_causal_edge(store, signal):
    source_url = "https://research.test/quartz"
    target_url = "https://reports.test/quartz"
    await add(store, 0, "Quartz article", url=source_url,
              screen_text=target_url if signal == "source_link" else "Quartz article",
              clicks=1 if signal == "matched_click" else 0)
    if signal == "matched_click":
        await store.execute("UPDATE observations SET caption=? WHERE id=1", (f"Clicked {target_url}",))
    if signal == "referrer_url":
        target_url += "?referrer=" + source_url
    await add(store, 20, "Quartz report", url=target_url, navigated_from="Quartz article")
    chain = await resolve_why(store, "Why did I open Quartz report?")
    assert chain.edges and all(e.confidence >= T.WHY_CHAIN_MIN_CONFIDENCE for e in chain.edges)


async def test_unrelated_interaction_does_not_corroborate_a_model_referrer(store):
    await add(store, 0, "Payroll dashboard", url="https://payroll.test/", clicks=1)
    await add(store, 20, "Cooking article", url="https://recipes.test/soup", keys=4, navigated_from="Payroll dashboard")
    assert not (await resolve_why(store, "Why did I open Cooking article?")).edges


async def test_visible_path_link_survives_an_intervening_same_host_visit(store):
    first = await add(store, 0, "Landing", url="https://example.test/", screen_text="Download")
    await add(store, 20, "Payroll dashboard", url="https://example.test/payroll")
    await add(store, 40, "Installer", url="https://example.test/download")
    chain = await resolve_why(store, "Why did I open Installer?")
    assert chain.moments[0].observation_id == first


async def test_same_sitting_return_prefers_first_arrival_and_original_cause(store):
    origin = await add(store, 0, "Saffron launch post", screen_text="Mina shared https://reports.test/saffron")
    arrival = await add(store, 20, "Saffron report", url="https://reports.test/saffron")
    await add(store, 2700, "Saffron followup post", screen_text="Nora shared https://reports.test/saffron")
    await add(store, 2720, "Saffron report", url="https://reports.test/saffron")
    chain = await resolve_why(store, "why did I end up reading Saffron report?")
    assert chain.effect.at == T0 + 20
    assert [m.observation_id for m in chain.moments] == [origin, arrival]


async def test_separate_sittings_are_ambiguous_without_context_and_use_question_time(store):
    await add(store, 0, "Saffron research", screen_text="Saffron report")
    await add(store, 20, "Saffron report", url="https://reports.test/saffron")
    await add(store, 86400, "Saffron launch post", screen_text="Mina shared https://reports.test/saffron")
    await add(store, 86420, "Saffron report", url="https://reports.test/saffron")
    await add(store, 86460, "Saffron report", url="https://reports.test/saffron")
    question = "why did I open Saffron report?"
    chain = await resolve_why(store, question)
    assert chain.effect is None and not chain.edges
    assert [m.at for m in chain.candidates] == [T0 + 20, T0 + 86420]
    assert "Which visit" in render_chain(chain)
    assert (await resolve_why(store, question, reference_at=T0 + 30)).effect.at == T0 + 20
    assert (await resolve_why(store, question, reference_at=T0 + 86480)).effect.at == T0 + 86420


@pytest.mark.parametrize("cue", ["again", "the last time", "most recently", "this time"])
async def test_repeat_cues_select_latest_arrival_in_latest_sitting(store, cue):
    for dt in (20, 40, 86420, 86460):
        await add(store, dt, "Saffron report", url="https://reports.test/saffron")
    chain = await resolve_why(store, f"why did I open Saffron report {cue}?")
    assert chain.effect.at == T0 + 86460 and not chain.candidates


async def test_sitting_gap_is_configurable_and_applies_to_consecutive_matches(store, monkeypatch):
    monkeypatch.setattr(T, "WHY_SITTING_GAP_S", 30)
    for dt in (0, 30, 60, 91, 110):
        await add(store, dt, "Saffron report", url="https://reports.test/saffron")
    question = "why did I open Saffron report?"
    assert (await resolve_why(store, question, reference_at=T0 + 70)).effect.at == T0
    assert (await resolve_why(store, question, reference_at=T0 + 120)).effect.at == T0 + 91
    assert [m.at for m in (await resolve_why(store, question)).candidates] == [T0, T0 + 91]


async def test_question_time_filters_visits_before_the_effect_pool_limit(store):
    for i in range(T.WHY_EFFECT_POOL + 20):
        await add(store, i, "Repeated article", screen_text="A stable repeated article.")
    chain = await resolve_why(store, "Why did I open Repeated article?", reference_at=T0 + 5)
    assert chain.effect is not None and chain.effect.at == T0
    latest = await resolve_why(store, "Why did I open Repeated article again?", reference_at=T0 + 5)
    assert latest.effect.at == T0 + 5


async def test_latest_effect_survives_the_candidate_pool_limit(store, monkeypatch):
    monkeypatch.setattr(T, "WHY_EFFECT_POOL", 2)
    ids = [await add(store, i, "Quartz report", url=f"https://reports.test/{i}",
                     screen_text=" ".join(f"distinct{i}word{k}" for k in range(20))) for i in range(5)]
    assert len(set(ids)) == 5
    chain = await resolve_why(store, "Why did I open Quartz report again?")
    assert chain.effect.at == T0 + 4


async def test_equal_time_effects_return_ambiguity_and_disable_chain_assertion(store):
    from discrete.bubble.engine import BubbleEngine, Prepared

    await add(store, 0, "Index", screen_text="Saffron report")
    await add(store, 20, "Saffron report", url="https://reports.test/a")
    await add(store, 20, "Saffron report", url="https://reports.test/b")
    question = "why did I open Saffron report?"
    result = RetrievalResult(question, fallback_plan(question))
    await expand_why(store, result)
    chain = result.why_chain
    assert chain.effect is None and len(chain.candidates) == 2 and not chain.edges
    assert "ambiguous" in render_chain(chain) and "which" in render_chain(chain).lower()
    assert not result.hits and evaluate(result).mode == "ambiguous"
    prepared = Prepared(question, question, "Browser", "text field", result, evaluate(result))
    prompt = BubbleEngine(store).messages(prepared)[-1]["content"]
    assert "2026-" in prompt and "Ask which visit" in prompt


async def test_start_browsing_anchors_arrival_in_latest_relevant_burst(store):
    for dt in (0, 86400):
        await add(store, dt, "Cobalt launch post", screen_text="https://registry.test/models")
        await add(store, dt + 20, "Cobalt catalogs", url="https://registry.test/models")
        await add(store, dt + 40, "Settings", url="https://registry.test/settings", screen_text="Cobalt catalogs")
    chain = await resolve_why(store, "Why did I start browsing cobalt catalogs on registry.test?",
                              reference_at=T0 + 86460)
    assert chain.effect.at == T0 + 86420
    assert chain.moments[0].at == T0 + 86400


def test_effect_without_temporal_cues_skips_date_detection(monkeypatch):
    from discrete.retrieval import query

    calls = []

    def unnecessary(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(query, "search_dates", unnecessary)
    assert fallback_plan("Why did I open Cobalt catalogs?").interval is None
    assert not calls


async def test_empty_provenance_checkpoint_is_not_reinferred(store, monkeypatch):
    from discrete.retrieval import why

    await add(store, 0, "Payroll dashboard")
    await add(store, 20, "Cooking article")
    assert await build_provenance(store, backfill=True) == 0

    def unnecessary(*args, **kwargs):
        raise AssertionError("fresh empty checkpoints are already processed")

    monkeypatch.setattr(why, "infer_edges", unnecessary)
    assert not (await resolve_why(store, "Why did I open Cooking article?")).edges


async def test_dense_provenance_backfill_and_why_have_bounded_cost(store, monkeypatch):
    from discrete.retrieval import why

    for i in range(2000):
        await add(store, i * 0.1, f"Subject{i}", url=f"https://example.test/item/{i}",
                  screen_text=f"Unique subject{i} content")
    inferred = []
    original_infer = why.infer_edges

    def counted_infer(moments, *, target_id=None, **kwargs):
        inferred.append(target_id)
        return original_infer(moments, target_id=target_id, **kwargs)

    monkeypatch.setattr(why, "infer_edges", counted_infer)
    # Cold localization may initialize date detection for the numeric title.
    # Provenance must infer only destinations actually walked.
    start = time.perf_counter()
    chain = await resolve_why(store, "why did I open Subject1999?")
    assert time.perf_counter() - start < 1.5
    assert not chain.edges
    assert inferred == [chain.effect.sighting_id]
    reads = 0
    original = store.attention_moments

    async def counted(*args):
        nonlocal reads
        reads += 1
        return await original(*args)

    monkeypatch.setattr(store, "attention_moments", counted)
    start = time.perf_counter()
    await build_provenance(store, backfill=True)
    assert time.perf_counter() - start < 3.0
    assert reads == 1  # shared extraction across all batches
    def no_inference(*args, **kwargs):
        raise AssertionError("backfilled zero-cause destinations must not be re-inferred")

    monkeypatch.setattr(why, "infer_edges", no_inference)
    start = time.perf_counter()
    assert not (await resolve_why(store, "why did I open Subject1999?")).edges
    assert time.perf_counter() - start < 0.3
