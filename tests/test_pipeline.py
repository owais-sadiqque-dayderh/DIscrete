"""Behavioural tests for the Stage 1-3 guards. Each names the insights.md entry it pins."""

from __future__ import annotations

import re
import sys

import pytest

from discrete import textkit as tk
from discrete.capture import prompts as p1
from discrete.models import ClaimDraft, ObservationDraft
from discrete.process import prompts as p2
from discrete.process.consolidate import run_consolidation
from discrete.process.embed_worker import drain_outbox
from discrete.process.entities import promote_recurring
from discrete.process.episodes import EpisodeTracker
from discrete.process.pipeline import ProcessPipeline
from discrete.retrieval.grounding import evaluate
from discrete.retrieval.hybrid import retrieve
from discrete.retrieval.query import build_plan, parse_interval

T0 = 1_789_000_000.0


def obs(app="Code", site="", title="main.py", text="", caption="", t=T0, clicks=1, session="s1", **kw):
    return ObservationDraft(app=app, site=site, window_title=title, screen_text=text, caption=caption,
                            started_at=t, ended_at=t + 5, clicks=clicks, session_id=session, **kw)


async def add(store, tracker, draft):
    oid, new = await store.add_observation(draft)
    assert new
    await tracker.on_observation(oid, await store.get_observation(oid), session_id=draft.session_id)
    return oid


# ------------------------------------------------------------------- Stage 1

@pytest.mark.display
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_capture_modules_import_and_find_a_window():
    """§7.11: a missing pyobjc constant silently disabled all AX."""
    from discrete.capture import accessibility as ax

    assert ax._HAS_QUARTZ
    assert isinstance(ax.window_context(), ax.WindowContext)


@pytest.mark.display
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_ocr_runs_in_accurate_mode():
    """§7.12: level 1 is *fast*, which garbled every capture."""
    from PIL import Image, ImageDraw, ImageFont

    from discrete.capture import ocr

    assert ocr._ACCURATE == 0
    img = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(img).text((20, 30), "Quarterly archive extension", fill="black",
                             font=ImageFont.load_default(size=48))
    assert "archive" in ocr.ocr_image(img).lower()


async def test_navigation_boundary_captures_each_article(monkeypatch):
    """§7.32: a burst of navigations was coalesced into one before/after; each
    window change must finalize the article in flight and open a fresh pair."""
    import asyncio

    from discrete.capture import accessibility as ax
    from discrete.capture import screen as sc

    def wc(title):
        return ax.WindowContext(app="Safari", bundle_id="com.apple.Safari",
                                window_title=title, url=f"https://news.ycombinator.com/{title}", pid=42)

    articles = ["a1", "a2", "a3"]
    state = {"i": 0}
    monkeypatch.setattr(sc.ax, "window_context", lambda: wc(articles[min(state["i"], len(articles) - 1)]))

    observer = sc.ScreenObserver(queue=asyncio.Queue())
    observer._loop = asyncio.get_running_loop()
    monkeypatch.setattr(observer.grabber, "grab", lambda *a, **k: None)
    finalized: list[str] = []

    async def fake_finalize(pending, ctx, after, *, stale):
        finalized.append(pending["before_ctx"].window_title)

    monkeypatch.setattr(observer, "_finalize", fake_finalize)

    for i in range(len(articles)):
        state["i"] = i
        observer.arm("window_change")  # each new article resets the debounce today
    await observer._flush()            # the last article is still pending
    await asyncio.sleep(0)             # let the rotated finalize tasks run
    if observer._timer:
        observer._timer.cancel()

    assert sorted(finalized) == articles  # one capture per article, not one for the burst


def test_self_window_is_agent_origin():
    from discrete.capture.suppress import classify_origin

    assert classify_origin("Terminal", "DIscrete — python3 .venv/bin/discrete run — 120x30", "logs") == "agent"
    assert classify_origin("Code", "store.py — DIscrete", "code") == "user"


def test_prompts_have_no_literal_schema_values():
    """§7.5: models copy literal example values."""
    for text in (p2.DRAFT_EPISODE_PROMPT, p2.RELATE_PROMPT, p2.CONSOLIDATE_PROMPT, p2.ENTITY_PAGE_PROMPT, p1.CAPTURE_PROMPT):
        assert not re.search(r'":\s*-?\d', text), text[:60]


# ------------------------------------------------------------------- Stage 2

async def test_dedup_same_app_only_and_extends_interval(store):
    long_text = " ".join(f"word{i}" for i in range(80))
    a, _ = await store.add_observation(obs(text=long_text, t=T0))
    b, new = await store.add_observation(obs(text=long_text + " tail", t=T0 + 30))
    assert not new and b == a
    row = await store.get_observation(a)
    assert row["dup_count"] == 2 and row["ended_at"] == T0 + 35
    _, new_other_app = await store.add_observation(obs(app="Safari", text=long_text + " tail2", t=T0 + 60))
    assert new_other_app


async def test_brief_alt_tab_is_absorbed(store):
    tr = EpisodeTracker(store)
    for i, app in enumerate(["Code", "Code", "Slack", "Code"]):
        await add(store, tr, obs(app=app, text=f"text {i} {app}", t=T0 + 10 * i))
    await tr.close_current()
    eps = await store.q("SELECT id FROM episodes")
    assert len(eps) == 1


async def test_sustained_switch_splits_retroactively(store):
    tr = EpisodeTracker(store)
    ids = []
    for i, (app, dt) in enumerate([("Code", 0), ("Code", 10), ("Slack", 20), ("Slack", 60), ("Slack", 130)]):
        ids.append(await add(store, tr, obs(app=app, text=f"text {i} {app}", t=T0 + dt)))
    await tr.close_current()
    rows = await store.q("SELECT id, episode_id FROM observations ORDER BY id")
    ep = {r["id"]: r["episode_id"] for r in rows}
    assert ep[ids[0]] == ep[ids[1]]
    assert ep[ids[2]] == ep[ids[3]] == ep[ids[4]] != ep[ids[0]]  # boundary at the switch point
    first = await store.get_episode(ep[ids[0]])
    assert first["ended_at"] == T0 + 15  # interval excludes the moved observations


async def test_orphaned_open_episode_is_recovered(store):
    """§7.16: episodes were left open forever when the process stopped."""
    tr = EpisodeTracker(store)
    await add(store, tr, obs(text="hello"))
    assert await store.recover_open_episodes() == [tr.state.episode_id]


async def _one_episode(store, texts, **kw):
    tr = EpisodeTracker(store)
    ids = [await add(store, tr, obs(text=t, t=T0 + 10 * i, **kw)) for i, t in enumerate(texts)]
    (eid,) = await tr.close_current()
    return eid, ids


async def test_draft_keeps_only_verified_claims(store, fake):
    """§7.17: missing/garbage evidence bypassed quote verification."""
    eid, (o1, o2) = await _one_episode(store, ["Editing the Dooran logo in Adobe Illustrator artboard",
                                               "Exported dooran-final.svg for the client"])
    fake.draft = {"title": "Logo work", "summary": "Worked on the Dooran logo.", "claims": [
        {"kind": "event", "text": "The user edited the Dooran logo in Adobe Illustrator", "entities": ["Dooran", "Adobe Illustrator"],
         "evidence": [{"observation_id": o1, "quote": "Editing the Dooran logo"}]},
        {"kind": "event", "text": "The user exported the final Dooran file", "entities": ["Dooran"],
         "evidence": [{"observation_id": o2, "quote": "a quote that is not there"}]},
        {"kind": "fact", "text": "The user owns a boat", "entities": [], "evidence": []},
        {"kind": "fact", "text": "The user likes jazz", "entities": [],
         "evidence": [{"observation_id": 99999, "quote": "jazz"}]},
    ]}
    out = await ProcessPipeline(store).draft_episode(eid)
    assert out.added == 2 and out.rejected == {"unsupported": 2}
    for c in await store.live_claims():
        for e in await store.evidence_for_claim(c["id"]):
            body = (await store.get_observation(e["observation_id"]))["screen_text"]
            assert tk.norm_key(e["quote"]) in tk.norm_key(body)
        assert 0 < c["confidence"] < 1 and c["modality"] == "engaged"
    assert (await store.get_episode(eid))["status"] == "closed"


async def test_mentions_need_name_and_recurrence(store, fake):
    """§7.18: batch-level mentions and immediate entity creation."""
    for day in range(2):
        tr = EpisodeTracker(store)
        oid = await add(store, tr, obs(text=f"Grainient gradient export {day}", t=T0 + day * 86400))
        (eid,) = await tr.close_current()
        fake.draft = {"title": "x", "summary": "y", "claims": [
            {"kind": "event", "text": f"The user exported a gradient from Grainient {day}",
             "entities": ["Grainient", "Figma"], "evidence": [{"observation_id": oid, "quote": "Grainient gradient"}]}]}
        await ProcessPipeline(store).draft_episode(eid)
        if day == 0:
            assert await store.q("SELECT * FROM entities") == []
    await promote_recurring(store)
    ents = {r["name"]: r["id"] for r in await store.q("SELECT * FROM entities")}
    assert "Grainient" in ents and "Figma" in ents  # Figma: named twice by the drafter
    grain_mentions = await store.q("SELECT * FROM mentions WHERE entity_id=?", (ents["Grainient"],))
    figma_mentions = await store.q("SELECT * FROM mentions WHERE entity_id=?", (ents["Figma"],))
    assert len(grain_mentions) == 2 and figma_mentions == []  # Figma never appears in claim text


async def test_reinforce_accumulates_confidence(store, fake):
    """§7.19: reinforcement reset confidence instead of accumulating."""
    confs = []
    for day in range(3):
        tr = EpisodeTracker(store)
        oid = await add(store, tr, obs(text=f"Adobe Illustrator logo session {day}", t=T0 + day * 86400))
        (eid,) = await tr.close_current()
        fake.draft = {"title": "x", "summary": "y", "claims": [
            {"kind": "fact", "text": "The user uses Adobe Illustrator for logo work", "entities": [],
             "evidence": [{"observation_id": oid, "quote": "Adobe Illustrator logo session"}]}]}
        live = await store.live_claims(kind="fact")
        fake.relations = {live[0]["id"]: "same"} if live else {}
        await drain_outbox(store)
        await ProcessPipeline(store).draft_episode(eid)
        confs.append((await store.live_claims(kind="fact"))[0]["confidence"])
    assert len(await store.live_claims(kind="fact")) == 1
    assert confs[0] < confs[1] < confs[2]


async def test_consolidation_keeps_every_output(store, fake):
    """§7.20: only the first output survived; splits were lost."""
    eid, (o1,) = await _one_episode(store, ["prefers dark themes and monospace fonts everywhere"])
    ids = []
    for text in ("The user prefers dark themes", "The user prefers dark mode themes"):
        ids.append(await store.create_claim(ClaimDraft(kind="preference", text=text, episode_id=eid,
                                                       evidence=[{"observation_id": o1, "quote": "prefers dark themes"}])))
    await drain_outbox(store)
    fake.consolidation = {"claims": [{"text": "The user prefers dark themes", "input_ids": ids},
                                     {"text": "The user prefers monospace fonts", "input_ids": [ids[1]]}]}
    await run_consolidation(store)
    live = await store.live_claims(kind="preference")
    assert sorted(c["text"] for c in live) == ["The user prefers dark themes", "The user prefers monospace fonts"]
    for c in live:
        assert await store.evidence_for_claim(c["id"])


async def test_retention_keeps_rows_and_stated_claims_survive_lint(store, fake):
    """§7.21: retention deleted rows, cascading evidence away."""
    eid, (o1,) = await _one_episode(store, ["old screen text about invoices"], )
    cid = await store.create_claim(ClaimDraft(kind="fact", text="The user reviewed invoices",
                                              evidence=[{"observation_id": o1, "quote": "invoices"}]))
    stated = await store.create_claim(ClaimDraft(kind="fact", text="The user's manager is Sam", modality="stated"))
    await store.execute("UPDATE observations SET started_at=0")
    await store.execute("UPDATE claims SET created_at=0")
    assert await store.clear_old_screen_text(ttl_days=90) == 1
    assert (await store.get_observation(o1))["screen_text"] == ""
    assert await store.evidence_for_claim(cid)
    await store.supersede_unsupported(14)
    assert (await store.get_claim(stated))["status"] == "live"
    assert (await store.get_claim(stated))["confidence"] > 0.25


async def test_outbox_covers_every_kind(store, fake):
    """§7.22: episodes and entities never left the outbox."""
    eid, (o1,) = await _one_episode(store, ["OpenRouter model pricing page"])
    fake.draft = {"title": "Pricing", "summary": "Compared model prices.", "claims": []}
    await ProcessPipeline(store).draft_episode(eid)
    await store.create_entity("OpenRouter", "website", "openrouter")
    await drain_outbox(store)
    s = await store.stats()
    assert s["outbox_pending"] == 0
    assert s["index_coverage"] == {"observation": 1.0, "episode": 1.0, "claim": 1.0, "entity": 1.0}


async def test_dense_search_is_ranked(store, fake):
    """§7.23: dense ids were in insertion order."""
    for t in ("zebra stripes savanna", "kitchen recipes pasta", "zebra savanna herd migration"):
        await store.add_observation(obs(text=t, t=T0 + len(t)))
    await drain_outbox(store)
    from tests.conftest import bow_vector

    res = await store.dense_search("observation", bow_vector("zebra savanna migration"), limit=3)
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)


# ------------------------------------------------------------------- Stage 3

def test_interval_parser_ignores_non_temporal_words():
    """§7.24: 'on hacker news' parsed as a date a week earlier."""
    assert parse_interval("what was I reading on hacker news") is None
    s, e = parse_interval("what did I do yesterday", ref=T0)
    assert e - s == pytest.approx(86400, abs=3600)
    assert parse_interval("what did I do last tuesday", ref=T0) is not None


async def test_llm_plan_cannot_disable_retrieval(fake):
    """§7.24: the LLM returned needs_store=false and free-text intents."""
    plan = await build_plan("what was I reading on hacker news")
    assert plan.needs_store and plan.intent == "recall"


async def test_retrieve_end_to_end(store, fake):
    eid, (o1, o2) = await _one_episode(store, ["Reading Hacker News thread about SQLite FTS5 ranking",
                                               "Kitchen timer app settings"])
    fake.draft = {"title": "Reading HN", "summary": "Read about SQLite FTS5.", "claims": [
        {"kind": "event", "text": "The user read a Hacker News thread about SQLite FTS5 ranking", "entities": [],
         "evidence": [{"observation_id": o1, "quote": "Hacker News thread about SQLite FTS5"}]}]}
    await ProcessPipeline(store).draft_episode(eid)
    await drain_outbox(store)
    res = await retrieve(store, "what was I reading about sqlite ranking")
    assert res.hits and "sqlite" in res.hits[0].text.lower()
    assert evaluate(res).mode == "grounded"


def test_enumerate_intent_distinct_from_aggregate():
    """§17: 'list all X' wants every record enumerated, not an entity roll-up or a count."""
    from discrete.retrieval.query import fallback_plan

    assert fallback_plan("list all the articles i opened in hacker news").intent == "enumerate"
    assert fallback_plan("show me every site i visited today").intent == "enumerate"
    assert fallback_plan("what did i read on hacker news").intent == "enumerate"
    assert fallback_plan("how many articles did i read").intent == "aggregate"
    assert fallback_plan("which tools have i used").intent == "aggregate"


async def test_enumerate_lists_every_article_without_topk_cap(store):
    """§17: enumeration bypasses TOP_K (8) and MMR, and dedups by article identity."""

    topics = ["python", "rust", "gpu tombs", "ios release", "ai models", "startups",
              "number theory", "jazz music", "deep space", "ocean life", "chess", "compilers"]
    titles = [f"HN Article {i}: {topics[i]}" for i in range(12)]
    for i, ttl in enumerate(titles):
        await store.add_observation(obs(app="Safari", site="news.ycombinator.com",
                                        title=ttl, text=f"Hacker News thread about {topics[i]} detail {i}", t=T0 + i))
    # A re-capture of one article (same app/title/url) must not become a second line.
    await store.add_observation(obs(app="Safari", site="news.ycombinator.com",
                                    title=titles[0], text=f"Hacker News thread about {topics[0]} revisited", t=T0 + 100))

    res = await retrieve(store, "list all the articles i opened in hacker news")
    assert res.plan.intent == "enumerate" and res.exhaustive
    assert len(res.hits) == 12  # every distinct article, well past TOP_K
    assert all(h.ref_kind == "observation" for h in res.hits)
    assert any("number theory" in h.title for h in res.hits)


async def test_exhaustive_mode_shows_undrafted_episodes(store, fake):
    from discrete.timeutil import now

    tr = EpisodeTracker(store)
    await add(store, tr, obs(text="budget spreadsheet", caption="The user edited the Q3 budget", t=now() - 60))
    res = await retrieve(store, "what did I do today")
    assert res.exhaustive and "Q3 budget" in res.hits[0].body


def test_structural_candidates_skip_ui_chrome_and_paths():
    """insights.md §7.27: sidebar labels, line-spanning spans and path fragments are not entities."""
    from discrete.process.entities import promotable, structural_candidates

    ax_text = "Dashboard\nHome\nProjects\nFilter\nUnnamed Unnamed\nOpenRouter timeutil.py planS1\n/Users/owaissadiqque/x"
    cands = structural_candidates(ax_text)
    assert "OpenRouter" in cands and "timeutil.py" in cands and "planS1" in cands
    assert not any("\n" in c for c in cands) and "Dashboard Home" not in cands
    assert not promotable("Users/owaissadiqque") and not promotable("Unnamed Unnamed")
    assert not promotable("Thinking\nThe")
    assert promotable("Conductor/DIscrete") and promotable("OpenRouter")


# ------------------------------------------------------------ docs/C6.md

async def test_one_relation_call_per_episode_and_durable_claims_embedded_once(store, fake):
    from discrete.process.embed_worker import claim_text

    for text in ("The user uses Neovim for Python", "The user prefers dark themes"):
        await store.create_claim(ClaimDraft(kind="fact" if "Neovim" in text else "preference", text=text))
    await drain_outbox(store)
    eid, (o1,) = await _one_episode(store, ["Neovim config for Python and a dark theme with monospace fonts"])
    fake.embedded.clear()
    fake.draft = {"title": "Editor setup", "summary": "Set up the editor.", "claims": [
        {"kind": "fact", "text": "The user configures Neovim for Python", "entities": [],
         "evidence": [{"observation_id": o1, "quote": "Neovim config for Python"}]},
        {"kind": "preference", "text": "The user prefers a dark theme", "entities": [],
         "evidence": [{"observation_id": o1, "quote": "dark theme"}]},
        {"kind": "event", "text": "The user set up Neovim with monospace fonts", "entities": [],
         "evidence": [{"observation_id": o1, "quote": "monospace fonts"}]},
    ]}
    out = await ProcessPipeline(store).draft_episode(eid)
    assert out.added == 3
    assert fake.calls.count("relate") == 1  # both durable claims in one call
    await drain_outbox(store)
    new = [c for c in await store.live_claims() if c["episode_id"] == eid and c["kind"] != "event"]
    assert len(new) == 2
    for c in new:
        assert fake.embedded.count(claim_text(c["text"], c["context"])) == 1


async def test_pending_cases_are_written_to_the_data_dir(store, fake):
    from discrete.config import data_dir, golden_dir
    from discrete.retrieval import gaps

    gaps._append_pending_case("what was that font site", "observation:3")
    assert gaps.pending_cases_path() == data_dir() / "auto_cases.pending.yaml"
    assert gaps.pending_cases_path().exists()
    assert not (golden_dir() / "auto_cases.pending.yaml").exists()


def test_no_site_specific_rules_in_topic_neutral_code():
    """The page/enumeration code and textkit may not name particular sites or apps,
    nor parse caption wording (docs/C6.md §2). Checks string literals, not comments."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    banned = ("hacker news", "ycombinator", "blueprint", "openrouter", "google.com", "finder", "illustrator",
              "github", "switched from", "clicked on a link", "arc\n")
    for rel in ("discrete/textkit.py", "discrete/retrieval/pages.py", "discrete/retrieval/enumerate.py",
                "discrete/retrieval/hybrid.py", "tools/replay_workaa.py"):
        tree = ast.parse((root / rel).read_text())
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                low = node.value.lower()
                assert not any(b in low for b in banned), f"{rel}: {node.value!r}"


def test_replay_maps_domain_apps_to_browser_without_app_names():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "replay", Path(__file__).resolve().parent.parent / "tools" / "replay_workaa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.app_and_site("logosystem.co", "") == ("Browser", "logosystem.co")
    assert mod.app_and_site("Adobe Illustrator 2026", "https://example.com/x") == ("Adobe Illustrator 2026",
                                                                                  "example.com")
    assert mod.app_and_site("", "https://example.com/x") == ("Browser", "example.com")


async def test_episodes_close_at_the_drafters_reach(store, monkeypatch):
    """A 92-observation episode lost its first 52 before drafting (Helium, live 2026-10-06)."""
    from discrete.config import T

    monkeypatch.setattr(T, "EPISODE_MAX_OBS", 5)
    tr = EpisodeTracker(store)
    closed = []
    for i in range(12):
        oid, _ = await store.add_observation(obs(text=f"distinct page number {i} " * 3, t=T0 + 10 * i))
        closed += await tr.on_observation(oid, await store.get_observation(oid), session_id="s1")
    assert len(closed) == 2  # 5 + 5, the last 2 still open


async def test_a_long_episode_is_drafted_in_chunks_and_nothing_is_dropped(store, fake, monkeypatch):
    from discrete.config import T

    monkeypatch.setattr(T, "EPISODE_MAX_OBS", 1000)
    monkeypatch.setattr(T, "DRAFT_MAX_OBS", 10)
    eid, ids = await _one_episode(store, [f"Helium download step {i} unique words {i * 7}" for i in range(25)])
    prompts = []
    real = fake.chat

    async def spy(ep, messages, **kw):
        prompts.append(messages[-1]["content"] if isinstance(messages[-1]["content"], str) else "")
        return await real(ep, messages, **kw)

    monkeypatch.setattr(fake, "chat", spy)
    fake.draft = {"title": "t", "summary": "s", "claims": [
        {"kind": "event", "text": "The user viewed Helium download step 0", "action": "",
         "evidence": [{"observation_id": ids[0], "quote": "Helium download step 0"}]}]}
    await ProcessPipeline(store).draft_episode(eid)
    drafts = [p for p in prompts if "one episode of a person's computer activity" in p]
    assert len(drafts) == 3  # 10 + 10 + 5
    assert f"observation_id={ids[0]}]" in drafts[0] and f"observation_id={ids[-1]}]" in drafts[-1]
