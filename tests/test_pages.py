"""First-class pages: identities, sighting metrics, model-free routes and privacy."""

from __future__ import annotations

import time

import pytest

from discrete.bubble.engine import BubbleEngine
from discrete.capture import privacy
from discrete.config import T
from discrete.models import ClaimDraft, ObservationDraft
from discrete.page_identity import canonical_url, classify_kind, document_identity
from discrete.process.dream import attention_context
from discrete.retrieval.hybrid import retrieve
from discrete.retrieval.query import build_plan, fallback_plan

T0 = 1_790_000_000.0


@pytest.mark.parametrize(("raw", "expected"), [
    ("HTTPS://WWW.Example.COM:443/blog/post/?utm_source=x&b=2&a=1#part", "https://example.com/blog/post?a=1&b=2"),
    ("http://Example.com:80/", "http://example.com"),
    ("https://example.com:8080/A/?fbclid=x&id=17&ref_src=feed", "https://example.com:8080/A?id=17"),
    ("https://youtube.com/watch?v=one&si=tracking#t=30", "https://youtube.com/watch?v=one"),
    ("https://example.com/search?q=hello+world&utm_campaign=a", "https://example.com/search?q=hello+world"),
    ("file://localhost/Users/me/Report%20one.pdf#page=2", "file:///Users/me/Report%20one.pdf"),
    ("/Users/me/../me/Report.pdf", "file:///Users/me/Report.pdf"),
    ("https://[::1]:443/", "https://[::1]"),
    ("https://example.com:bad/", ""),
    ("not a URL", ""),
])
def test_canonicalization(raw, expected):
    assert canonical_url(raw) == expected
    if expected:
        assert canonical_url(expected) == expected


@pytest.mark.parametrize(("url", "listing", "kind"), [
    ("https://github.com/openai/example", False, "repo"),
    ("https://gitlab.com/group/repo/-/blob/main/README.md", False, "repo"),
    ("https://github.com/topics/memory", False, "listing"),
    ("https://youtube.com/watch?v=123", False, "video"),
    ("https://youtube.com/", False, "listing"),
    ("https://example.org/blog/long-read", False, "article"),
    ("https://example.org/products/item", False, "product"),
    ("https://chatgpt.com/c/one", False, "chat"),
    ("https://docs.google.com/document/d/one/edit", False, "doc"),
    ("https://docs.python.org/3/library/", False, "doc"),
    ("https://google.com/search?q=read", True, "search"),
    ("https://example.org/blog/", True, "listing"),
    ("file:///Users/me/Report.pdf", False, "doc"),
    ("file:///Users/me/image.png", False, "file"),
    ("https://example.org/", False, "other"),
    ("doc://notes/Inbox%20%E2%80%93%205%20notes", False, "listing"),
])
def test_kind_rules(url, listing, kind):
    assert classify_kind(url if url.startswith("doc://") else canonical_url(url), is_listing=listing) == kind


def test_title_only_document_identity():
    assert document_identity("TextEdit", "Report.txt — TextEdit") == "doc://textedit/Report.txt"
    assert document_identity("Notes", "Travel plans") == "doc://notes/Travel%20plans"
    assert document_identity("Finder", "image.png") == "file-title://finder/image.png"
    assert document_identity("Preview", "/Users/me/Report.pdf") == "file:///Users/me/Report.pdf"
    assert document_identity("Arc", "Space 1") == ""
    assert document_identity("TextEdit", "Untitled") == ""
    assert document_identity("Notes", "Report") != document_identity("TextEdit", "Report")
    assert document_identity("Notes", "Notes") == ""


async def see(store, offset, url, *, title="Page", app="Arc", keys=0, clicks=0, text=""):
    oid, _ = await store.add_observation(ObservationDraft(
        app=app, window_title=title, url=url, screen_text=text or f"content {url or title}",
        started_at=T0 + offset, ended_at=T0 + offset + 2, keys=keys, clicks=clicks))
    assert oid is not None
    return oid


async def test_returns_dwell_and_collapsed_sightings(store):
    url = "https://github.com/team/repo"
    a = await see(store, 0, url, clicks=1)
    assert await see(store, 20, url) == a
    await see(store, 60, "https://example.org/blog/article")
    assert await see(store, 120, url, keys=1) == a
    rows = {p["canonical_url"]: p for p in await store.list_pages()}
    p = rows[url]
    assert p["visits"] == 2
    assert p["dwell_s"] == 90  # 20 + 40 + last 30
    assert p["active_dwell_s"] == 50  # first 20 and last 30
    assert p["engaged"] == 1
    assert p["first_seen"] == T0 and p["last_seen"] == T0 + 122
    assert (await store.get_observation(a))["page_id"] == p["id"]
    stored = await store.q1("SELECT * FROM pages WHERE id=?", (p["id"],))
    assert stored["dwell_s"] == p["dwell_s"] and stored["visits"] == 2


async def test_idle_and_out_of_order_capture_refresh_neighbors(store):
    await see(store, 0, "https://example.org/a", clicks=1)
    await see(store, 1000, "https://example.org/a")
    assert (await store.list_pages())[0]["visits"] == 2
    await see(store, 60, "https://example.org/b")
    rows = {p["canonical_url"]: p for p in await store.list_pages()}
    assert rows["https://example.org/a"]["dwell_s"] == 90
    assert rows["https://example.org/a"]["active_dwell_s"] == 60
    assert rows["https://example.org/b"]["dwell_s"] == T.DWELL_GAP_CAP_S


async def test_window_clips_dwell_and_uses_next_sighting_outside_interval(store):
    await see(store, 0, "https://example.org/a", clicks=1)
    await see(store, 60, "https://example.org/b")
    p = (await store.list_pages(start=T0 + 10, end=T0 + 40))[0]
    assert p["dwell_s"] == p["active_dwell_s"] == 30
    assert p["visits"] == 0  # the visit began before the requested window
    p = (await store.list_pages(start=T0, end=T0 + 70, search="/a"))[0]
    assert p["dwell_s"] == 60


async def test_deleting_and_discarding_updates_totals(store):
    a = await see(store, 0, "https://example.org/a")
    b = await see(store, 60, "https://example.org/b")
    await store.execute("DELETE FROM observations WHERE id=?", (b,))
    assert (await store.list_pages())[0]["dwell_s"] == T.DWELL_LAST_S
    await store.discard_observation(a)
    assert await store.list_pages() == []
    assert (await store.get_observation(a))["page_id"] is None


async def test_backfill_idempotent_and_no_model_calls(store, fake):
    oid = await see(store, 0, "https://www.example.org/blog/a?utm_source=feed#part")
    await store.execute("DELETE FROM pages")  # simulate legacy observations with no links
    assert await store.backfill_pages() == 1
    first = [dict(r) for r in await store.q("SELECT * FROM pages")]
    assert await store.backfill_pages() == 1
    assert [dict(r) for r in await store.q("SELECT * FROM pages")] == first
    assert (await store.get_observation(oid))["page_id"] == first[0]["id"]
    assert fake.calls == [] and fake.embedded == []


async def test_near_duplicate_text_on_different_pages_is_not_collapsed(store):
    text = " ".join(f"unique{i}" for i in range(200))
    a = await see(store, 0, "https://example.org/a", text=text)
    b = await see(store, 60, "https://example.org/b", text=text)
    assert a != b
    assert len(await store.list_pages()) == 2


async def test_title_and_listing_enrichment_updates_page(store):
    oid = await see(store, 0, "https://example.org/feed")
    await store.enrich_observation(oid, caption="reading", entities=[], page_title="Reading feed",
                                   is_listing=True, navigated_from="")
    p = (await store.list_pages())[0]
    assert p["title"] == "Reading feed" and p["kind"] == "listing"


async def test_document_identity_stays_grounded_after_enrichment(store):
    oid = await see(store, 0, "", app="TextEdit", title="Actual.txt")
    await store.enrich_observation(oid, caption="reading", entities=[], page_title="Inferred document name",
                                   is_listing=False, navigated_from="")
    p = (await store.list_pages())[0]
    assert p["canonical_url"] == "doc://textedit/Actual.txt"
    assert p["title"] == "Inferred document name"


async def test_sighting_updates_recompute_both_old_and_new_neighbors(store):
    await see(store, 0, "https://example.org/a")
    b = await see(store, 60, "https://example.org/b")
    await see(store, 120, "https://example.org/c")
    await store.execute("UPDATE sightings SET started_at=?,ended_at=?,keys=1 WHERE obs_id=?",
                        (T0 + 150, T0 + 151, b))
    rows = {p["canonical_url"]: p for p in await store.list_pages()}
    assert rows["https://example.org/a"]["dwell_s"] == 120
    assert rows["https://example.org/c"]["dwell_s"] == 30
    assert rows["https://example.org/b"]["active_dwell_s"] == 30


async def test_page_privacy_and_redaction(store, monkeypatch):
    monkeypatch.setattr(privacy, "_SITES", ("blocked.example",))
    denied = await see(store, 0, "https://sub.blocked.example/private")
    denied_app = await see(store, 20, "", app="1Password", title="Secrets.txt")
    self_row, _ = await store.add_observation(ObservationDraft(
        origin="self", app="Arc", url="https://example.org/self", screen_text="answer"))
    token = "sk-proj-" + "a" * 30
    safe = await see(store, 60, f"https://example.org/article?api_key={token}", title=f"Title {token}")
    rows = await store.list_pages()
    assert len(rows) == 1
    assert token not in str(rows)
    assert "REDACTED" in rows[0]["title"] and "REDACTED" in rows[0]["canonical_url"]
    for oid in (denied, denied_app, self_row):
        assert (await store.get_observation(oid))["page_id"] is None
    assert (await store.get_observation(safe))["page_id"] is not None
    assert await store.backfill_pages() == 1
    # A newly denied site is removed on the next page read, not resurfaced from cache.
    monkeypatch.setattr(privacy, "_SITES", ("example.org",))
    assert await store.list_pages() == []


async def test_forget_unlinks_page_and_backfill_does_not_restore_it(store):
    oid = await see(store, 0, "https://example.org/blog/forget")
    cid = await store.create_claim(ClaimDraft(kind="fact", text="Read this page",
                                             evidence=[{"observation_id": oid, "quote": "content"}]))
    event = await store.forget_claim(cid)
    assert (await store.get_observation(oid))["page_id"] is None
    assert await store.backfill_pages() == 0
    await store.revert_event(event)
    assert len(await store.list_pages()) == 1


@pytest.mark.parametrize(("question", "operation"), [
    ("what did I read most?", "read_most"),
    ("which repos do I return to?", "returns"),
    ("which docs did I edit?", "edited"),
    ("how long on example.org?", "duration"),
    ("how much time did I spend on example.org today?", "duration"),
    ("what did I read most and did I buy it?", ""),
    ("which repos do I prefer?", ""),
    ("which docs should I edit?", ""),
    ("why did I return to github.com?", ""),
    ("how long did I spend installing Helium?", ""),
    ("how long on example.org today and did I buy it?", ""),
    ("what time on Monday did I open the report?", ""),
])
def test_guarded_planner_routes(question, operation):
    assert fallback_plan(question).page_operation == operation


async def test_page_retrieval_and_writer_do_not_call_any_model(store, fake, monkeypatch):
    monkeypatch.setattr("discrete.retrieval.page_activity.now", lambda: T0 + 1000)
    await see(store, 0, "https://github.com/team/repo")
    await see(store, 60, "https://example.org/blog/article")
    await see(store, 120, "https://github.com/team/repo")
    await see(store, 180, "", title="Report.txt", app="TextEdit", keys=3)
    for question in ("what did I read most?", "which repos do I return to?", "which docs did I edit?",
                     "how long on github.com?"):
        plan = await build_plan(question)
        assert plan.intent in ("aggregate", "enumerate")
        result = await retrieve(store, question)
        assert result.hits and result.page_answer
        assert all(h.ref_kind == "page" for h in result.hits)
    # Exercise the public headless Bubble path, including writer bypass.
    engine = BubbleEngine(store)
    prepared = await engine.prepare("which repos do I return to?", app="Terminal", field_desc="test")
    chunks = []
    output = await engine.write(prepared, lambda text: chunks.append(text) or True)
    assert "Repositories revisited" in output and "github.com/team/repo" in output
    assert fake.calls == [] and fake.embedded == []


async def test_dream_context_includes_page_signals_without_new_claims(store, monkeypatch):
    monkeypatch.setattr("discrete.process.dream.now", lambda: T0 + 1000)
    await see(store, 0, "https://github.com/team/repo", clicks=1)
    context = await attention_context(store)
    assert "Page-level interest/habit signals" in context
    assert "github.com/team/repo" in context and "context only" in context
    assert await store.scalar("SELECT COUNT(*) FROM claims") == 0


async def test_cli_pages_is_model_free(tmp_db, fake, monkeypatch, capsys):
    from discrete.cli import build_parser, cmd_pages
    from discrete.store import Store

    monkeypatch.setenv("DISCRETE_DB", tmp_db)
    s = await Store.open(tmp_db)
    await s.add_observation(ObservationDraft(app="Arc", url="https://github.com/team/repo", screen_text="repo",
                                             started_at=time.time(), ended_at=time.time()))
    await s.close()
    args = build_parser().parse_args(["pages", "--backfill", "--since", "7d", "--kind", "repo", "--top", "1"])
    assert await cmd_pages(args) == 0
    assert "github.com/team/repo" in capsys.readouterr().out
    assert fake.calls == [] and fake.embedded == []
