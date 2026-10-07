"""Synthetic visit graphs: no held-out names/domains, models or replay fixtures."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime

import pytest

from discrete.config import T
from discrete.models import AttentionMoment, Hit, ObservationDraft, ProvenanceEdge, QueryPlan, RetrievalResult
from discrete.process.provenance import build_provenance, infer_edges, valid_edge
from discrete.retrieval.grounding import evaluate
from discrete.retrieval.hybrid import expand_why
from discrete.retrieval.query import fallback_plan
from discrete.retrieval.support import verify_support
from discrete.retrieval.why import chain_hits, render_chain, resolve_why, walk_back
from discrete.store import Store

T0 = 1_790_000_000.0


async def obs(store, offset, title, text="", *, app="Browser", url="", caption="", keys=0,
              clipboard=0, navigated_from="", origin="user", session="s"):
    oid, _ = await store.add_observation(ObservationDraft(
        started_at=T0 + offset, ended_at=T0 + offset + 1, app=app, page_title=title, window_title=title,
        screen_text=text, url=url, caption=caption or title, keys=keys, clipboard=clipboard,
        navigated_from=navigated_from, session_id=session, origin=origin))
    return oid


@pytest.mark.parametrize("kind,source,target", [
    ("navigated_from", dict(title="Quartz article", url="https://quartz.test/story"),
     dict(title="Research paper", navigated_from="Quartz article")),
    ("navigation", dict(title="Reading list", text="Detailed report https://reports.test/quartz"),
     dict(title="Detailed report", url="https://reports.test/quartz")),
    ("navigation", dict(title="Index", text="Quantum orchard report"), dict(title="Quantum orchard report")),
    ("search", dict(title="Quartz - Web Search", url="https://search.test/?q=Quartz", keys=5),
     dict(title="Quartz", url="https://quartz.test/")),
    ("clipboard", dict(title="Report", url="https://reports.test/quartz", clipboard=1),
     dict(title="Assistant", text="Explain https://reports.test/quartz", keys=25)),
    ("message_share", dict(title="Direct Messages", text="Mina shared: https://reports.test/quartz"),
     dict(title="Quartz report", url="https://reports.test/quartz")),
    ("message_share", dict(title="Direct Messages", text="Mina sent a post by Zephyr Daniels"),
     dict(title="Zephyr Daniels profile")),
    ("media_mention", dict(title="Ravi's post", text="Quartz File Edit", caption="A demo video is embedded."),
     dict(title="Quartz - Web Search", url="https://search.test/?q=Quartz", keys=6)),
    ("app_switch", dict(title="Project", text="Zephyr invoices", app="Editor"),
     dict(title="Zephyr invoice spreadsheet", app="Sheets")),
    ("typed_navigation", dict(title="VectorBloom launch post", text="VectorBloom embeddings",
                             caption="A launch post announces VectorBloom embeddings", url="https://social.test/post/1"),
     dict(title="Home", text="VectorBloom", url="https://social.test/home", keys=8,
          caption='A VectorBloom post is visible. The user is typing "registry.test" into the command bar.')),
])
async def test_edge_kinds_have_real_quotes(store, kind, source, target):
    await obs(store, 0, **source)
    await obs(store, 20, **target)
    moments = await store.attention_moments(T0, T0 + 25)
    edges = infer_edges(moments)
    assert kind in {e.kind for e in edges}
    assert all(e.gap_s == 20 and valid_edge(e, {m.sighting_id: m for m in moments}) for e in edges)
    assert await build_provenance(store) == len(edges)
    assert await build_provenance(store) == 0
    assert await store.provenance_edges(T0, T0 + 25) == edges


async def test_incremental_late_enrichment_and_out_of_order_source(store):
    target = await obs(store, 20, "Quartz report", url="https://reports.test/quartz")
    assert await build_provenance(store) == 0
    await obs(store, 0, "Direct Messages", text="Mina shared https://reports.test/quartz")
    assert await build_provenance(store) == 1
    await store.execute("UPDATE observations SET screen_text='', caption='', updated_at=updated_at+1 WHERE id!=?", (target,))
    assert not await store.provenance_edges(T0, T0 + 25)
    # Future synthetic updated_at values should not make full backfill loop forever.


async def test_source_enrichment_can_create_previously_absent_edge(store):
    source = await obs(store, 0, "Direct Messages", "nothing relevant")
    await obs(store, 20, "Quartz report", url="https://reports.test/quartz")
    assert await build_provenance(store) == 0
    from discrete.timeutil import now

    await store.execute("UPDATE observations SET screen_text=?, updated_at=? WHERE id=?",
                        ("Mina shared https://reports.test/quartz", now(), source))
    assert await build_provenance(store) == 1


async def test_temporal_only_and_clipboard_count_without_payload_do_not_connect(store):
    await obs(store, 0, "Music", "Orchard soundtrack", clipboard=1)
    await obs(store, 20, "Quartz report", "Some unrelated text", keys=15)
    assert infer_edges(await store.attention_moments(T0, T0 + 25)) == []


async def test_self_capture_and_session_boundaries_are_excluded(store):
    await obs(store, 0, "Direct Messages", "Quartz report", origin="self")
    await obs(store, 10, "Direct Messages", "Quartz report", session="other")
    await obs(store, 20, "Quartz report")
    assert not infer_edges(await store.attention_moments(T0, T0 + 25))


async def test_navigation_chain_and_refusal_without_origin(store):
    root = await obs(store, 0, "Direct Messages", "Mina shared https://reports.test/quartz")
    mid = await obs(store, 20, "Quartz report", "Read the Quartz appendix", url="https://reports.test/quartz")
    end = await obs(store, 40, "Quartz appendix", navigated_from="Quartz report")
    chain = await resolve_why(store, "why did I open the Quartz appendix?")
    assert chain.effect.observation_id == end
    assert [m.observation_id for m in chain.moments] == [root, mid, end]
    assert [e.kind for e in chain.edges] == ["message_share", "navigated_from"]
    assert "source [" in render_chain(chain)
    assert not (await resolve_why(store, "why did I open the nonexistent glacier?")).effect
    lonely = await obs(store, 100, "Solitary fern")
    chain = await resolve_why(store, "why did I open Solitary fern?")
    assert chain.effect.observation_id == lonely and chain.reason == "why not known"


async def test_two_copied_urls_are_both_preserved(store):
    a = await obs(store, 0, "Report A", url="https://reports.test/a", clipboard=1)
    b = await obs(store, 10, "Report B", url="https://reports.test/b", clipboard=1)
    c = await obs(store, 20, "Assistant", "Explain https://reports.test/a and https://reports.test/b", keys=40)
    chain = await resolve_why(store, "why did I open Assistant?")
    assert {m.observation_id for m in chain.moments} == {a, b, c}
    assert [e.kind for e in chain.edges] == ["clipboard", "clipboard"]


async def test_clock_anchor_uses_sightings_instead_of_merged_observation_span(store):
    await obs(store, 0, "Assistant", "An old discussion")
    a = await obs(store, 7200, "Quartz report", url="https://reports.test/quartz", clipboard=1)
    b = await obs(store, 7220, "Assistant", "Explain https://reports.test/quartz", keys=25)
    clock = datetime.fromtimestamp(T0 + 7220).strftime("%H:%M")
    chain = await resolve_why(store, f"why did I open Assistant around {clock}?")
    assert chain.effect.observation_id == b and chain.moments[0].observation_id == a
    assert fallback_plan(f"why did I open Assistant around {clock}?").interval is None


async def test_clock_anchor_overrides_repeat_cue_and_first_arrival(store):
    await obs(store, 0, "Quartz report", url="https://reports.test/quartz")
    await obs(store, 7200, "Quartz report", url="https://reports.test/quartz")
    await obs(store, 86400 + 10800, "Quartz report", url="https://reports.test/quartz")
    clock = datetime.fromtimestamp(T0 + 7200).strftime("%H:%M")
    chain = await resolve_why(store, f"why did I open Quartz report again around {clock}?")
    assert chain.effect.at == T0 + 7200


@pytest.mark.parametrize("anchor", ["yesterday", datetime.fromtimestamp(T0).strftime("on %Y-%m-%d")])
async def test_date_anchor_selects_older_sitting_first_arrival(store, anchor):
    for dt in (20, 40, 86420, 86460):
        await obs(store, dt, "Quartz report", url="https://reports.test/quartz")
    chain = await resolve_why(store, f"why did I open Quartz report {anchor}?", reference_at=T0 + 86480)
    assert chain.effect.at == T0 + 20
    latest = await resolve_why(store, f"why did I open Quartz report again {anchor}?", reference_at=T0 + 86480)
    assert latest.effect.at == T0 + 40


async def test_download_sitting_starts_at_action_instead_of_product_homepage(store):
    await obs(store, 0, "Quartz browser", url="https://quartz.test/")
    arrival = await obs(store, 20, "Thank you for downloading Quartz browser", url="https://quartz.test/download")
    await obs(store, 40, "Thank you for downloading Quartz browser", url="https://quartz.test/download")
    chain = await resolve_why(store, "why did I download Quartz browser?")
    assert chain.effect.at == T0 + 20 and chain.effect.observation_id == arrival


async def test_quoted_dialog_return_uses_first_matching_state(store):
    await obs(store, 0, "Importer", "Import from Cedar", app="Birch")
    arrival = await obs(store, 20, "Importer", "Import profile still in use", app="Birch")
    await obs(store, 40, "Birch Import profile still in use", "Import profile still in use", app="Birch")
    chain = await resolve_why(store, 'why did Birch say "Import profile still in use"?')
    assert chain.effect.at == T0 + 20 and chain.effect.observation_id == arrival


def _node(n, at, title="Quartz", text="Quartz"):
    return AttentionMoment(n, n, at, at + 1, {"screen_text": text, "page_title": title, "app": "Browser"},
                           app="Browser", session_id="s")


def _edge(a, b, confidence=0.9):
    return ProvenanceEdge(a.sighting_id, b.sighting_id, "navigation", confidence,
                          "screen_text", "Quartz", "screen_text", "Quartz", b.at - a.at)


def test_walk_bounds_cycles_stale_quotes_and_confidence(monkeypatch):
    nodes = [_node(i, T0 + i * 10) for i in range(1, 6)]
    edges = [_edge(a, b) for a, b in zip(nodes, nodes[1:], strict=False)]
    edges += [replace(edges[0], source=2, target=1, gap_s=0), replace(edges[-1], source=999)]
    assert len(walk_back(nodes[-1], nodes, edges).edges) == 4
    monkeypatch.setattr(T, "WHY_CHAIN_DEPTH", 2)
    assert len(walk_back(nodes[-1], nodes, edges).edges) == 2
    monkeypatch.setattr(T, "WHY_CHAIN_WINDOW_S", 15)
    assert len(walk_back(nodes[-1], nodes, edges).edges) == 1
    assert not walk_back(nodes[-1], nodes, [replace(edges[3], confidence=0.3)]).edges
    assert not walk_back(nodes[-1], nodes, [replace(edges[3], source_quote="invented")]).edges


async def test_error_state_context_is_recent_app_evidence_not_process_telemetry(store):
    source = await obs(store, 0, "Quartz report", "Quartz", app="Cedar")
    await obs(store, 10, "Importer", "Import from Cedar", app="Birch")
    error = await obs(store, 20, "Importer", "Import profile still in use. First close the browser you are importing from.",
                      app="Birch")
    chain = await resolve_why(store, 'why did Birch say "Import profile still in use"?')
    assert chain.effect.observation_id == error
    assert any(e.kind == "state_context" for e in chain.edges)
    assert source in {m.observation_id for m in chain.moments}
    assert chain.confidence <= 0.6


async def test_error_does_not_assume_unrelated_or_long_absent_app_is_running(store):
    await obs(store, -3600, "Quartz", app="Cedar")
    await obs(store, 0, "Music", app="Player")
    await obs(store, 10, "Importer", "Import from Cedar", app="Birch")
    await obs(store, 20, "Importer", "Import profile still in use", app="Birch")
    assert not any(e.kind == "state_context" for e in (await resolve_why(
        store, 'why did Birch say "Import profile still in use"?')).edges)


async def test_chain_records_require_support_and_partial_chain_fails_closed(store, fake):
    await obs(store, 0, "Direct Messages", "Mina shared https://reports.test/quartz")
    await obs(store, 20, "Quartz report", url="https://reports.test/quartz")
    result = RetrievalResult("why did I open Quartz report?", fallback_plan("why did I open Quartz report?"))
    await expand_why(store, result)
    assert result.why_chain.edges and all(not h.strong for h in result.hits)
    await verify_support(result)
    assert all(h.strong for h in result.hits) and evaluate(result).mode == "why"
    result.hits = chain_hits(result.why_chain)
    fake.support_indices = [1]
    await verify_support(result)
    assert result.hits == [] and result.why_chain is None and evaluate(result).mode == "empty"


async def test_non_why_unchanged_and_effect_alone_cannot_answer_why(store, fake):
    await obs(store, 0, "Quartz report")
    assert (await resolve_why(store, "what did I read?")).reason == "not a why question"
    hit = Hit("observation", 1, "Quartz", "", strong=True)
    assert evaluate(RetrievalResult("what did I read?", QueryPlan("", intent="recall"), [hit])).mode == "grounded"
    assert evaluate(RetrievalResult("why did I open Quartz?", QueryPlan("", intent="why"), [hit])).mode == "thin"
    assert fake.calls == []


async def test_model_free_backfill_and_read_only_old_schema(tmp_path, fake):
    path = str(tmp_path / "memory.db")
    store = await Store.open(path)
    await obs(store, 0, "Index", "Quartz report")
    await obs(store, 20, "Quartz report")
    assert await build_provenance(store, backfill=True) == 1
    await store.execute("DROP TABLE attention_provenance")
    await store.close()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    store = Store(conn)
    before = conn.total_changes
    assert (await resolve_why(store, "why did I open Quartz report?")).edges
    assert conn.total_changes == before and fake.calls == [] and fake.embedded == []
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO meta VALUES ('cannot', 'write')")
    await store.close()


async def test_graph_cascades_on_evidence_deletion(store):
    source = await obs(store, 0, "Index", "Quartz report")
    await obs(store, 20, "Quartz report")
    await build_provenance(store)
    await store.execute("DELETE FROM observations WHERE id=?", (source,))
    assert not await store.provenance_edges(T0, T0 + 30)


async def test_clipboard_buttons_and_visible_sidebar_urls_are_not_paste_evidence(store):
    await obs(store, 0, "Report A", "Copy link https://reports.test/a", url="https://reports.test/a")
    await obs(store, 10, "Assistant", "Explain https://reports.test/a", keys=25)
    assert not any(e.kind == "clipboard" for e in infer_edges(await store.attention_moments(T0, T0 + 15)))
    await store.reset_provenance()
    await obs(store, 20, "Report B", url="https://reports.test/b", clipboard=1)
    await obs(store, 30, "Assistant", "Open tabs:\nhttps://reports.test/b\nUnrelated typing", keys=25)
    assert not any(e.kind == "clipboard" and e.target == 4 for e in infer_edges(
        await store.attention_moments(T0 + 20, T0 + 35)))


async def test_url_scheme_tokens_do_not_connect_unrelated_searches(store):
    await obs(store, 0, "Assistant", caption='The user is typing "https://reports.test/quartz" in the prompt input', keys=8)
    await obs(store, 20, "Fern - Search", url="https://search.test/?q=fern")
    assert not infer_edges(await store.attention_moments(T0, T0 + 30))


async def test_explicit_domain_cannot_match_merely_related_foreign_page(store):
    await obs(store, 0, "Quartz report", "Discusses a paper on quartz.test", url="https://other.test/")
    assert not (await resolve_why(store, "why did I open Quartz report on quartz.test?")).effect


async def test_post_origin_stops_unrelated_earlier_same_site_visits(store):
    await obs(store, 0, "Invoices", url="https://social.test/invoices")
    root = await obs(store, 20, "Quartz launch post", "Quartz File Edit", url="https://social.test/posts/1",
                     caption="A post shows an embedded demo video")
    await obs(store, 40, "Quartz - Search", url="https://search.test/?q=Quartz")
    chain = await resolve_why(store, "why did I open Quartz - Search?")
    assert chain.moments[0].observation_id == root


async def test_action_trails_accompany_graph_and_remain_verified(store, fake):
    from discrete.models import ClaimDraft
    from discrete.process.trails import build_trails

    root = await obs(store, 0, "Quartz launch post", "Quartz File Edit", caption="A post shows a demo video")
    await obs(store, 20, "Quartz - Search", url="https://search.test/?q=Quartz")
    end = await obs(store, 40, "Quartz download", "Thank you for downloading Quartz!")
    cid = await store.create_claim(ClaimDraft("event", "The user downloaded Quartz", action="downloaded",
                                           occurred_from=T0 + 40, occurred_to=T0 + 41,
                                           evidence=[{"observation_id": end, "quote": "Thank you for downloading Quartz!"}]))
    fake.trail = {"links": [{"observation_id": root, "role": "trigger", "quote": "Quartz File Edit", "confidence": 0.7,
                             "explanation": "A demo video showed Quartz."}]}
    await build_trails(store)
    result = RetrievalResult("why did I download Quartz?", fallback_plan("why did I download Quartz?"),
                             [Hit("claim", cid, "The user downloaded Quartz", "", claim_kind="event", action="downloaded")])
    await expand_why(store, result)
    assert result.why_chain.edges and result.why_chain.trails
    assert any(h.via.startswith(f"led to claim {cid}") for h in result.hits)


async def test_no_connecting_evidence_stays_refusal_after_support(store, fake):
    await obs(store, 0, "Quartz note", "Interesting details")
    end = await obs(store, 20, "Quartz report", "Different details")
    question = "why did I open Quartz report?"
    result = RetrievalResult(question, fallback_plan(question), [Hit("observation", end, "Quartz report", "", strong=True)])
    await expand_why(store, result)
    await verify_support(result)
    assert result.why_chain.reason == "why not known" and evaluate(result).mode == "thin"


async def test_chain_support_timeout_fails_closed(store, fake, monkeypatch):
    import asyncio

    from discrete.retrieval import support

    await obs(store, 0, "Index", "Quartz report")
    await obs(store, 20, "Quartz report")
    q = "why did I open Quartz report?"
    chain = await resolve_why(store, q)
    result = RetrievalResult(q, fallback_plan(q), chain_hits(chain), why_chain=chain)

    async def slow(*args):
        await asyncio.sleep(1)

    monkeypatch.setattr(support, "_supported", slow)
    monkeypatch.setattr(T, "SUPPORT_TIMEOUT_S", 0.001)
    await verify_support(result)
    assert not result.hits and result.why_chain is None


async def test_drafting_builds_graph_even_without_model_configuration(store, fake):
    from discrete.config import Endpoint
    from discrete.process.pipeline import ProcessPipeline

    a = await obs(store, 0, "Index", "Quartz report")
    b = await obs(store, 20, "Quartz report")
    ep = await store.open_episode("s", T0)
    await store.execute("UPDATE observations SET episode_id=? WHERE id IN (?,?)", (ep, a, b))
    await store.mark_episode_ready(ep)
    pipe = ProcessPipeline(store)
    pipe.ep = Endpoint("insight", "", "", "")
    await pipe.draft_episode(ep)
    assert await store.provenance_edges(T0, T0 + 25)
    assert fake.calls == []


async def test_actual_deduplicated_return_visit_has_its_own_time_and_cause(store):
    await obs(store, 0, "Direct Messages", "Ravi shared https://reports.test/quartz", session="old")
    first = await obs(store, 20, "Quartz report", "Report text", url="https://reports.test/quartz", session="old")
    recent = await obs(store, 7200, "Direct Messages", "Mina shared https://reports.test/quartz", session="new")
    returned = await obs(store, 7220, "Quartz report", "Report text", url="https://reports.test/quartz", session="new")
    assert first == returned
    clock = datetime.fromtimestamp(T0 + 7220).strftime("%H:%M")
    chain = await resolve_why(store, f"why did I open Quartz report around {clock}?")
    assert chain.effect.at == T0 + 7220
    assert [m.observation_id for m in chain.moments] == [recent, returned]
    assert len(await store.q("SELECT id FROM sightings WHERE obs_id=?", (returned,))) == 2


async def test_source_quote_changed_during_build_is_not_persisted(store):
    source = await obs(store, 0, "Index", "Quartz report")
    await obs(store, 20, "Quartz report")
    moments = await store.attention_moments(T0, T0 + 25)
    edges = infer_edges(moments)
    await store.execute("UPDATE observations SET screen_text='', caption='' WHERE id=?", (source,))
    assert await store.save_provenance(moments[-1], edges, "v1") == 0
    assert not await store.provenance_edges(T0, T0 + 25)


async def test_debug_cli_backfill_does_not_call_models(store, tmp_db, fake, monkeypatch, capsys):
    from argparse import Namespace

    from discrete.cli import cmd_why

    await obs(store, 0, "Index", "Quartz report")
    await obs(store, 20, "Quartz report")
    monkeypatch.setenv("DISCRETE_DB", tmp_db)
    assert await cmd_why(Namespace(text="why did I open Quartz report?", chain=True, backfill=True)) == 0
    assert "source [" in capsys.readouterr().out
    assert fake.calls == [] and fake.embedded == []


async def test_cancelled_graph_pass_does_not_reopen_committed_episode(store, fake, monkeypatch):
    import asyncio

    from discrete.config import Endpoint
    from discrete.process import provenance
    from discrete.process.pipeline import ProcessPipeline

    oid = await obs(store, 0, "Quartz report")
    ep = await store.open_episode("s", T0)
    await store.execute("UPDATE observations SET episode_id=? WHERE id=?", (ep, oid))
    await store.mark_episode_ready(ep)
    pipe = ProcessPipeline(store)
    pipe.ep = Endpoint("insight", "", "", "")

    async def cancel(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(provenance, "build_provenance", cancel)
    with pytest.raises(asyncio.CancelledError):
        await pipe.draft_episode(ep)
    assert (await store.get_episode(ep))["status"] == "closed"
