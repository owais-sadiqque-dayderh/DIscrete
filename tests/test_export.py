"""The Markdown mirror keeps navigable links, stable bytes and Git history."""

from __future__ import annotations

import re
import subprocess

from discrete.export import export_markdown
from discrete.models import ClaimDraft, ObservationDraft
from discrete.timeutil import day_key

T0 = 1_790_000_000.0


async def _observation(store, title: str, at: float) -> int:
    oid, _ = await store.add_observation(ObservationDraft(
        app="Arc", window_title=title, screen_text=f"{title} browser download details", caption=title,
        session_id="export", started_at=at, ended_at=at + 2, keys=1))
    return oid


async def _claim(store, oid: int, text: str, *, kind: str = "fact", action: str = "") -> int:
    return await store.create_claim(ClaimDraft(
        kind=kind, text=text, modality="engaged", action=action, occurred_from=T0 + 10,
        occurred_to=T0 + 12, evidence=[{"observation_id": oid, "quote": text}]))


def _tracked_files(out) -> list[str]:
    result = subprocess.run(["git", "-C", str(out), "ls-files"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


async def test_pages_links_and_causal_trail(store, tmp_path):
    source = await _observation(store, "Theo's post", T0)
    download = await _observation(store, "Helium download", T0 + 10)
    fact = await _claim(store, source, "The user likes the Helium browser")
    action = await _claim(store, download, "The user downloaded Helium", kind="event", action="downloaded")
    person = await store.create_entity("Theo", "person", "theo")
    thing = await store.create_entity("Helium", "software", "helium")
    await store.record_mention(person, "claim", fact)
    await store.record_mention(thing, "claim", fact)
    await store.update_entity_page(thing, "A browser the user explored.")
    await store.save_digest(day_key(T0 + 10), "The user explored Helium.", 1)
    await store.record_trail(action, [{"observation_id": source, "relation": "led_to", "confidence": 0.8,
                                       "explanation": "The post showed Helium.", "quote": "Helium"}], reason="test")

    out = tmp_path / "memory"
    report = await export_markdown(store, out, git=False)
    assert {"MEMORY.md", "beliefs.md", "actions.md", "changes.md"} <= set(report["files_written"])
    assert f"days/{day_key(T0 + 10)}.md" in report["files_written"]
    assert "[strength: tentative, seen in 1 episode" in (out / "beliefs.md").read_text()
    assert "; source: Arc — Theo's post; added:" in (out / "beliefs.md").read_text()
    assert "led to by: The post showed Helium." in (out / "actions.md").read_text()
    assert "The user explored Helium." in (out / "days" / f"{day_key(T0 + 10)}.md").read_text()
    index = (out / "MEMORY.md").read_text()
    assert "[[people/" in index and "[[things/" in index
    for markdown in out.rglob("*.md"):
        for link in re.findall(r"\[\[([^\]]+)\]\]", markdown.read_text()):
            assert (out / f"{link}.md").is_file(), (markdown, link)


async def test_identical_export_and_forgetting_commit_only_managed_files(store, tmp_path):
    oid = await _observation(store, "Theme settings", T0)
    cid = await _claim(store, oid, "The user prefers dark mode", kind="preference")
    out = tmp_path / "memory"
    out.mkdir()
    note = out / "my-notes.md"
    note.write_text("Keep this note.\n")
    first = await export_markdown(store, out)
    assert first["committed"] and first["commit_sha"]
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*.md")}
    second = await export_markdown(store, out)
    assert second == {"files_written": [], "files_removed": [], "committed": False, "commit_sha": None}
    assert before == {p.relative_to(out): p.read_bytes() for p in out.rglob("*.md")}

    await store.forget_claim(cid)
    third = await export_markdown(store, out)
    assert third["committed"] and third["commit_sha"] != first["commit_sha"]
    assert "The user prefers dark mode" not in (out / "beliefs.md").read_text()
    assert note.read_text() == "Keep this note.\n"
    assert "my-notes.md" not in _tracked_files(out)


async def test_retired_entity_page_is_removed_but_neighboring_note_survives(store, tmp_path):
    entity = await store.create_entity("Old name", "software", "old-name")
    out = tmp_path / "memory"
    first = await export_markdown(store, out, git=False)
    old_page = next(path for path in first["files_written"] if path.startswith("entities/"))
    note = out / "entities" / "my-note.md"
    note.write_text("Keep this note.\n")

    await store.execute("UPDATE entities SET merged_into=? WHERE id=?", (entity, entity))
    second = await export_markdown(store, out, git=False)

    assert old_page in second["files_removed"]
    assert not (out / old_page).exists()
    assert note.read_text() == "Keep this note.\n"
