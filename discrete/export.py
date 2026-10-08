"""Render the living store as a small, reviewable Markdown memory repository.

The database remains authoritative: these pages are a deterministic mirror, so a
Git diff shows changed beliefs instead of formatting churn. Wiki links make the
same files navigable in Obsidian. A manifest limits cleanup to files this exporter
created, leaving notes placed beside the mirror alone.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from .process.profile import render_profile
from .store import Store
from .timeutil import day_key, to_local

_ROOT_FILES = {"MEMORY.md", "beliefs.md", "actions.md", "changes.md", ".discrete-export"}


def _slug(name: str, entity_id: int) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-") or "entity"
    return f"{stem}-{entity_id}"


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace("[[", "[ [")


def _date(ts: float) -> str:
    return day_key(ts) if ts else "unknown"


def _source(observations: list) -> str:
    labels = []
    for obs in sorted(observations, key=lambda row: (row["started_at"], row["id"])):
        app = _text(obs["app"] or obs["site"] or "observation")
        title = _text(obs["window_title"] or obs["url"] or obs["caption"] or "untitled")
        label = f"{app} — {title}"
        if label not in labels:
            labels.append(label)
    return ", ".join(labels[:3]) or "memory store"


def _meta(claim, source: str, *, strength=None) -> str:
    parts = []
    if strength:
        parts += [f"strength: {strength.tier}, {strength.describe()}",
                  f"first: {_date(strength.first)}", f"last: {_date(strength.last)}"]
    parts += [f"id: #{claim['id']}", f"source: {source}", f"added: {_date(claim['created_at'])}"]
    return f"[{'; '.join(parts)}]"


def _bullet(claim, source: str, links: list[str] = (), *, strength=None) -> str:
    flag = f" **{claim['flag']}**" if claim["flag"] in ("stale", "contested") else ""
    refs = " " + " ".join(f"[[{path}]]" for path in links) if links else ""
    return f"- {_text(claim['text'])}{flag}{refs} {_meta(claim, source, strength=strength)}"


def _page(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", "", *lines]).rstrip() + "\n"


def _run_git(out_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(out_dir), *args], capture_output=True, text=True, check=False)


def _previous_files(out_dir: Path) -> set[str]:
    marker = out_dir / ".discrete-export"
    if not marker.is_file():
        return set()
    try:
        paths = json.loads(marker.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return set()
    allowed = set()
    for path in paths:
        if not isinstance(path, str):
            continue
        p = Path(path)
        if not p.is_absolute() and ".." not in p.parts and (
            path in _ROOT_FILES or (len(p.parts) == 2 and p.suffix == ".md"
                                    and p.parts[0] in {"days", "entities", "people", "things"})
        ):
            allowed.add(path)
    return allowed


def _write_pages(out_dir: Path, pages: dict[str, str], git: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    previous = _previous_files(out_dir)
    written = []
    for path, content in sorted(pages.items()):
        target = out_dir / path
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Export path leaves output directory: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
            written.append(path)
    removed = []
    for path in sorted(previous - pages.keys()):
        target = out_dir / path
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Export path leaves output directory: {path}")
        if target.is_file():
            target.unlink()
            removed.append(path)

    committed = False
    sha = None
    if git and shutil.which("git"):
        if not (out_dir / ".git").exists():
            _run_git(out_dir, "init", "-q")
        tracked_removed = [path for path in removed if _run_git(out_dir, "ls-files", "--error-unmatch", path).returncode == 0]
        staged = sorted(set(pages) | set(tracked_removed))
        if staged:
            added = _run_git(out_dir, "add", "-A", "--", *staged)
            if added.returncode == 0:
                changed = _run_git(out_dir, "diff", "--cached", "--name-only", "--", *staged).stdout.splitlines()
                if changed:
                    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
                    message = f"memory export {stamp}: {len(changed)} files changed"
                    result = _run_git(out_dir, "-c", "user.name=DIscrete", "-c", "user.email=discrete@localhost",
                                      "commit", "-q", "--only", "-m", message, "--", *changed)
                    committed = result.returncode == 0
                    if committed:
                        sha = _run_git(out_dir, "rev-parse", "HEAD").stdout.strip()
    return {"files_written": written, "files_removed": removed, "committed": committed, "commit_sha": sha}


async def export_markdown(store: Store, out_dir: Path, *, git: bool = True, allow_plaintext: bool = False) -> dict:
    """Write the store's current memory pages and optionally commit their diff."""
    if getattr(store.conn, "_discrete_encrypted", False) and not allow_plaintext:
        from .encryption import EncryptionError

        raise EncryptionError("Markdown exports contain plaintext. Use `discrete export --allow-plaintext` "
                              "to explicitly export an encrypted store. Automatic Markdown exports are disabled.")
    claims = sorted(await store.live_claims(), key=lambda row: row["id"])
    strengths = await store.strengths([row["id"] for row in claims])
    evidence = {row["id"]: await store.evidence_for_claim(row["id"]) for row in claims}
    observation_ids = sorted({item["observation_id"] for rows in evidence.values() for item in rows})
    links_into = await store.links_into("claim", [row["id"] for row in claims if row["action"]])
    observation_ids += [row["src_id"] for row in links_into if row["src_kind"] == "observation"]
    observations = {row["id"]: row for row in await store.get_observations(sorted(set(observation_ids)))}
    sources = {cid: _source([observations[item["observation_id"]] for item in rows
                             if item["relation"] == "supports" and item["observation_id"] in observations])
               for cid, rows in evidence.items()}

    entities = sorted(await store.graph_overview(limit=1_000_000), key=lambda row: row["id"])
    paths = {row["id"]: f"entities/{_slug(row['name'], row['id'])}" for row in entities}
    mentions: dict[int, list[str]] = {}
    pages: dict[str, str] = {}
    for item in entities:
        entity = await store.get_entity(item["id"])
        if entity is None:
            continue
        path = paths[item["id"]]
        related_claims = await store.claims_for_entity(item["id"], limit=max(30, len(claims)))
        for claim in related_claims:
            mentions.setdefault(claim["id"], []).append(path)
        neighbors = await store.neighbors(item["id"], limit=max(8, len(entities)))
        related = [paths[row["id"]] for row in neighbors if row["id"] in paths]
        lines = [_text(entity["summary"] or "No summary yet."), "", f"Kind: {_text(entity['kind'])}", "", "## Memories", ""]
        lines += [_bullet(claim, sources.get(claim["id"], "memory store")) for claim in related_claims]
        if related:
            lines += ["", "Related: " + ", ".join(f"[[{ref}]]" for ref in related)]
        pages[f"{path}.md"] = _page(entity["name"], lines)
        group = "people" if entity["kind"] in {"person", "people"} else "things"
        pages[f"{group}/{Path(path).name}.md"] = _page(entity["name"], [f"See [[{path}]]."])

    lasting = [row for row in claims if row["kind"] in {"pattern", "preference", "fact"}]
    belief_lines = []
    for kind in ("pattern", "preference", "fact"):
        belief_lines += [f"## {kind.title()}s", ""]
        belief_lines += [_bullet(row, sources[row["id"]], sorted(mentions.get(row["id"], [])),
                                 strength=strengths.get(row["id"]))
                         for row in lasting if row["kind"] == kind]
        belief_lines.append("")
    pages["beliefs.md"] = _page("Beliefs", belief_lines)

    actions = sorted((row for row in claims if row["action"]),
                     key=lambda row: (row["occurred_from"], row["id"]), reverse=True)
    by_action: dict[int, list] = {}
    for link in links_into:
        if link["relation"] in {"led_to", "step"}:
            by_action.setdefault(link["dst_id"], []).append(link)
    action_lines = []
    for claim in actions:
        action_lines += [_bullet(claim, sources[claim["id"]], sorted(mentions.get(claim["id"], []))),
                         f"  - {_date(claim['occurred_from'])} · {_text(claim['action'])}"]
        for link in sorted(by_action.get(claim["id"], []), key=lambda row: (row["relation"], row["src_id"])):
            obs = observations.get(link["src_id"]) if link["src_kind"] == "observation" else None
            source = _source([obs]) if obs else "memory store"
            time = to_local(obs["started_at"]).isoformat(timespec="minutes") if obs else "unknown time"
            role = "led to by" if link["relation"] == "led_to" else "step"
            action_lines.append(f"  - {role}: {_text(link['reason'])} — {source}, {time}")
    pages["actions.md"] = _page("Completed actions", action_lines)

    events = [row for row in claims if row["kind"] == "event"]
    days = set(await store.days_with_episodes()) | {day_key(row["occurred_from"]) for row in events}
    for day in sorted(days):
        digest = await store.get_digest(day)
        day_lines = [_text(digest["text"]) if digest else "No digest yet.", "", "## Events", ""]
        day_events = sorted((row for row in events if day_key(row["occurred_from"]) == day),
                            key=lambda row: (row["occurred_from"], row["id"]))
        for claim in day_events:
            time = to_local(claim["occurred_from"]).strftime("%H:%M")
            day_lines.append(f"- {time} {_text(claim['text'])} {_meta(claim, sources[claim['id']])}")
        pages[f"days/{day}.md"] = _page(day, day_lines)

    changes = await store.recent_events(limit=200)
    change_lines = [f"- {to_local(row['at']).isoformat(timespec='seconds')} {_text(row['op'])}"
                    f" · {_text(row['actor'])} · #{row['claim_id'] if row['claim_id'] is not None else '—'}"
                    f" · {_text(row['reason'])} [source: memory log; added: {_date(row['at'])}]"
                    for row in changes]
    pages["changes.md"] = _page("Changes", change_lines)

    profile = render_profile(await store.get_profile())
    index = ["A readable mirror of DIscrete's current memory. Edit the underlying store; Git records each export.", ""]
    if profile:
        index += ["## Profile", "", profile, ""]
    index += ["## Browse", "", "- [[beliefs]] — lasting memories", "- [[actions]] — completed actions",
              "- [[changes]] — recent changes", "", "## Recent days", ""]
    index += [f"- [[days/{day}]]" for day in sorted(days, reverse=True)[:40]]
    index += ["", "## People and things", ""]
    top_entities = sorted(entities, key=lambda row: (-row["memories"], -row["links"], row["id"]))[:12]
    for entity in top_entities:
        group = "people" if entity["kind"] in {"person", "people"} else "things"
        index.append(f"- [[{group}/{Path(paths[entity['id']]).name}]] — {_text(entity['name'])}")
    pages["MEMORY.md"] = _page("Memory", index)

    pages[".discrete-export"] = json.dumps({"files": sorted(pages | {".discrete-export": ""})},
                                             indent=2, ensure_ascii=False) + "\n"
    return _write_pages(out_dir, pages, git)
