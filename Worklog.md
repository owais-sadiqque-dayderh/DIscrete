# Worklog — exp_discrete (Jarvis memory, built on DIscrete)

Running log for improving DIscrete with ideas from Hindsight, Honcho, mem0, Screenpipe, Cognition and others.
Times are IST (UTC+5:30). The earlier work (the Hindsight experiment, Phases 1–20) is in the closed log
[`COMP_001/Worklog.md`](../../Worklog.md); read its Phases 15–20 for how we got here.

**Goal:** a memory that keeps watching what the user does and turns it into a trustworthy picture of who they
are: habits, preferences, projects and people. An action model will consume it later. DIscrete is the base
because its claims are grounded, verified and scoped to the owner; it was better in live use (closed log,
Phases 18–19).

---

## Status at a glance

| Item | State |
|---|---|
| `exp_discrete/` created from `Discrete/` (working tree incl. uncommitted changes, own `.git` history) | ✅ 2026-10-06 |
| Separate data dir `~/.cache/exp_discrete` seeded with a snapshot of the live db | ✅ |
| Own venv, tests | ✅ 112 passing |
| Feature plan (dreaming, firmer beliefs + cleanup, graph, version history) | 💬 Under discussion (below) |
| Disk space | ⚠️ Fell to ~270 MB free during setup; ~6.9 GB free a few minutes later (macOS reclaimed purgeable space). Still 97% full |
| OpenRouter credit | ⚠️ ~$3.35 left (shared with live DIscrete) |

---

## Workspace

| What | Where |
|---|---|
| Code | `COMP_001/exp_hindsight/exp_discrete/` (same layout as `Discrete/`) |
| Live, untouched copy | `COMP_001/Discrete/` (what `jarvis` runs) |
| Data | `~/.cache/exp_discrete/discrete.db` (snapshot of the live db, 2026-10-06 ~21:00); frame paths still point at `~/.cache/discrete/frames` |
| Config | `exp_discrete/.env` (copy of the live `.env`; only `DISCRETE_DATA_DIR` changed) |
| Tests | `.venv/bin/python -m pytest -q` |
| Run a command against the snapshot | `.venv/bin/discrete process`, `consolidate`, `ask "…" --records`, `query "…"`, `stats` |

Don't run `exp_discrete` and live DIscrete's `run` at the same time: both grab Option+Tab and both pay for
captions. Feature work on memory (dreaming, cleanup, graph) runs offline on the snapshot and needs no capture.

---

## Baseline: what DIscrete already has (audited 2026-10-06)

This audit shows where each proposed feature starts from. Code paths are relative to `discrete/`.

| Area | What exists | Gap |
|---|---|---|
| Write-time reconcile (`process/pipeline.py`, `store/drafts.py`) | Each new claim is compared with similar live claims: **same** adds evidence to the old claim and raises its confidence; **contradicts** invalidates the old claim (`valid_to`, `invalidates_id`); **refines** links with `refines_id` | Only the top few similar claims at write time are compared; nothing re-checks old contradictions later |
| Belief strength (`confidence.py`, `store/claims.py::_recompute_confidence`) | Confidence comes from the number of distinct episodes and days, and from the best modality per episode (stated > engaged > passive) | No decay; no "established" vs "tentative" tier; repeated **events** never become a habit or preference |
| Consolidation (`process/consolidate.py`, hourly or `DISCRETE_CONSOLIDATION_INTERVAL_MIN`) | Clusters near-duplicate **facts and preferences** (cosine ≥ `CONSOLIDATE_SIM`, plus refines links), has the LLM rewrite each cluster into a non-redundant set, has the verifier check it, and writes the result as a new version in the same lineage; also entity pages and day digests | Events (35 of the 42 live claims) are never looked at; no step that finds patterns across sessions |
| Cleanup (`_lint`) | Supersedes claims that stay unsupported past `UNSUPPORTED_TTL_DAYS`; repairs index gaps; counts lineage anomalies | No outdated-preference sweep; no user "forget / that's wrong" |
| Version data (`claims` table) | `lineage_id`, `version`, `derived_from`, `superseded_by`, `refines_id`, `invalidates_id`, `valid_from`/`valid_to`, `status`. Old rows are kept; nothing is hard-deleted | No change log with a reason and actor; no `history` or `revert` command; nothing shows the user what changed |
| Entities (`process/entities.py`, `store/entities.py`) | Candidates from AX text, the VLM and claims; promoted when seen in ≥ `ENTITY_MIN_EPISODES` episodes; alias resolution (exact → key → fuzzy → LLM adjudication → embedding); entity pages; `mentions` (entity↔claim) | On the live snapshot: **0 entities promoted** from 80 candidates. The `entity_edges` table exists but **no code writes it** (0 rows). Retrieval doesn't expand through entities |
| Privacy | Denied apps and sites (closed log, Phase 19), self-capture tagging | **No secret redaction** of OCR/AX text (Hindsight's agent has it) |
| Live data volume | 251 captures, 42 live claims (35 events, 3 facts, 4 preferences) | Too little to test dreaming on; the 46-question bench session or a few more days of capture is needed |

---

## Phase 0 — Setup (2026-10-06, ~20:45–21:00 IST)

- Copied `Discrete/` → `exp_hindsight/exp_discrete/` with `rsync`, leaving out `.venv`, `__pycache__`, the
  pytest and ruff caches, `*.egg-info` and `.DS_Store`. The copy keeps `.git` (history up to `ff5a268`), and the
  uncommitted live-DIscrete changes come along as uncommitted changes (block lists, `CONSOLIDATION_INTERVAL_MIN`).
  A baseline commit in `exp_discrete` would make later diffs clean; that's waiting for the user's go-ahead.
- `.env`: `DISCRETE_DATA_DIR=~/.cache/exp_discrete`. The live db was snapshotted with `sqlite3 .backup`, which
  is safe while DIscrete runs.
- **Disk full:** `pip install` failed with `ENOSPC` (272 MB free). Removed the partial venv and a frames copy,
  then built the venv with `uv venv` + `uv pip install --link-mode clone -e '.[dev]'`; APFS clones take almost
  no space. 112 tests pass.

---

## Feature plan (under discussion)

Sources: Cognition "agent memory repo" (cognition.com/agent-memory-repo), Hindsight (code in
`exp_hindsight/`), mem0, Honcho, Screenpipe, Graphiti, and lessons from the closed log.

### F1. Dreaming: find patterns across sessions → new beliefs
- **What:** a periodic job, separate from the hourly merge. It reads a rolling window of events, episodes and
  day digests (e.g. 14 days) and proposes **pattern** claims ("works on the PARSEC memory project most
  evenings", "checks cricket scores on the phone app, not the web"). Each proposal cites the event claims it
  comes from (`derived_from`).
- **Guards learned in Hindsight (closed log, Phases 13 and 17):**
  - The span check is in code: ≥ K distinct days or episodes.
  - The verifier is a different model (`VERIFY_LM`); muse approved all of its own dream insights.
  - Text the user didn't produce (agent output, PDFs, other people's messages) doesn't count as support.
  - Old patterns are audited each run: confirm, weaken or retire.
- **Why first:** it's the only planned feature that produces new knowledge about the user. Everything else
  tidies what is already stored.

### F2. Firmer beliefs over time + cleanup (Cognition's "clean up memory")
- **Strengthen:** confidence already rises with repeated evidence. Add an explicit tier (tentative →
  established once seen on ≥ N days in ≥ M episodes) and keep `first_seen`/`last_seen`/`n_days` for Bubble to
  phrase it ("on 6 days since Sep 30"). The claim text stays as it is, so it can't drift.
- **Merge duplicates:** already done for facts and preferences. Dreaming does the equivalent for events
  (repeated events → one pattern; the events themselves stay as dated history).
- **Remove outdated:** preferences and patterns that go unreinforced for a while lose weight (decay), and get
  retired when contradicted by newer engaged or stated evidence. Events never decay; they are history.
- **Check sources to resolve contradictions:** a sweep that finds pairs of live claims which can't both be true
  (same entity or topic), rereads their evidence (observations) and keeps the better-supported current one
  (newer, engaged or stated over passive), with the reason recorded.
- **User corrections** (from mem0 and Honcho): `jarvis forget <id>` and "that's wrong" in Bubble invalidate the
  claim and record a stated negative, so the same claim isn't re-learned.

### F3. Graph (relationship mapping)
- DIscrete already has half a graph: entities ↔ claims via `mentions`. What's missing:
  1. **Entities actually promoted** on real data. Today the gate is too strict or candidates don't recur;
     tune it on the snapshot.
  2. **Edges:** fill the existing `entity_edges` from co-occurrence in an episode or claim (deterministic,
     free).
  3. **Retrieval through the graph:** one-hop expansion from the entities in a question ("everything about
     the billing launch", "who did I talk to about X").
- **Typed relations** (person —works on→ project, Graphiti/mem0-graph style) only if the bench shows questions
  that need them. Extracting relations costs an LLM call per claim, and graph memory has mixed results on
  benchmarks.
- A graph view in the UI later would help the user inspect what was learned.

### F4. Version history (Cognition uses git; what do others do?)
- **A graph is not version control.** A graph says what is connected to what *now*. Version control says what
  we believed, when, why it changed, who changed it (draft, merge, dream or the user), and lets you undo it.
  The two are separate axes; a bi-temporal graph (Graphiti's `valid_at`/`invalid_at`) gives "as of" queries,
  but no reasons and no revert.
- **What others do:**
  - Cognition: memory as Markdown in a git repo, so it gets history, diff, blame and revert for free; git
    merges handle many agents writing at once.
  - Hindsight: an `observation_history` table that snapshots the old text before each update (bounded).
  - mem0: a per-memory history of ADD/UPDATE/DELETE events.
  - Graphiti: bi-temporal edges.
- **DIscrete already stores most of it** (lineage, version, derived_from, superseded_by, valid_from/to; no hard
  deletes). Proposal:
  - an append-only `claim_events` table (op, actor, reason, before/after ids, time);
  - `jarvis history <claim>`, `jarvis changes [--since]` and `jarvis revert <event>`.
  - Every automatic change (merge, dream, cleanup) then becomes reviewable and undoable, which is the
    safety net that makes autonomous dreaming acceptable.
- **Git itself:** not as the main store (vectors, FTS and continuous writes from capture don't fit). An
  optional **Markdown export** of the profile and beliefs into a git repo (Cognition-style) is cheap: a
  readable and editable mirror and a natural hand-off format for the action model.

### Other ideas worth considering
- **Profile document** (Honcho's peer representation, Hindsight's mental models): one living summary of who
  the user is, regenerated by dreaming from established beliefs. It's the input the action model would read.
- **Secret redaction** (Hindsight agent `redact.py`): DIscrete has none; this is a security fix, not a
  feature.
- **Deterministic time windows and the writer's completion rule** (closed log, Phase 17): small, proven wins.
- **Audio/meetings** (Screenpipe): a big missing input, but a large build; later.

### Proposed order (to be agreed)
1. P0: secret redaction; a dream/cleanup eval (the 46-question bench tests recall, not patterns); free disk.
2. P1: version history (`claim_events`, history/changes/revert), then dreaming v1, then user corrections.
3. P2: contradiction sweep + decay + belief tiers; profile document.
4. P3: graph: entity promotion fix, co-occurrence edges, one-hop retrieval; graph view.
5. Later: typed relations, Markdown/git mirror, audio.

---

## Known issues

1. **Disk 97% full** (fell to ~270 MB during setup, ~6.9 GB later). Live DIscrete writes frames and db pages and may fail. Candidates to free:
   `uv cache prune` (`~/.cache/uv` 11 GB), `~/.cache/codex-runtimes` (4.5 GB), `~/.cache/hindsight` (3.2 GB),
   `exp_hindsight/node_modules`. The user decides.
2. **Credit ~$3.35.** Dreaming development and evals need a top-up.
3. **Small snapshot:** 42 live claims are too few to develop dreaming on.
4. Leaked keys: rotate the HF and OpenRouter keys (closed log, Phase 18) if not done yet.

## Next steps

- [ ] Agree the feature list and order with the user.
- [ ] Baseline commit in `exp_discrete` (with the user's OK) so changes are diffable.
- [ ] Free disk; top up credit.
- [ ] Build the dream/cleanup eval from the 46-question session plus a few days of fresh capture.
