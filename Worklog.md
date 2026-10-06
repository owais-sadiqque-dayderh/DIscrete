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
| Feature plan (dreaming, firmer beliefs + cleanup, graph, version history) | ✅ Agreed with the user 2026-10-06 ~21:30; building on branch `feat/memory-evolution` |
| Branch `feat/memory-evolution` (baseline `main` = `41e2f9f`) | 🚧 Phases 1–4 + export built, 169 tests; **benchmarks running — see HANDOFF** |
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

## Phase 1 — Research, bench rebuild, branch (2026-10-06, ~21:30–23:30 IST)

- **User decisions:** agreed order (change log before dreaming); dreaming interval as a `.env` option; Markdown
  memory "whatever is good"; baseline commit yes (benchmarks before/after, fallback). Work on a branch, brief at the
  end with setup notes; the user decides on merging. New goals during the phase:
  - **deep "why" answers**: e.g. "why did I download Helium?" should give "you saw Theo's X post (his video showed
    the Helium browser), searched for helium, then downloaded it";
  - **audit the input side**: is the AX tree too long, OCR vs a VLM, is the schema right, is it enough for preferences;
  - the reliability bar of a large company; delegate sub-tasks to other agents if useful; pause and ask when a
    resource (credit, disk) runs out.
- **Git:** `main` = baseline commit `41e2f9f` (the copy as-is). Branch `feat/memory-evolution`. Baseline worktree
  `exp_hindsight/exp_discrete_main` (git-excluded from Hindsight) runs the old code on the same data
  (`PYTHONPATH=../exp_discrete_main`).
- **Research** (sources in the feature plan below):
  - **Cognition memory repo:** Markdown + git; dreaming = add patterns, merge duplicates, remove outdated, check
    sources.
  - **Honcho's dreamer:** asynchronous; explicit → deductive → inductive → abductive; conclusions build only on more
    certain ones; peer cards.
  - **Claude Code's Auto Dream:** needs ≥24 h and ≥5 sessions; 4 phases; prunes contradictions and stale entries.
  - **Letta:** sleep-time agents.
  - **mem0:** ADD/UPDATE/DELETE; its resolver silently deleted still-true memories; V3 dropped it; `history()` can't
    find deleted ids.
  - **Graphiti:** bi-temporal, invalidates rather than deletes.
  - **MoM/P-Mem (arXiv 2609.25054):** typed provenance (support / supersede / contest / revoke …); recovers
    committed errors that CRUD stores can't.
  - **A-MEM:** links and neighbour evolution.
  - **Screenpipe:** raw capture + pipes on schedules.
- **The bench session had been lost.** `~/.cache/discrete` was reset ~22:06 on 2026-10-06, so DIscrete's
  Sep 30/Oct 1 store behind the 46-question bench was gone. Rebuilt it:
  - Hindsight bank `bench-v0` still held every observation's full text (transcript v1) and all 132 frames.
  - `bench/export_hindsight_session.py` → `bench/private/g46/manifest.jsonl` (136 observations, 132 frames).
  - `tools/replay_session.py` replays it with any checkout's pipeline. The golden file is copied to
    `bench/private/g46/cases.yaml`.
  - `bench/private/` is git-ignored (real screen data).
  - `bench/run_g46.py`: the same judged harness as the closed log (DeepSeek v4 pro judge, DIscrete's answer-check
    prompt, expected strings). It scores any checkout (`--code`) on a copy of a db.
- **Second eval session:** `bench/export_discrete_session.py` turned the 2026-10-06 live snapshot into
  `bench/private/live1006/manifest.jsonl` (251 captures, 225 frames). A golden set with "why" questions is being
  written by a delegated agent.
- **Cost:** replaying 136 captures through DIscrete costs about $0.10 (captions dominate). Credit is $3.04.

## Phase 2 — Security and reliability

- **Secret redaction** (`capture/redact.py`, ported from the Hindsight agent with its tests, 25):
  - redacts screen text, titles and URLs in `ScreenObserver._capture`; a capture with a secret keeps no frame (its
    caption is written from the redacted text);
  - also redacts captions from the VLM and Bubble's field text;
  - `DISCRETE_REDACT_SECRETS` (default on). Capture-level test included.
- **Transient provider errors:** OpenRouter's upstream `404 model_not_found` ("Provider returned error") and
  502/503 are retried with backoff (`LLM_PROVIDER_RETRIES=3`), for streams only before the first word.
  - Seen live in this phase: the baseline replay lost the drafts of episodes 8 and 11 to it. DIscrete retried them
    only 15 minutes later, and a replay never.
  - A wrong model name (no "provider" in the error) is not retried.

## Phase 3 — Change log, corrections, causal trails, actions

- **Change log** (`store/history.py`, migration 7, `memory_events` + `memory_event_refs`). Every decision is an
  event with actor, reason, and each touched claim's before/after:
  - draft add / reinforce (with the evidence it added) / refine / invalidate / contest;
  - consolidation merge;
  - unsupported expiry;
  - user forget / correct / gap answer;
  - dream / trail / profile / cleanup (Phase 4).
  - `revert_event` undoes any of them (a revert is an event, so it can be reverted). It refuses when later events
    touched the same claims, unless `--force`. Nothing is hard-deleted, so the data for every version already
    existed; the log adds why and undo.
  - CLI: `history`, `changes`, `revert`.
- **Forget/correct** (`store/corrections.py`):
  - Forget retires the whole lineage and adds a suppression (text + vector). Drafting then drops a claim that
    matches it (norm text, Jaccard ≥ 0.8 or cosine ≥ 0.90), so it isn't re-learned.
  - Correct writes a `stated` version in the same lineage.
- **Completed actions are always drafted.** In the live snapshot, episode 5 drafted 5 events and none was the
  Helium download/install, its most important action (cap of 5). Now:
  - the prompt puts actions with consequences first, with an `action` verb (`claims.action`), under a separate cap
    (6);
  - long episodes get +1 event per 3 min past 5 (cap 10);
  - an event's time is its cited moments, not the whole episode.
- **Causal trails** (`process/trails.py`, `store/links.py`). For each action:
  - the drafting model reads the 30 minutes before it and proposes trigger/step moments;
  - code keeps only moments that precede the action, whose quote is really on that screen and names the action's
    distinctive term (e.g. "helium", picked by rarity, action verbs and their inflections excluded);
  - the verifier then confirms each.
  - Links (`led_to` / `step`) + a `trail` event; revertible.
- **"Why" retrieval:**
  - "why did I…", "what made me…" plan as intent `why`;
  - `expand_why` puts the trail right after the matched action, or without one the moments just before it naming
    the same thing (left to the support check);
  - the support check keeps pre-verified trail links only next to a kept action;
  - writer mode `why`: a hedged, time-ordered reason, never a stated motive.

## Phase 4 — Dreaming, beliefs, profile, graph

- **Dreaming** (`process/dream.py`, `store/beliefs.py`). Runs when both `DREAM_INTERVAL_MIN` (default 1440) has
  passed and `DREAM_MIN_NEW_EPISODES` (5) episodes closed, or with `discrete dream`. Steps are independent:
  1. **Patterns:**
     - categories: project, interest, habit, tool, relationship, preference;
     - code gates: cited ids exist, support from ≥ `DREAM_MIN_EPISODES` (3) episodes, habits/preferences also
       ≥ `DREAM_MIN_DAYS` (2) days, frequency words need ≥ 5 days, never resting on another pattern, starts "The
       user", atomic;
     - then the independent verifier;
     - a restated pattern (cosine ≥ 0.85) confirms the old one.
     - A pattern is a claim (`kind='pattern'`) with `derived_from` and its supports' evidence, so strength and
       confidence come from the same episode/day count.
  2. **Audit:** confirm / retire existing patterns (verified); patterns whose support was all withdrawn retire.
  3. **Contradiction sweep:**
     - checks lasting memories with cosine 0.55–0.82, each pair asked once (stored as `checked` links);
     - a conflict goes to the stated > engaged > passive, then newer, claim; the loser is invalidated with the
       reason;
     - too close to call → both flagged contested.
  4. **Staleness:** facts, preferences and patterns unseen for 45 days are flagged stale (cleared when seen again);
     patterns unseen for 120 days retire; events never go stale.
  5. **Profile** (`process/profile.py`):
     - sectioned (work, tools, interests, habits, preferences, people);
     - every point cites memories, is checked in code and by the verifier;
     - stored in meta, revertible (`discrete profile`).
- **Belief strength:** `store.strengths()` gives episodes, days, first/last and tier (tentative / recurring /
  established). Shown on lasting claims in Bubble's records, and the writer is told to word tentative ones as recent.
  Questions about the user as a whole (`about_user` in the plan) get the profile as a record.
- **Graph:**
  - entities are promoted and `entity_edges` rebuilt every `ENTITY_INTERVAL_MIN` (10) and at each merge. The live
    snapshot had 0 entities: 80 candidates (Arc in 7 episodes, Helium in 3) were waiting for an hourly merge that a
    40-minute session never reached. `entity_edges` was never written by any code.
  - Search adds memories mentioning entities the question names as one more ranking list (`ENTITY_RECALL`), with
    `GRAPH_EXPAND` also their neighbours'.
  - `discrete graph [NAME]`.
- **jarvis for the branch:** `exp_discrete/jarvis` (state in `~/.cache/jarvis-exp_discrete`) with profile, patterns,
  dream, changes, history, forget, correct, revert, graph, export; `search` replaces `why` for debugging.
- **Tests:** 166 passing (history 7, trails 6, dream 14, redaction 26, provider retry 1, plus the original suite);
  ruff clean.
- **Delegated (running):**
  - input-source and schema audit (Claude Opus) → `docs/audit_inputs.md`;
  - Markdown/Obsidian + git export (Codex GPT-6-Sol) → `discrete/export.py`;
  - live-session golden questions incl. "why" (Claude Opus) → `bench/private/live1006/cases.yaml`.

## Phase 5 — Baseline score, the input audit, fixes from it (2026-10-07, ~00:15–01:30 IST)

- **Session break:** the previous session ended at about 00:14 and killed two rebuilds mid-run (branch g46,
  live baseline). Their partial dbs were deleted and the rebuilds restarted with `nohup`, so a session end no longer
  kills them. Each writes `DONE` at the end of its log.
- **Baseline score on the rebuilt g46 session** (main code, `results/g46-base.json`):

  | Metric | Score |
  |---|---|
  | overall | **0.652** (30/46) |
  | answers | 0.659 |
  | refusals | 0.6 |
  | supported | 0.717 |
  | hit@8 | 0.707 |
  | MRR | 0.386 |
  | p50 latency | 15.9 s |
  | cost for 46 answers | $0.10 |

  By category: aggregate 2/2, enumerate 0/2, profile 1/3, recall 22/30, temporal 2/4, unanswerable 3/5.
  - This is lower than the closed log's 0.717 on the original Sep 30/Oct 1 store. That store is gone; this one is
    rebuilt from the same raw captures with new captions/drafts and muse as verifier. Only same-db comparisons
    count from here.
- **Verifier model (the user's question).** Every DIscrete `.env` already sets `VERIFY_LM_MODEL_NAME` to muse;
  Gemini appears only in an old comment on `.env` line 36. So all runs here are muse end to end. An ablation is
  running (`g46/branch-geminiverify.db`, `VERIFY_LM_MODEL_NAME=google/gemini-3.8-flash` in the env) to measure
  whether an independent verifier matters. That matters most for the new self-checks: the dream verifier, the trail
  verifier, the profile.
- **Input audit delivered** (`docs/audit_inputs.md`, committed `11c1934`; delegated to Claude Opus, no paid calls).
  Short answers:
  - **AX:** the right primary source and not too long (median 1.7k chars, 1 row in 227 truncated), but noisy: 43%
    single-word lines, 15% duplicate lines, median 26% of AX tokens not visible on screen. A hidden Notes preview
    produced 4 fake "preferences".
  - **OCR:** good (local, ~350 ms). It read "Helium" off Theo's video, but loses small text because it runs after
    the frame is shrunk to 1600 px.
  - **Captions:** 12/14 accurate, no hallucinated content, but 81% of claim evidence quotes the caption, not screen
    text.
  - **Don't replace AX/OCR with a VLM** (~30× the cost, loses exact text).
  - **Schema:** sound core. Missing: sightings/dwell (attention), pages as objects, text provenance, revealed
    choices/signals, tasks above episodes, preference slots.
  - **Root cause of the Helium miss: the drafter read only an episode's last 40 observations**
    (`rows[-DRAFT_MAX_OBS:]`). Episode 5 had 92, so 52 rows (the whole Helium chain) never reached the model.
    Phase 3's "5-event cap" diagnosis was wrong.
  - **Security:** obs 239 of the live snapshot shows `.env` in TextEdit with OpenRouter keys (before redaction
    existed). **The user should rotate the OpenRouter key** and delete that row and frame from
    `~/.cache/discrete` / `~/.cache/exp_discrete`.
- **Fixed from the audit** (commit `28af5f4`):
  1. **Drafting reads every observation:** episodes close at `EPISODE_MAX_OBS` (40), and longer ones are drafted in
     consecutive chunks (claims combined, caps scaled).
  2. **DIscrete's own output** (a terminal running jarvis/discrete commands, printed `[claim N ·` records, its
     db in a viewer) is classified as agent output. The replay tool now applies `classify_origin` like live capture.
  3. **Redaction precision.** The catch-all for unnamed key material was hitting Google Docs ids and Flights search
     tokens, and every hit dropped the frame. Now only confident credential matches drop a frame; the catch-all
     masks text only and never runs on URLs. On the two sessions only the two `.env` captures lose their frame.
     `bench/redact_manifest.py` applies capture redaction to replay manifests (they were sending the key to the
     caption model again).
- **Delegated now** (Codex GPT-6-Sol, separate files, uncommitted until reviewed):
  - audit #4 (clean AX at the source, visible-only text) + #7 (OCR at native resolution) → `capture/*.py`;
  - audit #10 (deterministic action and choice detectors) → `process/actions.py`.
- **Also done in Phase 5:**
  - audit #6: chrome-free text for the drafter and caption model; `DRAFT_TEXT_CHARS_PER_OBS` 1200 → 2000;
  - audit #11: the caption prompt separates a page from embedded media and names brands and apps inside media;
  - audit #9: **sightings**, migration 8. Every sighting is recorded, duplicates included. Attention = dwell until
    the next sighting (capped at 180 s); active = input. `discrete attention`, time questions get a totals record,
    and dreaming and the profile read it as context;
  - audit #5: **evidence provenance**, migration 9 (`evidence.field`); the drafter prefers screen-text quotes;
  - tests for the episode cap and chunked drafting.
  - Suite: 204 tests (including delegated agents' uncommitted tests), ruff clean.
- **Replay made parallel** (`tools/replay_session.py`): captions run `DISCRETE_ENRICH_CONCURRENCY` at a time, then
  segmentation and drafting run in capture order. The result is identical; 20 captures took 69 s instead of about
  5 min.
  - The outdated branch rebuilds were stopped. Only `live1006/base.db` (main code via a copy of the fast tool at
    `/tmp/replay_session_fast.py`) is running; its log ends in `DONE`.
  - **Final rebuilds to run once the action detectors are wired in:** `g46/branch.db`,
    `g46/branch-geminiverify.db` (`VERIFY_LM_MODEL_NAME=google/gemini-3.8-flash`), `live1006/branch.db`. Use
    `DISCRETE_ENRICH_CONCURRENCY=6 .venv/bin/python tools/replay_session.py --manifest … --db …`, then
    `discrete --db … process --recover --consolidate`, then the HANDOFF eval protocol.
- **Action and choice detectors** (audit #10, Codex; commit `45fe056`): quote-backed rules. On the live rows:
  5/5 correct and none wrong (Helium download, `.dmg`, install, DuckDuckGo chosen, Helium as default), 0.14 ms per
  row. Their hints follow the drafter's observations as leads it must still quote. New action verb `chose`.
- **Final rebuilds launched at `45fe056`** (`DISCRETE_ENRICH_CONCURRENCY=4`): `g46/branch.db`,
  `g46/branch-geminiverify.db`, `live1006/branch.db`, plus `live1006/base.db` (main). A background watcher
  reports when every log ends in `DONE`.
- **Then run `bench/run_matrix.sh`.** It dreams on copies, then scores g46-base / g46-branch /
  g46-branch-geminiverify / g46-base+branchretrieval (drafting held fixed) / live-base / live-branch, and prints both
  comparisons.
- **Capture text, round 1** (audit #4/#7, Codex; commit `f2fdff5`):
  - adjacent runs merged; descriptions only without text children;
  - text views read through `AXVisibleCharacterRange` (Terminal 16,000 → 1,199 chars);
  - nav landmarks skipped, offscreen text culled;
  - OCR on native pixels (+56 ms median).
  - My review added: the native copy is capped at 2× the storage width and kept only on the newest cached frame
    (memory).
  - **Arc unchanged** (38.6% single-word lines). Round 2 delegated (Codex): Chromium text markers / line grouping.
    Capture changes affect live capture only; replays reuse stored text.
- **Delegated, running:**
  - capture text round 2, Chromium/Arc (Codex);
  - an independent code review of the whole branch (Claude Opus, report-only).
    taskId `node:delegated-task:command%3Amcp%3A0d191c62-be5f-41f9-bdd3-072cde8bb4de%3Adelegate-task%3Aexp-discrete-branch-review-r1`.
- **Next (me):**
  - audit #6 (filter chrome before the drafter and the caption);
  - #11 (caption prompt: separate the page from embedded media, name brands in media);
  - #5 (evidence provenance; lower weight for caption-only quotes);
  - #9 (sightings, active dwell, pages; migration 8);
  - then rebuild both sessions and score: baseline vs branch vs branch with the Gemini verifier.

---

## ▶ HANDOFF — read this first if you are picking up (written 2026-10-07 ~00:30 IST)

The previous agent (Claude Opus 5.5, user's main subscription) hit its time limit. The user wants another Claude to
carry on, delegating sub-tasks to Codex/other Claude agents where useful (`delegate_task` via the T3 orchestrator;
`orchestrator_capabilities` lists providers: `codex` gpt-6-sol / gpt-6.1-sol, `claudeAgent` opus-5-5, etc.).

**User's standing instructions** (all still apply):
- Build on DIscrete in this folder only, on branch `feat/memory-evolution`. `main` = baseline `41e2f9f`. Never
  touch `COMP_001/Discrete/` (the user's live copy, run by `COMP_001/jarvis`).
- Research what others do, implement what's good, innovate where not. Experiment, test, **evaluate against the
  baseline**, log everything in this file, move on. Elite production quality (the reliability bar of a large
  company).
- Goals: deep "why" answers (the Helium example); learn the user's preferences, habits, projects and interests from
  screen activity; the audit of AX / OCR / VLM / schema; the 4 features (dreaming, belief strengthening + cleanup,
  graph, version history).
- At the end: brief the user on what changed and what didn't, with numbers, and give **setup notes to test the
  branch**. The user decides on merging. Don't merge, push or open PRs. Commit on the branch only, with explicit
  paths (`git add <files>`, never `-A`: delegated agents write files concurrently).
- Resources: don't compromise on development, but pause and tell the user when a resource runs out (OpenRouter
  credit was $3.04 at 23:30 on 2026-10-06; check `curl -s https://openrouter.ai/api/v1/credits -H "Authorization:
  Bearer $CORE_LM_API_KEY"`; disk was 95% full).
- Use DIscrete's exact models for every comparison (`.env`): muse-spark-1.3 (caption/insight/core, minimal
  reasoning), gemini-3.8-flash verifier, gpt-4o-mini utility, text-embedding-3-small, cohere rerank-4-fast. Judge:
  deepseek-v4-pro.

**State of the code** (`git log --oneline main..feat/memory-evolution`, 13 commits, 169 tests pass, ruff clean):
see Phases 2–4 above. Plus:
- the Markdown export (`discrete/export.py`, by a Codex agent; wired as `discrete export`, and auto-run after each
  dream);
- the why fallback for any matched moment (commit `54b2afe`).

**Jobs that were running at handoff** (superseded by Phase 5: rebuilds restarted with nohup; logs end in DONE) (all write under `bench/private/`; check whether they finished):
1. **g46 baseline score:** `bench/private/g46/eval-base.log` → `bench/private/results/g46-base.json`. It scores the
   main code on `g46/base.db`.
2. **g46 branch rebuild:** `bench/private/g46/replay-branch.log` → `g46/branch.db`. It replays with the branch
   pipeline (action capture, precise event times, trails), then `process --recover --consolidate`.
3. **Live-session baseline rebuild:** `bench/private/live1006/replay-base.log` → `live1006/base.db`, with main code.
4. **Delegated input/schema audit** (Claude Opus, still running): `task_status` with taskId
   `node:delegated-task:command%3Amcp%3A5a593636-4fc7-4dd7-be2d-b82cb7825024%3Adelegate-task%3Aexp-discrete-input-audit-r1`.
   Output goes to `docs/audit_inputs.md` and `bench/audit/` (uncommitted; review, then commit with explicit paths).

**Evaluation protocol (to do next).** All commands run from `exp_discrete/`:
```bash
# A. Baseline vs branch on the 46-question session (Sep 30/Oct 1)
.venv/bin/python bench/run_g46.py --db bench/private/g46/base.db --name g46-base --code ../exp_discrete_main  # if not done
cp bench/private/g46/branch.db /tmp/g46-branch-dream.db                                        # never dream on the original
.venv/bin/python -m discrete.cli --db /tmp/g46-branch-dream.db dream --force                   # patterns, conflicts, profile
.venv/bin/python bench/run_g46.py --db /tmp/g46-branch-dream.db --name g46-branch
.venv/bin/python bench/run_g46.py --compare g46-base g46-branch
#    ablation: is a change from drafting or from retrieval/dream?
cp bench/private/g46/base.db /tmp/g46-base-plus.db && .venv/bin/python -m discrete.cli --db /tmp/g46-base-plus.db dream --force
.venv/bin/python bench/run_g46.py --db /tmp/g46-base-plus.db --name g46-base+dream

# B. Live session of 2026-10-06 (30 questions incl. 6 "why"): bench/private/live1006/cases.yaml
.venv/bin/python tools/replay_session.py --manifest bench/private/live1006/manifest.jsonl --db bench/private/live1006/branch.db
.venv/bin/python -m discrete.cli --db bench/private/live1006/branch.db process --recover --consolidate
.venv/bin/python bench/run_g46.py --cases bench/private/live1006/cases.yaml --db bench/private/live1006/base.db --name live-base --code ../exp_discrete_main
.venv/bin/python bench/run_g46.py --cases bench/private/live1006/cases.yaml --db bench/private/live1006/branch.db --name live-branch
.venv/bin/python bench/run_g46.py --compare live-base live-branch
```
- Noise: about 2 of 46 questions from the judge alone (closed log). Report per category, especially `why`,
  `profile` and `unanswerable` (the new features must not make it overclaim).
- Inspect `discrete --db X changes`, `profile --ids`, and the `trail` events (`changes --op trail`) for quality.
  Bad patterns or links are prompt/gate bugs: fix, re-run, log.
- Note: `g46/base.db` was opened once by branch code (an export smoke test), so it is at schema 7. The baseline code
  ignores the extra tables/column; scores are unaffected.

**Known gaps / next steps after the evaluation:**
1. The live golden set shows causes that aren't "actions" ("why did I open ChatGPT", "why did I look at Omar's
   profile": a DM). Stored trails cover only completed actions. Query-time `expand_why` covers the rest heuristically
   (same-thing terms, `navigated_from`, the last 2 moments). If why scores stay low, consider trails for every
   engaged navigation event (cost: one muse call each) or an on-demand trail call at query time (latency).
2. Act on `docs/audit_inputs.md`. Change capture/preprocessing only where measured; then rebuild both sessions and
   re-score.
3. Live-test the branch on the user's machine (setup notes below), then dream for real after a few days of data.
4. README section for the new features; update `.env` comments if settings change.
5. Final brief to the user: changed vs not, scores before/after with noise, cost/latency deltas, setup notes.

**Setup notes for the user to test the branch** (draft; finalize after evaluation):
```bash
cd ~/Documents/PARSEC/COMP_001/exp_hindsight/exp_discrete      # branch feat/memory-evolution; .venv already built
ln -sf "$PWD/jarvis" ~/.local/bin/jarvis-exp                     # separate from your normal `jarvis`
jarvis stop                                                      # stop the live DIscrete (same Option+Tab)
jarvis-exp start                                                 # data in ~/.cache/exp_discrete (its own db)
# use the Mac normally; then:
jarvis-exp ask "why did I download helium?"
jarvis-exp dream && jarvis-exp profile && jarvis-exp patterns
jarvis-exp changes 2h ; jarvis-exp history <id> ; jarvis-exp forget <id> ; jarvis-exp revert <event>
jarvis-exp graph ; jarvis-exp export                             # Markdown in ~/.cache/exp_discrete/memory (Obsidian vault)
jarvis-exp stop && jarvis start                                  # back to the normal one
```

---

## Feature plan (agreed; see Phases 1–4 for what was built)

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
