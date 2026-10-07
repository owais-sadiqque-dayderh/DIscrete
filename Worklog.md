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

## ▶ HANDOFF v2 — READ THIS FIRST (updated 2026-10-07 ~09:00 IST; keep it current)

**Who you are:** the next orchestrating agent. The user's standing rules, all still in force:
- **Orchestrate; let Codex engineer.** Delegate implementation, tests and audits through T3 `delegate_task`
  (`providerInstanceId: codex`, model `gpt-6.1-sol`, reasoningEffort `high`/`xhigh`; use `gpt-6-sol` xhigh for
  independent reviews). You plan, review diffs, run evaluations, commit and log. Claude's limit is small; Codex's
  is generous.
- Work only in `COMP_001/exp_hindsight/exp_discrete`, branch **`feat/memory-evolution`** (`main` = baseline
  `41e2f9f`, also checked out as `../exp_discrete_main`).
  - Never touch `COMP_001/Discrete/` (the user's live copy).
  - Don't merge, push or open PRs.
  - Commit with explicit paths only (`git add <files>`, never `-A`): agents edit concurrently.
- Use DIscrete's exact models for every comparison (`.env`): muse-spark-1.3-contributor (minimal reasoning),
  gemini-3.8-flash verifier, gpt-4o-mini utility, text-embedding-3-small, cohere rerank-4-fast. Judge:
  deepseek-v4-pro.
- Log everything in this file as you go; the user spawns a fresh agent when one hits its limit.
- At the end: brief the user on what changed and what didn't, with before/after numbers, and give setup notes to test
  the branch. The user decides whether to merge.

**State right now:**
- Branch HEAD `669a298`+worklog commits; suite ≈ 524 tests (run `.venv/bin/python -m pytest -q`).
- Committed this session (Phase 7, details below):
  - action dedup and agent credit;
  - #18 chunk checkpoints;
  - `bench/redraft.py`;
  - episodes by task + `tasks` layer (m011);
  - `discrete check` invariants/SLOs/health counters;
  - the shared retry policy;
  - review-r2 fixes 1–9;
  - draft state as columns (m012);
  - bench harness hardening;
  - the AX markers verdict (off);
  - the AX node walk (6× faster);
  - caption reuse measured and NOT shipped.
- **Working tree clean** (09:15). Encryption at rest + "why" for any moment were committed together in `669a298`
  (they share cli/config/db). 528 tests, ruff clean; no golden-set terms in the code (grep-checked).
  - Why dry run on `live1006/branch.db`, read-only, no models: all 6 held-out why questions produce the expected
    chain. Theo's video (OCR) → "helium" search → homepage → download; Cognition's X post → spec page; copied
    GitHub + Cognition URLs → the typed ChatGPT question; Kaustubh's DM → Agently post → Omar's profile; EmbeddingGemma
    post → typed openrouter.ai → Models; Helium import dialog ← Arc recently in use (hedged).
  - `g46/branch.db` finds no effect for these questions, as expected: they belong to the live session.
  - `discrete why "…" --chain [--backfill]`. Design: `docs/attention_provenance.md`.
  - **Before scoring, backfill provenance edges** on every branch db copy:
    `discrete --db X why "x" --backfill` (or check the CLI help). Edges are built after drafting, and the old dbs
    have none.
- **Muse is back** (12:10 IST; the user okayed `openai/gpt-6-luna` as a fallback if it goes down again; apply
  the same model to baseline and branch, and note it in the results).
- **Since the last handoff update:**
  - pages committed `54e6f0f` (m014, 581 tests).
  - Review r3 of `669a298` found **7 findings**:
    - P1: a denied site's cached frame reaches an allowed site's vision job.
    - P1: model `page_title`/`navigated_from` bypass redaction and reach why quotes.
    - P1: a custom-path encrypted store gets plaintext frames.
    - P2: same-host timing becomes a false navigation cause.
    - P2: a model-supplied referrer is treated as observed.
    - P2: quadratic provenance cost.
    - P2: ambiguous why picks the oldest visit.
  - Codex is fixing all 7: taskId
    `node:delegated-task:command%3Amcp%3Ad8b8c8e8-cb3d-4b23-a1db-f6827aaed2cd%3Adelegate-task%3Aexp-discrete-review-r3-fix-1`.
  - **Score the branch only after these fixes:** #4/#5/#7 change why answers.
- **Eval worktree recreated** at `54e6f0f` (`../exp_discrete_eval`; the symlinks as described below). `live-base`
  (main code, live session) scoring started 12:12 from there: log `bench/private/results/live-base.log`.
  - g46-base already exists (0.652).
  - The branch scores need a NEW worktree at the commit with the r3 fixes.
- **Harness false positive fixed** (`48396d9`, 12:40). The outage detector matched DIscrete's normal warning
  "support verification timed out; results left unverified" and re-asked questions that already had answers.
  - Now a non-empty answer is infra only on a hard failure: traceback, exception, HTTP status, or overload.
  - `live-base` restarted from the eval worktree; the fixed `run_g46.py` was copied in.
- **Note for the eval:** with the support check timing out (`SUPPORT_TIMEOUT_S` 8 s, seen often in the logs), records
  go to the writer unverified. That is DIscrete behaviour, identical for base and branch, but worth a later look
  (the gemini-verify ablation measures it).
- **Review r3 fixes done** (Codex, 605 tests, UNCOMMITTED; report `/tmp/discrete-review-implementation-report.md`).
  All 7 have regression tests:
  - denied frames are never cached or forwarded;
  - every model field is redacted before storage and provenance;
  - frame encryption is bound to the owning store, failing closed when unsure;
  - same-host succession is context only (0.30);
  - an uncorroborated model referrer gets 0.35;
  - provenance cost is bounded: 2,000 moments backfill in 0.68 s, a why query in 198 ms;
  - an ambiguous effect asks for clarification.
  - **Regression caught in my review.** "Prefer the latest visit" moved the Agent Memory Repo why chain from its true
    origin (18 → 20, Cognition's X post) to a later same-session return (218 → … → 235).
  - Follow-up running: the **first arrival in the latest sitting**; "again"/"last time" select the latest; far-apart
    sittings with no time context → ask. Task `…exp-discrete-why-effect-select-1` (cmd d8b8c8e8). Commit r3 fixes +
    this together, then make the branch eval worktree.
- **Baseline scores (main code, valid):**
  - `g46-base` **0.652 (30/46)**.
  - `live-base` **0.433 (13/30)**, scored 12:45 with the hardened harness, 0 infra errors. By category: why 4/6,
    unanswerable 4/5, recall 4/9, temporal 1/4, profile 0/3, enumerate 0/2, aggregate 0/1. Supported 0.633, hit@k
    0.52, p50 19.1 s, $0.07. File: `bench/private/results/live-base.json`.
  - **No branch scores yet.**
- **13:00 — r3 fixes + why effect selection committed** `528a57b` (616 tests).
  - By default, all six live why chains resolve to their true origins.
- **Branch evaluation RUNNING** from the frozen worktree `../exp_discrete_eval` @ `528a57b`.
  - Script: `bench/private/eval_branch.sh`. Log: `bench/private/results/branch-driver.log`, which ends `ALL DONE`
    plus two compare tables.
  - Phase 1, 4 parallel jobs: g46-branch, live-branch (old drafts + final code), g46-branch-v2, live-branch-v2
    (redrafted with the final code).
  - Phase 2: g46-branch-geminiverify, g46-base+branchretrieval.
  - Each db is copied to `*-dream.db`, gets the page + provenance backfill (no model), then `dream --force`, then is
    scored.
  - If interrupted: check which `results/<name>.json` exist and re-run only the missing jobs. Read per-name logs and
    the `"valid"` flag.
 Everything from the 07:20–08:05 attempt was invalidated by the outage.
- OpenRouter credit: **$7.32** left ($45 total, $37.68 used). Disk: 12 GiB free.

**Evaluation plan, to run once muse answers:**
1. Commit everything. Then make a **fresh frozen worktree** at that commit:
   `git worktree remove ../exp_discrete_eval --force` (it is stale, at `e57ca8f`), then
   `git worktree add --detach ../exp_discrete_eval <commit>`, `cd ../exp_discrete_eval`,
   `ln -s ../exp_discrete/.venv .venv`, `cp ../exp_discrete/.env .env`,
   `rm -rf bench/private && ln -s ../../exp_discrete/bench/private bench/private`. Score only from there (agents
   edit the main checkout).
2. Re-create the v2 dbs with the final code (only drafting is paid for):
   `rm -f bench/private/{g46,live1006}/branch-v2*.db`, then
   `.venv/bin/python bench/redraft.py bench/private/g46/branch.db bench/private/g46/branch-v2.db` (same for live1006),
   then `.venv/bin/python -m discrete.cli --db … process --recover --consolidate`.
   - Caveat: `branch.db` was replayed at `45fe056`. Episode-boundary changes (#8) only show up in a **full replay**
     (`tools/replay_session.py`, which also pays for captions). Do one full replay of live1006 later if credit allows.
3. Adapt `bench/private/eval_parallel.sh` (it waits on old PIDs; remove those lines and the redraft lines). Run with
   **≤4 concurrent jobs**, against `--code .` (the frozen worktree) for the branch and `../exp_discrete_main` for the
   baseline.
   - Scores: `g46-branch`, `g46-branch-v2`, `g46-branch-geminiverify`, `g46-base+branchretrieval`, `live-base`,
     `live-branch`, `live-branch-v2`.
   - Each branch db is dreamed on a `-dream.db` copy first. Delete the old `*-dream.db` from the failed attempt:
     `g46/branch-dream.db` lost its profile to a 503.
   - The harness now flags invalid runs (`"valid": false`); never report one as a score.
4. Compare with `.venv/bin/python bench/run_g46.py --compare …`. Report per category, especially `why` (6 live
   questions), `profile`, `unanswerable`. Noise is about 2 of 46 questions.
5. Also run `discrete --db <db> check --expect bench/expect_g46.yaml` (and the live1006 one) on each db.

**Open engineering ideas (not started):**
- trails/why quality tuning after the scores;
- Touch ID / key recovery for encryption;
- pages as first-class objects (audit schema #2);
- preference slots (#6);
- table-aware drafting;
- live-test the branch on the user's Mac (setup notes in HANDOFF v1 → still valid, plus `.[encryption]` if wanted).

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

## Phase 6 — Independent review and its fixes; capture text round 2 (2026-10-07, ~02:00–03:30 IST)

- **Code review of the whole branch** (delegated, Claude Opus, report-only; 19 findings, 9 reproduced with
  scripts). All fixed except #18, with regression tests (222 pass). Commits `cfcc7ee`, `99abe03`:
  - **#1/#2 containment.** A list/bare JSON reply crashed trails with an AttributeError, re-sent the same paid
    prompt every minute, and starved all later maintenance; a crashing dream step left the dream "due" forever.
    Fixes:
    - `llm.complete_obj` turns a non-object reply into an ordinary failed call;
    - each maintenance and dream step is isolated;
    - `last_dream` is set in `finally`;
    - trails retry up to `TRAIL_MAX_ATTEMPTS`, then give up visibly.
  - **#3/#6/#7 lossless undo.**
    - Changes record evidence added and evicted by the cap in full, plus untracked columns (`occurred_to`,
      `derived_from`, `flag`).
    - Reverting a revert is a `redo` that restores suppressions, links and the profile, and re-activates the
      original event.
    - An old profile can't overwrite a newer one without `--force`.
  - **#4 forget reaches everything.**
    - Profile points citing a dead memory are hidden at once.
    - `forget` refreshes the Markdown export.
    - Git history for the export is now opt-in (`DISCRETE_MEMORY_EXPORT_GIT`): git would keep forgotten memories
      forever.
  - **#5:** `correct` targets the lineage's live version, keeps `action`, and carries the trail.
  - **#9/#10:** the conflict winner must still be live; patterns are never `stated`.
  - **#11:** patterns need `DREAM_MIN_OCCASIONS` (2) separate sittings (≥60 min apart). 40-capture episodes made one
    long read look like several sessions.
  - **#12/#13:** one `redact_capture` policy shared by live capture and replay. The frame is dropped on a confident
    credential or a catch-all hit in OCR'd text. Replay decodes a frame only when a caption slot is free.
  - **#14:** attention is aggregated in SQL with `LEAD()` over a capped interval; the time-spent regex was
    tightened.
  - **#15/#16/#17/#19:** trail links only in `links_into`; actions sorted before slicing; no distinctive term → no
    link; `rebuild_edges` computed on the reader.
  - **Open, low (#18):** a failed later chunk discards earlier chunks' paid calls on retry.
- **Capture text round 2** (Codex; commit `98ab3b9`):
  - the node walk drops container labels repeated by their children and skips tiny description-only icons;
  - a Chromium text-marker fast path (visible text in one call). The Mac locked mid-measurement, so it is
    **off** (`DISCRETE_AX_TEXT_MARKERS=false`) until verified on real pages.
  - Arc's arc tree: per-run AXStaticText under AXGroup/AXLink/AXHeading; no AXInlineTextBox.
- **First look at `g46/branch.db`** (rebuilt at `45fe056`):
  - **Actions:** 9 action claims (downloaded ChatGPT.dmg, chose Engineering in onboarding, created docs/PDF, sent
    PMO.pdf / the UCL PDF).
  - **Trails:** 8 links, all sensible steps; e.g. the typed `amo run … send PMO.pdf to Yacine` command → the send.
  - **Issues to fix next:**
    1. **Actor:** sends executed by the user's AMO agent are written "the user sent" (the closed log saw the same
       wording question).
    2. **Duplicate action events:** #33/#37 and #38/#48 say the same thing. Events skip the merge, so actions
       need a dedup (same action + object within an episode window).

## Phase 7 — Action dedup, agent credit; the evaluation matrix (2026-10-07, ~07:00 IST →)

- **Action dedup + agent credit** (commit `6465694`, 223 tests): a drafted action matching a live action with the
  same verb on the same thing (token Jaccard ≥ `ACTION_DEDUP_JACCARD` 0.5, within `ACTION_DEDUP_WINDOW_S` 3 h)
  reinforces it instead of adding a duplicate. This fixes Phase 6 issue 2 (two "sent PMO.pdf", two "created the UCL
  PDF"). The drafter writes "the user had <agent> do X" when the user typed a command to an agent and the screen
  shows it done (Phase 6 issue 1).
- The previous session hit its limit before scoring. Picked up by a new Claude session (07:xx IST) on the user's
  instruction, delegating to Codex where possible.
- **All four rebuilds finished** (logs end in `DONE`): `g46/branch.db`, `g46/branch-geminiverify.db`,
  `live1006/branch.db`, `live1006/base.db`. They were built at `45fe056`, **before** the dedup commit, so this
  matrix measures the branch without dedup.
- Resources at start: OpenRouter $7.35 left ($45 total), disk 13 GiB free (47%).
- **`bench/run_matrix.sh` launched** (`nohup`, log `bench/private/results/matrix.log`). It dreams on copies, then
  scores g46-branch, g46-branch-geminiverify, g46-base+branchretrieval, live-base and live-branch (g46-base is
  already scored).
- **Delegated to Codex** (gpt-6.1-sol, uncommitted until reviewed): review finding #18 (keep successful chunk drafts
  when a later chunk fails, so a retry re-requests only the failed chunks) and a README section for the branch's
  features. taskId `node:delegated-task:command%3Amcp%3Aa31f5b55-8276-48a6-85cd-9a27a9d59a69%3Adelegate-task%3Aexp-discrete-r18-readme-1`.
- **#18 fixed, README table** (Codex, reviewed; commit `a73e1d1`, 231 tests):
  - Migration 10 `episode_draft_chunks` checkpoints each successful chunk reply.
  - The key is a sha256 of the episode's observation rows and the exact prompts, so a data, filter or prompt change
    invalidates it. My review added `default=str` so a non-JSON column can't crash drafting.
  - A retry, including after a restart, requests only the missing chunks. A trigger clears the checkpoints when the
    episode closes, in the same transaction.
  - README "Memory that learns…" became a feature table, accurate to the code (export git history is opt-in).
- **`bench/redraft.py`** (new): copies a rebuilt store and clears everything drafting and consolidation produced:
  claims, evidence, mentions, links, change log, digests, gaps, promoted entities/aliases/edges, their embeddings and
  outbox rows, draft checkpoints.
  - It keeps observations, captions, sightings, episode boundaries and entity candidates, and sets episodes back to
    `ready`.
  - Then `process --recover --consolidate` rebuilds the memory with the current code; captions are not paid for
    again.
- **Re-drafts with the final code** (dedup + #18) launched: `g46/branch-v2.db`, `live1006/branch-v2.db`; logs
  `*/redraft-v2.log` end in `DONE`. They are scored as `g46-branch-v2` / `live-branch-v2` after the matrix.
- **Working mode (user, 07:30):** Claude orchestrates; Codex does the engineering (generous limits). Three Codex
  tasks running in parallel on disjoint files:
  - audit #8, episodes by task + `tasks` layer (`process/episodes.py`, `store/episodes.py`, m011): taskId `…exp-discrete-audit8-tasks-1`;
  - audit #12 (a)-(c), invariant checker / `discrete check`, SLOs, silent-loss health counters: `…exp-discrete-audit12-hygiene-1`;
  - audit #11, reuse captions for an unchanged page (shared live + replay policy): `…exp-discrete-audit11-captions-1`.
- **Evaluation pinned to a frozen worktree** `exp_hindsight/exp_discrete_eval` (detached at `e57ca8f`; `.venv`,
  `.env` copy, `bench/private` symlinked). Agents now edit the main checkout, and `run_g46.py --code .` would have
  scored half-edited code.
  - The matrix driver loops were stopped. The in-flight g46 dream and g46 re-draft finished on the code they had
    loaded.
  - `bench/private/eval_driver.sh` (log `results/driver.log`, ends `ALL DONE`) runs the live re-draft, the matrix
    and the v2 scores, all from the frozen code.
  - Score future branch states the same way: a new worktree per evaluated commit.
- **Codex results (reviewed, not yet committed):**
  - **#8 episodes by task** (33 tests):
    - time away from an episode's context now counts across any detour;
    - topic-shift splits work across app/site;
    - adjacent undrafted episodes are merged before drafting (embedding centroids, with adjacency, session, time and
      size guards);
    - migration m011 adds `tasks` + episode mapping and assignment without model calls (embeddings, token cosine,
      entity/time guards), `discrete tasks`, and retrieval adds bounded sibling episodes of a strong match (still
      checked by the support verifier).
    - Episode boundaries change, so it needs a **full replay** to evaluate; `redraft.py` keeps old boundaries.
  - **#12 hygiene:** `discrete check [--expect FILE] [--slo]` (store invariants, expected claims), SLOs in
    `discrete bench`, draft coverage/truncation flags, silent-loss health counters in `stats`.
    - On the current dbs: every store passes "no claim cites a self row" and "every quote is on screen".
    - The typed `downloaded` action (Helium/ChatGPT.dmg) exists **only in branch stores**: the baselines describe the
      download but never record it as an action.
    - Legacy stores lack the new telemetry, so those checks fail. They are being changed to report `n/a`.
    - `enrich.total` p90 of 205–378 s in the branch rebuilds is queueing from parallel replay, not live latency.
  - **#11 caption reuse — not shipped.**
    - Offline it would have reused **0/130** (g46) and **0/222** (live1006) captions: capture's pHash dedup already
      collapses near-identical frames, so every stored row differs enough to need its own caption. The nearest miss
      was FotMob at 22% text change.
    - Being removed (dead weight); report kept at `docs/caption_reuse_report.md`.
- **Integration task** (Codex, `…exp-discrete-integrate-r1`): remove #11 cleanly from shared files; `n/a` for
  missing telemetry; `redraft.py` clears tasks and respects merged-episode tombstones; whole-suite interaction
  fixes. Next: an independent Codex review of the combined diff, then commits.
- **Integrated and committed** `ecc870e` (#8 + #12; #11 removed, report kept): 328 tests, ruff clean.
  - `check` reports `n/a` when telemetry is missing.
  - `redraft.py` clears tasks and keeps merged-episode tombstones.
  - `.env.example` documents all seven #8 settings.
- **Evaluation made parallel** (user asked why commands take so long). The time is OpenRouter calls: ~16 s per
  answer plus a judge call, ~30 s per drafted episode, minutes per dream.
  - The chain ran one job after another. `bench/private/eval_parallel.sh` (frozen worktree) runs all seven jobs at
    once: g46-branch, -geminiverify, base+branchretrieval, live-base, live-branch, and the two v2 jobs that wait for
    their re-drafts.
- **Bug found:** the dream on `g46/branch-dream.db` lost its profile to an HTTP 503 (`service_overloaded`), so
  g46-branch is scored **without a profile** (affects its 3 profile questions).
  - Codex is making every background model call share one retry policy (`…exp-discrete-dream-retry-1`).
- **Independent review** of `a73e1d1` + `ecc870e` by Codex gpt-6-sol, xhigh, report only:
  `…exp-discrete-review-r2`.
- **Retry policy (Codex), my review caught a regression:**
  - The shared retry budget was also applied as a hard per-attempt timeout. With Bubble's 8 s budget, muse's ~9.5 s
    first-chunk p50 would have cancelled about half of Bubble answers. Long background drafts would also have been
    cut at 90 s.
  - Sent back (`…exp-discrete-dream-retry-2`): each attempt keeps the transport timeout; the budget only gates
    starting another retry.
- **Review r2 results:** 9 findings, all reproduced (`/tmp/discrete_review_repro.py`):
  1. **High:** redraft deletes user forget/correct directives, so a forgotten memory could be re-learned.
  2. **Medium:** live drafting can close a just-split episode before the merge sees its successor.
  3. **Medium:** `_split` anchors the new episode to the last moved row's app instead of the first.
  4. **Medium:** delayed embeddings can't reunite an established task group.
  5. **Medium:** the checkpoint hash includes volatile columns (`dup_count`, `updated_at`).
  6. **Medium:** the evidence invariant passes a claim with one real and one fabricated quote.
  7. **Medium:** redraft keeps draft-derived entity candidates.
  8. **Low:** existing hits consume the sibling-expansion limit.
  9. **Low:** task backfill holds one write transaction over thousands of episodes (5.4 s at 2k).
  - **Effect on the running eval:** #1 doesn't apply (the bench dbs have no user directives). #7 may slightly bias
    entity promotion in the `-v2` dbs; noted for interpreting `g46-branch-v2` / `live-branch-v2`.
  - All 9 are being fixed by Codex with regression tests (`…exp-discrete-review-r2-fix-1`).
- **Retry policy committed** `c12ae56` (375 tests): attempts keep their transport timeout; budgets gate only retry
  starts.
- **Evaluation stopped (~08:05): provider outage.**
  - `g46-branch` crashed when one `ask` hit the harness's 300 s subprocess timeout. A manual repro showed retrieval in
    11 s, then 190 s of silence from the writer: three 60 s read timeouts under the old code's SDK retries.
  - A direct probe showed OpenRouter returning `service_overloaded` with `Retry-After: 60` for
    `meta/muse-spark-1.3-contributor` on every request.
  - Answers produced in this window are invalid (empty or timed out), so all eval jobs and the two v2 re-drafts were
    killed. No scores from this attempt are kept.
  - A background probe sends one tiny muse request every 2 min and reports after two successes in a row.
- **Harness gaps found:** `run_g46.py` lets a `TimeoutExpired` kill the whole run (it should record an error and
  continue). An outage also produces empty answers that are judged as wrong, so the harness should flag a run whose
  error rate exceeds a threshold as invalid instead of reporting a score. Both go to Codex.
- **Review r2 fixes committed** `f807356` (392 tests): all 9 findings, each with a regression test that fails on
  the old code. One follow-up from my review: the drafting grace timer and redraft's boundary lock were stored as
  unbounded per-episode `meta` rows. Codex is moving them to `episodes` columns (m012), task
  `…exp-discrete-meta-columns-1`.
- **User (08:20):** muse-spark-1.3 is down on Meta's side at OpenRouter and being fixed. Until it returns, do work
  that needs no muse calls. Delegated to Codex:
  - **"Why" for any moment** (`…exp-discrete-why-any-1`, xhigh): a deterministic attention-provenance graph over
    observations (navigation, search→result, copy→paste/typed, message/share→opened link, brand seen in media→searched,
    app switch with shared terms), plus a query-time why resolver that walks it back to an origin. It also explains
    error dialogs from the state at that moment. Real evidence only; the support verifier still applies. The live
    golden "why" questions stay held out; tests are synthetic, plus a dry run of chain-building on the real dbs.
  - **Live verification of the Chromium AX text-marker fast path** on this Mac (`…exp-discrete-ax-markers-live-1`):
    markers on vs off on 6–8 real pages. It measures length, noise and visible share against OCR, plus latency, then
    decides on the default and writes `docs/ax_text_markers_report.md`.
  - **Queued:** encryption at rest (audit #3: store + frames, key in the Keychain). It waits for m012, since both
    touch `store/db.py`.
- **m012 committed** `8e872a3`: episode draft state (grace timer, boundary lock) lives in `episodes` columns.
  Legacy meta rows are migrated and deleted.
- **Encryption at rest delegated** (`…exp-discrete-encrypt-1`, xhigh): store + frames. Codex is to research
  SQLCipher vs per-file AES-GCM vs column-level, with the key in the Keychain. It is opt-in
  (`DISCRETE_ENCRYPT_AT_REST`, default off on the branch) with crash-safe `discrete encrypt` / `decrypt`, so the bench
  dbs stay plaintext.
- **Bench harness hardened** `fa01372`:
  - timeouts are recorded, not fatal;
  - provider failures are retried and excluded from accuracy;
  - more than 10% of them marks the run invalid;
  - an endpoint preflight runs first, and `run_matrix.sh` stops on an invalid run.
- **AX text markers verified live** on Arc (`docs/ax_text_markers_report.md`, commit `c32d8cc`): **0/40 attempts
  worked** (`kAXErrorIllegalArgument`, then `kAXErrorNoValue` with a corrected argument type), 209 ms either way.
  - Stays **off**.
  - Follow-ups for the node walk: Hacker News returns 4.7k AX chars vs 2.4k visible (19% duplicate lines); tables
    lose their structure.
- **Encryption at rest done** (Codex; `docs/encryption.md`; to be committed with the why work, since they share
  `cli.py` / `config.py` / `db.py`):
  - **SQLCipher** (the `sqlcipher3` wheel works on py3.13 arm64) + **AES-256-GCM per frame**.
    - Rejected: SEE needs a licensed build; column-level encryption breaks FTS or leaks index terms.
  - Keys: a random 256-bit Keychain master key with separate db/frame subkeys, and `rotate-key`.
  - Migration: `discrete encrypt` / `decrypt` / `--recover`, staged and verified, with atomic swap; tested with real
    process crashes in each direction.
  - Behaviour: FTS and vector search still work; a missing or wrong key gives a clear error; exports need
    `--allow-plaintext`.
  - Opt-in `.[encryption]` + `DISCRETE_ENCRYPT_AT_REST`. Plaintext stores, including the bench dbs, are unchanged.
  - Overhead (3,000 synthetic rows): claim insert 267→630 µs, FTS search 155→199 µs, frame read 40→187 µs; vectors
    within noise.
  - Follow-ups: validation against the real Keychain / a locked session; Touch ID via a signed helper; a key
    recovery story.
- **AX node walk improved** (commit after `f00ef97`, `docs/ax_nodewalk_report.md`), measured live on 8 Arc pages:
  - median read **209 → 34 ms**; deadline hits **28/56 → 0/56**;
  - visible-content coverage held or improved everywhere (Python tutorial **2.7% → 96.4%**: it used to time out);
  - Hacker News duplicate lines 18.1% → 0% at the same 98.4% coverage;
  - tables emitted as ` | ` rows.
  - Affects live capture only: replays reuse stored text, so the bench scores won't show it.
- **Re-run plan when muse is healthy:** a new frozen worktree at the final commit (retry policy + review fixes).
  Re-create `branch-v2` with the fixed `redraft.py`, then score everything in parallel, with fewer concurrent jobs
  (≤4) to avoid adding load.

---

## HANDOFF v1 (superseded by HANDOFF v2 at the top; kept for history, written 2026-10-07 ~00:30 IST)

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

## Phase 7 follow-up — First-class pages and documents (2026-10-07)

- Implementation sub-agent; uncommitted, pending the concurrent independent review.
  No paid API calls, replays, benchmark runs, `.env` changes, or writes in `bench/private/`.
- `m014_pages`: `pages` + `observations.page_id`, with unique canonical URL, deterministic
  kind, latest sanitized title, first/last sighting, visits, foreground/active dwell and engagement.
  Known tracking parameters/fragments are stripped, hosts/default ports normalized; local
  AXDocument paths use file URLs. URL-less documents use captured window titles scoped to
  their app. Vision page titles label objects but cannot invent document identities.
- Transactional invalidation queues refresh affected pages on capture, duplicate sightings,
  enrichment, origin changes, deletes and sighting updates. Visits count page switches/returns
  and gaps >= `DISCRETE_PAGE_VISIT_GAP_S` (300 s), rather than distinct observations.
  Dwell uses existing attention caps/active-input rules and clips query intervals correctly.
  Near-duplicate collapse cannot credit a different page identity to an older observation.
- `discrete pages [--backfill] [--since 7d] [--kind repo] [--top N]` backfills without models.
  Guarded aggregate/enumerate routes answer most-read pages, revisited repo pages, documents
  with typing, and duration on a named page/site before any planner/embedding/model call.
  Output distinguishes estimated foreground time and typing signals from confirmed reading
  or saved edits. Dream/profile page signals remain uncitable context; existing claim evidence,
  episode/day gates and independent verification still apply.
- Denied apps/sites and nonuser observations never receive page links. Titles/URLs are redacted;
  changed privacy settings recheck existing links. Forgotten-lineage evidence is detached and
  stays detached through backfill; undo restores eligible links. The existing SQLCipher
  connection covers pages, queues and WAL automatically; encryption tests now exercise pages.
- Typed settings in config + `.env.example` + README: `DISCRETE_PAGES=true`,
  `DISCRETE_PAGE_VISIT_GAP_S=300`, `DISCRETE_PAGE_QUERY_TOP=10`. Design: `docs/pages.md`.
- Copied the live store and any sidecars using filesystem reads only, checking that source
  sizes/mtimes stayed stable during copying. Ran only deterministic backfill on
  `/tmp/discrete-pages-live1006-final-maumge6l/branch.db`: **48 pages**, **155 linked observations**,
  **220 sightings**. A second backfill was identical. Sanitized report:
  `/tmp/discrete-pages-live1006-final-maumge6l/pages-report.json`.
  Top foreground pages: Cognition Agent Memory Repo 266.1 s / 4 visits; ChatGPT Explain Agent
  Memory Repositories 164.6 s / 1; Notes All iCloud listing 107.1 s / 1; Theo's X post 105.2 s / 3;
  X Home Timeline 102.6 s / 7; Omar Ghandour 86.9 s / 2; GitHub AgentMemoryRepo 83.9 s / 2;
  OpenRouter Embedding Leaderboard 57.0 s / 2; OpenRouter Activity 47.7 s / 2; GitHub home 40.7 s / 1.
- Final validation: `.venv/bin/python -m pytest -q` **581 passed** (528 prior + 53 new page tests);
  `.venv/bin/ruff check discrete tests bench tools` clean; `git diff --check` clean.
  Title-only identities remain approximate, and backfill cannot recover visit times already
  lost by legacy capture collapse before sightings existed.

### 13:45 IST — live-branch scored; why REGRESSION found (root-caused)
- live-branch (branch code, old drafts, valid): **0.367** vs live-base 0.433. why **1/6** (base 4/6), unanswerable 5/5 (base 4/5), recall 4/9, temporal 1/4, profile 0/3. p50 15.1 s.
- Cause: `retrieval/support.py` verifier timeout is SUPPORT_TIMEOUT_S=8 s; muse takes ~9-11 s on why queries. The r3 fix drops all attention-provenance hits + why_chain on timeout -> writer refuses ("no reason on file"). On timeout, main left results unverified, so it still answered.
- Proof: `DISCRETE_SUPPORT_TIMEOUT_S=40 ... ask "Why did I start browsing embedding models on OpenRouter?"` on a live-branch-dream copy -> mode=why, correct EmbeddingGemma 2 -> openrouter.ai answer.
- Fix delegated to Codex (clientRequestId why-support-timeout-r1): SUPPORT_TIMEOUT_WHY_S=25, hedged unverified chain on timeout, keep the verified prefix on partial rejection.
- Implication: all branch why numbers in the current eval run are understated; re-score after the fix (move the frozen worktree to the fix commit).
- Why it's slow: each job answers 30-46 questions one at a time (~15-20 s answer + judge each). Redraft re-drafts whole sessions with muse (~12 min). 4 jobs share muse, so some answers come back empty -> infra retries.

### ~13:55 IST — why fix committed (e7a1150); eval round 2 queued
- e7a1150: SUPPORT_TIMEOUT_WHY_S=25; why timeout keeps chain as hedged "inferred, unverified" sequence (grounding -> mode why); partial rejection keeps the verified path adjacent to the effect; verifier errors still fail closed. 627 tests.
- Round 1 (528a57b) valid results: g46-branch **0.756** vs g46-base 0.652 (recall 28/30 vs 22/30; aggregate 0/2 vs 2/2). live-branch 0.367, live-branch-v2 0.367 (why 1/6, 0/6: timeout bug, discard these why numbers).
- /tmp/handover.sh waits for g46-branch-v2 to finish, kills round-1 phase 2, checks out e7a1150 in ../exp_discrete_eval (detached), and runs `bench/private/eval_branch2.sh` -> results/branch2-driver.log
  (live-branch-r2, live-branch-v2-r2, g46-branch-geminiverify, g46-base+branchretrieval; then both compares; ends "ALL DONE").
- If interrupted: check which results/*.json exist; re-run only the missing `score` lines from eval_branch2.sh.
- Harness bug found: the judge gets `records[:30]` *lines*; raw screen-text records (many short lines) push later records out, so correct answers are marked unsupported (g46-branch aggregate 0/2 is this artifact: e.g. the cricket score came from a record the judge never saw). Codex task harness-judge-blocks-r1: block-based judge input with a char budget + `--rejudge NAME` (judge-only, writes NAME-rj.json). Plan: rejudge ALL runs (base + branch) for a fair comparison.
- Also seen: support verification times out on many non-why queries too (8 s, muse as verifier per .env), in base and branch alike.

### ~14:25 IST — rejudged results (harness 1821835: block-based judge input) — USE THESE, the old 30-line judge was unfair to every run
| run (code @ commit) | overall | why | recall | profile | temporal | unans. | enum | aggr |
|---|---|---|---|---|---|---|---|---|
| g46-base-rj (main) | **0.913** | – | 28/30 | 3/3 | 4/4 | 4/5 | 1/2 | 2/2 |
| g46-branch-rj (528a57b, old drafts) | 0.889 (1 infra) | – | 28/30 | 2/3 | 4/4 | 3/4 | 1/2 | 2/2 |
| g46-branch-v2-rj (528a57b, redrafted) | 0.911 (1 infra) | – | 27/29 | 2/3 | 4/4 | 5/5 | 1/2 | 2/2 |
| live-base-rj (main) | **0.533** | 3/6 | 5/9 | 1/3 | 2/4 | 4/5 | 1/2 | 0/1 |
| live-branch-rj (528a57b) | 0.586 (1 infra) | 1/6 | 6/9 | 2/3 | 2/4 | 5/5 | 1/1 | 0/1 |
| live-branch-v2-rj (528a57b) | 0.600 | 2/6 | 6/9 | 2/3 | 2/4 | 5/5 | 1/2 | 0/1 |
| **live-branch-r2-rj (e7a1150 why fix)** | **0.690** (1 infra) | **5/6** | 6/9 | 2/3 | 2/4 | 4/4 | 1/2 | 0/1 |
- g46: branch ≈ main (the old "0.652 -> 0.756" gain was a judge artifact). live1006: branch +0.157, driven by why (3/6 -> 5/6) and profile/recall.
- why answers (r2) trace true origins: Kaustubh DM -> Omar/Agently, Cognition X post -> Agent Memory Repo, EmbeddingGemma 2 post -> OpenRouter. Miss: Helium starts at the Google search; the model-free chain finds Theo's post (obs 76, media_mention), but the support verifier rejects that record because it is not told which term linked it. Codex task why-edge-evidence-r1: persist the edge term (m015), self-explanatory chain records, verifier guidance.
- Round 2 still running (branch2-driver.log): live-branch-v2-r2, g46-branch-geminiverify, g46-base+branchretrieval. Rejudge each with `bench/run_g46.py --rejudge NAME`.

### ~14:45 IST — round 2 rejudged; Helium fix (0b7cac2); round 3 running
- live-branch-v2-r2-rj (e7a1150, redrafted) 0.667, why 4/6. g46-branch-geminiverify-rj 0.913. g46-base+branchretrieval-rj 0.933 (1 infra): branch retrieval on main's drafts is the best g46 number.
- Latency p50 / cost per run: g46-base 15.9 s $0.10; g46-branch 16.0 s $0.12; live-base 19.1 s $0.068; live-branch-r2 15.3 s $0.075. Supported: live-base 0.80 -> live-branch-r2 0.93.
- 0b7cac2: m015 `attention_provenance.term`; chain records explain the link (term, kind in plain words, caption, matched line); verifier prompt guidance for provenance. 651 tests. Live check on a live1006 copy: all 6 why answers trace true origins, incl. **Helium -> Theo's X post** (media_mention).
- Round 3 (`../exp_discrete_eval` @ 0b7cac2, `bench/private/eval_r3.sh` -> results/r3-driver.log): copies *-dream.db -> *-r3.db, model-free provenance re-backfill, scores live-branch-r3, live-branch-v2-r3, g46-branch-v2-r3, then rejudges (-rj). Ends "ALL DONE".
- Remaining live1006 misses (live-branch-r2-rj): empty writer stream (EmbeddingGemma size, muse returned no content, rc=0), event-anchored temporal ("right after I stopped DIscrete with jarvis stop" -> anchor never found), detail recall inside posts (Theo's build time, reranker price), enumerate from a listing page, "normally use" profile.
- Codex tasks running in parallel (disjoint files): empty-writer-r1 (bubble/LLM client: empty-stream retry + extractive fallback; harness flags empty answer as infra) and temporal-anchor-r1 (retrieval/planner: resolve the event anchor, window of distinct moments after/before).
- 7802a92: empty-writer recovery (Codex empty-writer-r1 + my tweak: a finished choice with text and no [DONE] is accepted). Root cause seen raw: muse returns content=null after ~300 reasoning tokens (encrypted reasoning_details) when the records lack the asked detail (EmbeddingGemma size range: the OCR'd benchmark table is garbled). The fallback quotes the top record; the real gap is the detail missing from capture/excerpt.
- temporal-anchor-r1 still running (it edits discrete/retrieval/*; don't commit those files until it reports).
- bca21cf: event-anchored temporal (Codex temporal-anchor-r1 + my guard). Model-free anchors on live1006: "stopped DIscrete with jarvis stop" -> Terminal #203 22:37:17, next Finder .cache -> frames -> DB Browser (matches golden note); "finished Helium setup" -> "Ready to browse?" 22:24:33, next GitHub (matches). My guard: no anchor for clock/date phrases ("after midnight on October 1" stays an interval) or when the subject isn't the user ("what did Yacine reply after I sent the PDF" stays recall/refusal). 691 tests.
- Next: when round 3 finishes, run round 4 @ bca21cf (all fixes) on live-branch, live-branch-v2, g46-branch-v2 (+ g46-base+branchretrieval) and rejudge; then the final brief.

### ~15:15 IST — round 3 results (0b7cac2) + final plan
- live-branch-r3-rj 0.667 (why **6/6**, unanswerable 3/5); live-branch-v2-r3-rj **0.724** (why 5/6, profile 3/3, unanswerable 4/4); g46-branch-v2-r3-rj 0.870 (vs 0.911 r1; per-question diff = writer variance: 2 losses, 1 gain).
- Writer over-claims completion ("You booked with Etihad" from a "Book with Etihad" button; "did I switch to A or B" not answered) -> Codex completion-evidence-r1 (prompt rule + premise-question flag + verifier rule).
- Variance: single runs of 30-46 questions swing ±1-2 questions. The final round repeats runs: `../exp_discrete_eval/bench/private/eval_final.sh` (live main x2, live branch-v2 x2, live branch old-drafts, g46 main, g46 branch; same harness; main via --code ../exp_discrete_main). Check out the final commit in exp_discrete_eval first.
- OpenRouter credit at 15:12: **$4.99** remaining (core key). Final round est. $1.5-2.
