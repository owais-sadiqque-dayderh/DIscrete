# DIscrete

A user model built from your own screen activity (macOS). It watches the focused
window, stores what you saw and did, distills it into episodes and claims, and
answers questions about it — in the terminal, or inline in any text field with
**Bubble** (Option+Tab).

All model calls go to an OpenAI-compatible API (OpenRouter by default); nothing
runs a local model. Design notes live in [`docs/`](docs/) (plans S1–S3, audit,
insights, and the change log for each revision, e.g. [`docs/C6.md`](docs/C6.md)).

## Quickstart

Requires macOS and Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # then put your API key(s) in .env
```

Grant permissions to the app you run it from (Terminal, iTerm, VS Code…) in
**System Settings → Privacy & Security**:

| Permission        | Needed for                                  |
|-------------------|---------------------------------------------|
| Screen Recording  | capturing the focused window                |
| Accessibility     | reading on-screen text, Bubble's hotkey and typing |
| Input Monitoring  | counting clicks/keys (interaction signal)   |

Restart the terminal after granting, then check everything:

```bash
.venv/bin/discrete doctor     # endpoints, permissions, a real test capture → "OK"
.venv/bin/discrete doctor --models   # one real request per model (after changing models in .env)
.venv/bin/discrete run        # capture + processing + Bubble on Option+Tab (Ctrl-C to stop)
```

## Everyday commands

| Command | What it does |
|---------|--------------|
| `discrete run [--no-bubble]` | the daemon: capture, captions, episodes, drafting, consolidation |
| `discrete ask "…" [--records] [--overlay]` | Bubble's pipeline headlessly; answer printed |
| `discrete query "…"` | show exactly what retrieval returns (debugging) |
| `discrete stats` | store summary and **model use per component** (calls, tokens, cost) |
| `discrete bench [--since last-run] [--eval] [--save N] [--compare N]` | cost, latency per stage, capture volume, quality, storage; saved snapshots and comparisons ([`docs/C7.md`](docs/C7.md)) |
| `discrete process [--recover] [--consolidate]` | draft waiting episodes now; `--recover` also retries failed work |
| `discrete backfill-pages` | one-off: add page fields to rows captured before C6 |
| `discrete eval [--cases F] [--verbose]` | score golden retrieval cases; add `--answers` to check final answers |
| `discrete gaps` / `discrete answer ID "…"` | knowledge gaps and answering them |

Add `--db PATH` before the command to use another store. Data lives in
`$DISCRETE_DATA_DIR` (default `~/.cache/discrete`): the SQLite store, frames,
the overlay log, and `auto_cases.pending.yaml` (retrieval misses to review).

Every tunable is an env var `DISCRETE_<NAME>` (see `discrete/config.py`). Models are
set per component in `.env`, with request settings (`_EXTRA_TOKENS`,
`_REASONING_EFFORT`, `_TEMPERATURE`, `_JSON_MODE`, `_TIMEOUT_S`, `_EXTRA_BODY`), so any
OpenAI-compatible model, reasoning models included, needs no code change. Check a
new model with `discrete doctor --models`.

## Memory that learns, explains and can be undone

The memory-evolution branch adds these features:

| Feature / command | What it does |
|-------------------|--------------|
| `discrete changes` / `discrete history ID` | show the change log and a memory's versions, with actor and reason |
| `discrete revert EVENT [--force]` | undo a change; redo by reverting the resulting undo event. Later changes to the same memories block undo unless forced |
| `discrete forget ID` / `discrete correct ID "…"` | retire a memory's lineage and suppress matching drafts, or replace its live wording with yours as a stated memory |
| Actions and choices | detectors flag completed downloads, installs, purchases, sends and other actions, plus explicitly selected options; the drafter still has to cite evidence and pass verification |
| Causal trails / `discrete ask "why did I …?"` | link an action to matching screen moments from the preceding 30 minutes, checked in code and by the verifier; answers give a hedged reason in time order, with nearby moments as a fallback |
| `discrete dream --force` | find patterns across separate sittings, audit old patterns, resolve conflicts, mark lasting memories stale and refresh a cited profile; events never go stale |
| `discrete profile [--ids]` | inspect the summary of learned projects, interests, habits, tools, relationships and preferences, with supporting memory IDs |
| Belief strength | lasting memories carry episode/day counts and a tentative, recurring or established tier; Bubble words tentative beliefs as recent |
| `discrete graph [NAME]` | show entities and their co-occurrence links; retrieval also searches memories mentioning named entities (`DISCRETE_GRAPH_EXPAND` adds neighbours) |
| `discrete export [--dir PATH]` | write Markdown with `[[links]]` for an Obsidian vault (default `<data dir>/memory`); refreshed after dreams and when forgetting from an existing export |
| `discrete attention [--since 7d] [--by site\|app\|page]` | estimate total and active time from window sightings, including duplicate captures; long gaps are capped, and dreaming reads these totals |
| Redaction | `DISCRETE_REDACT_SECRETS` (default true) masks recognised credentials in screen text, titles, URLs, captions and Bubble's field; confident credential matches and suspicious OCR key material also drop the screenshot |

Automatic dreaming needs both `DISCRETE_DREAM_INTERVAL_MIN` (default 1440 minutes)
and `DISCRETE_DREAM_MIN_NEW_EPISODES` (5 newly closed episodes); interval `0` makes
it manual only. Export git history is opt-in via `DISCRETE_MEMORY_EXPORT_GIT=true`
or `discrete export --git`: its history retains earlier versions, including
forgotten memories. See `.env.example` and `discrete/config.py` for settings.

## Layout

```
discrete/
  models.py        data shapes (no dependencies)
  store/           all SQL: db.py (schema + numbered migrations), one file per concern
  llm.py           API access + per-component usage meter and per-call records
  perf.py          stage timings and event counters (read by `discrete bench`)
  capture/         observer, enrichment (vision captions off the capture path), AX/OCR
  process/         episodes, drafting + merge, entities, consolidation, embeddings
  retrieval/       hybrid.py (search/rerank/MMR), enumerate.py, pages.py, query planning
  evaluation/      golden-case harness, bench.py (the measurement report)
  bubble/          hotkey, overlay pill, writer
  app.py cli.py
tools/replay_workaa.py   one-off: rebuild a store from the WorkAA prototype
docs/                    plans, audit, insights, change logs
bench/                   saved `discrete bench` snapshots, one per optimization step
golden/cases.yaml        golden questions for quality scoring
```

Dependencies point one way: `models, perf → store → llm → capture/process/retrieval → bubble/app/cli`.

## Development

```bash
.venv/bin/python -m pytest -q                    # all tests (two need a live screen)
.venv/bin/python -m pytest -q -m "not display"   # what CI runs
.venv/bin/ruff check discrete tools tests
.venv/bin/pyright
```

Tests use a fake model backend and a temp data dir; they never touch the
network or your real store. CI (`.github/workflows/ci.yml`) runs lint, type
check and tests on a macOS runner.

## Reliability and repeatable validation

Captures are stored before any model call. Vision captions run in a bounded pool
(`DISCRETE_ENRICH_MAX_PENDING`); the backlog waits on disk (row + saved frame) and
resumes after a restart, and segmentation follows capture order. An episode's
claims, evidence, merges, entity candidates and closure commit in one transaction.
An independent verifier (`VERIFY_LM_*`, default `INSIGHT_LM_*`) checks actor,
negation, scope and completion before memories are written, and checks search
results against the question before Bubble answers (`DISCRETE_SUPPORT_CHECK`).
That second check runs on every query, so give `VERIFY_LM` a fast model. Search
fails closed when verification is unavailable. Failed drafting/indexing is retried
after `DISCRETE_FAILED_WORK_RETRY_S` (900 s), then with doubling waits;
`discrete process --recover` retries everything immediately.

`golden/cases.yaml` is for questions about **your own** recorded activity
([`docs/C7.md`](docs/C7.md)). `golden/fixture.yaml` holds expectations about a
sanitized replay, scored against its own isolated store:

```bash
.venv/bin/python tools/replay_session.py --manifest golden/cua_manifest.jsonl --db /tmp/discrete-fixture.db
.venv/bin/discrete --db /tmp/discrete-fixture.db eval --cases golden/fixture.yaml --answers --verbose
```

Use a new database path for each replay. The manifest replays text through
processing and retrieval; live capture and vision need real screenshots
(`tools/capture_task.py` records a window, `tools/replay_session.py` processes a
manifest, `tools/evaluate_session.py` and `tools/compare_models.py` compare
revisions and models). `golden/cua_session.yaml` and `golden/cua_holdout.yaml`
need those live captures. `eval --answers` also generates each answer with
Bubble's writer and has it judged; contamination, missing list items and
unsupported answers fail, and a failing suite exits nonzero.
