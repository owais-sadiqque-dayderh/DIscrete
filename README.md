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
| `discrete process [--recover] [--consolidate]` | draft waiting episodes now |
| `discrete backfill-pages` | one-off: add page fields to rows captured before C6 |
| `discrete eval [--cases F] [--verbose]` | score golden retrieval cases |
| `discrete gaps` / `discrete answer ID "…"` | knowledge gaps and answering them |

Add `--db PATH` before the command to use another store. Data lives in
`$DISCRETE_DATA_DIR` (default `~/.cache/discrete`): the SQLite store, frames,
the overlay log, and `auto_cases.pending.yaml` (retrieval misses to review).

Every tunable is an env var `DISCRETE_<NAME>` (see `discrete/config.py`). Models are
set per component in `.env`, with request settings (`_EXTRA_TOKENS`,
`_REASONING_EFFORT`, `_TEMPERATURE`, `_JSON_MODE`, `_TIMEOUT_S`, `_EXTRA_BODY`), so any
OpenAI-compatible model, reasoning models included, needs no code change. Check a
new model with `discrete doctor --models`.

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
