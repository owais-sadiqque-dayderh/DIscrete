# Bench snapshots

One JSON file per optimization step, written by `discrete bench --save NAME`
(see [`docs/C7.md`](../docs/C7.md)). Compare any two with
`discrete bench --compare A B`, or one against the current store with
`discrete bench --compare A`.

Naming: `C7-baseline.json` for step 0, then `C8-<change>.json` and so on, each
taken over a comparable real work session.

`run_g46.py` scores golden questions using the answer checkout selected by
`--code` and an independent judge. It makes model calls, including one tiny
16-token health probe per answer/judge endpoint before scoring. A failed probe
aborts before copying the store or starting questions. The answer endpoint is
read through that checkout's `discrete.config.writer_endpoint()`. The judge
defaults to OpenRouter and uses `--judge-model`; `JUDGE_LM_API_BASE` and
`JUDGE_LM_API_KEY` can override its provider (key fallback: `OPENROUTER_API_KEY`,
then `CORE_LM_API_KEY`).

```sh
.venv/bin/python bench/run_g46.py --db /path/to/store.db --name example \
  --cases /path/to/cases.yaml --jobs 1 --infra-retries 2 --max-infra-errors 0.10
.venv/bin/python bench/run_g46.py --compare baseline example
```

Use `--jobs 1` for comparable latency and while providers are unstable. Larger
values parallelize both answers and judging, skew latency, and can trigger rate
limits. Every answer attempt has a 300-second timeout; timed-out process groups
are killed, the answer is empty, and the case records `error: "timeout"`.
Timeouts, failed answer processes with no output, provider errors (including
partial streamed answers), and failed or malformed judge responses are marked
`infra_error`. They receive up to `--infra-retries` additional attempts with
2/4/8/... second backoff capped at 30 seconds. Judge retries reuse the answer.
Empty answers without model-error diagnostics still count as incorrect.

Results preserve existing fields and add per-case `infra_error`, `attempts`, and
`judge_attempts`, plus summary `n_scored`, `infra_errors`, and `infra_error_rate`.
Infrastructure cases are excluded from accuracy, retrieval rates, and category
denominators; `n` still counts all selected questions. Latency samples include
final failed attempts, and answer cost includes all persisted retry calls.
If the final infrastructure-error fraction exceeds
`--max-infra-errors` (a fraction from 0 to 1, default 0.10), the JSON records
`valid: false` and the runner exits nonzero. A run with no scorable cases is also
invalid. `--compare` displays invalid runs as `INVALID` and individual excluded
cases as `INFRA`; legacy JSON without `valid` is treated as valid.
`run_matrix.sh` stops on failures and invalid results, including cached invalid
JSON, so provider outages cannot silently advance the matrix. Check provider
health before rescoring with `FORCE=1`.

Run the offline regression gate against a completed replay (no model calls or
store migrations):

```sh
.venv/bin/discrete --db bench/private/g46/branch.db check --expect bench/expect_g46.yaml
.venv/bin/discrete --db bench/private/live1006/branch.db check --expect bench/expect_live1006.yaml
# Add latency gates when the store includes capture and segmentation timings:
.venv/bin/discrete --db /path/to/store.db check --slo
```

Exit codes are 0 for a passing gate, 1 for failed checks, and 2 for a bad path or
expectation file. JSON lists each check and its offending ids. YAML accepts
`claims: [{name, text_regex, action?, kind?, min_count?}]`, `screen_text_cap`, and
`chrome_max_share` (default 0.30). Claims must be live; regexes ignore case. The
chrome check measures raw recurring lines per (app, site), including persistent
content, so it is a redundancy proxy rather than a DOM chrome classifier.

Checks report `pass`, `fail`, or `n/a` in JSON. Missing required columns,
delivered-chunk telemetry, or capture truncation flags in legacy stores report
`n/a` with a reason and do not fail the exit code. Recorded omissions and
unflagged cap hits in instrumented captures still fail. New drafting retains
coverage in `meta` after retry checkpoints are cleared. Imported golden captures
at the historical 12,000-character cap are flagged conservatively, as are current
capture cap hits. Latency gates use the same linear p90 as `bench`; stages with
no valid timing samples report `n/a`. Replay stores generally lack `capture.text`
and `segment.lag`, so `--slo` evaluates the stages that were measured.

`stats` exposes cumulative silent-loss counters in its health line. Worker events
persist through the ordinary performance flush. `text_truncated.<limit>` counts
actual utility cuts (4000 is the caption prompt limit); `draft_text_truncated`
counts cuts in requested drafts, excluding prompt hashing. AX reasons, cap hits,
redaction frame drops, draft failures/cancellations/abandonments, trail
failures/cancellations/abandonments, and delivered draft rows/chunks are separate
counters. `caption_text_at_risk` is a current-row risk count, not proof that chrome
filtering left enough text to require a caption cut.
