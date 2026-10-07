# Bench snapshots

One JSON file per optimization step, written by `discrete bench --save NAME`
(see [`docs/C7.md`](../docs/C7.md)). Compare any two with
`discrete bench --compare A B`, or one against the current store with
`discrete bench --compare A`.

Naming: `C7-baseline.json` for step 0, then `C8-<change>.json` and so on, each
taken over a comparable real work session.

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
