Result: not shipped — 0 reuses on both sessions; removed.

# Caption reuse — audit recommendation 11

Implemented on `feat/memory-evolution`, 2026-10-07. No model/API calls, live runs,
replays, or bench scripts were executed. No commits or pushes were made.

## Changed files in this implementation

| File | Change |
| --- | --- |
| `discrete/capture/caption_reuse.py` | Shared typed state and local eligibility policy using textkit word-shingle Jaccard and 64-bit pHash distance. |
| `discrete/capture/enrich.py` | Reuse before taking a model slot; predecessor task tracking; caption/page-field copying; metadata provenance and counters; missing replay hash calculation. |
| `discrete/capture/screen.py` | One localized job flag distinguishes a current frame from a before-only frame. |
| `tools/replay_session.py` | Register caption tasks through the same Enricher submission path as live capture. |
| `discrete/config.py` | Four localized typed reuse settings. |
| `.env.example` | Defaults, off switches, and eligibility requirements. |
| `tests/test_caption_reuse.py` | 42 fake-backed gate, integration, concurrency, recovery, and offline-measurement tests. |
| `tools/measure_caption_reuse.py` | Read-only observation-level estimator; optionally compute hashes from matching manifest frames locally. |
| `docs/caption_reuse_report.md` | This report. |

Concurrent changes in other files, and other changes within config, screen, or
`.env.example`, belong to other agents.

## Design

Defaults: `DISCRETE_CAPTION_REUSE=true`, `DISCRETE_CAPTION_REUSE_WINDOW_S=180`,
`DISCRETE_CAPTION_REUSE_MAX_TEXT_CHANGE=0.20`, and
`DISCRETE_CAPTION_REUSE_MAX_PHASH_DIST=8`. Set the boolean false or the window
to zero to disable reuse.

Both live capture and replay go through `Enricher._try_reuse`. Reuse requires a
final earlier user caption with exact app, site, URL, window title and session,
less than 20% changed word shingles, and at most eight differing pHash bits.
URL query strings and fragments remain part of page identity. Without a URL,
the nonempty window title identifies the window. Missing/invalid hashes, empty
text, missing current frames, failed captions, and non-user origins are excluded.

Any key, click or clipboard copy in either capture blocks reuse. This is stricter
than trying to infer a substantial action: one click can send a message or choose
a setting. Text/title/URL/caption media and loading signals also block reuse,
including paused media. No structured player-state signal currently exists; the
keyword gate is conservative but cannot detect visually unrecognized media.

Only consecutive captures can reuse. Sightings additionally block intervening
navigation and input that were collapsed into older observation rows. Pending
predecessor captions are awaited outside the model semaphore, with cancellation
shielding, and input counts are read again afterwards. Captures remain asynchronous.

The stored caption, entities, page title and listing status come from the original
captioned observation. Fresh screen text, frames and input counts stay on the new
observation. `navigated_from` is cleared rather than inventing a new arrival.

Existing `meta` entries named `caption_reuse:<observation_id>` record
`copied_from`, `previous`, `caption_started_at`, and policy version. No schema
change is needed. The capture `source` and `enrich_state='done'` retain their
existing meanings. Repeated reuse always compares against the original caption,
preventing age extension and cumulative text/frame drift. Pending provenance is
written before the copied caption; a fresh model attempt clears stale provenance.
`captions_reused`, Enricher's `caption_reused` count, and `enrich.reused` samples
make skips measurable.

## Offline measurements

The two baseline databases contain no stored frame hashes or frame paths.
Using only their stored fields therefore gives **zero skips in both**. Matching
manifest frames were additionally hashed locally, without modifying the databases,
manifests or frames. The result remains **zero skips**:

| Dataset | Observation rows | Baseline user model captions | Locally available usable hashes | Same-page pairs within 180 s after origin filtering | Pairs below 20% text change | Captions skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| g46 | 133 | 130 | 128 | 31 | 0 | **0 (0%)** |
| live1006 | 240 | 222 | 223 | 88 | 0 | **0 (0%)** |

The existing origin classifier excludes seven of live1006's 222 model-captioned
user rows, leaving 215 captioned user rows eligible for consideration. Both
estimates also apply the shared capture redactor; sensitive frames are excluded.

There are **no skipped pairs to inspect** in these datasets. Representative
rejected pairs, using observation IDs from their respective `base.db`:

| Dataset | Earlier → new | Page/window | Gap | Text change | Why not reused |
| --- | --- | --- | ---: | ---: | --- |
| g46 | 117 → 118 | FotMob — Football Live Scores | 5.5 s | 22.01% | Exceeds 20%, even with no keys/clicks/copies. |
| live1006 | 22 → 23 | Agent Memory Repo — Cognition | 27.3 s | 25.00% | Exceeds 20%, even with no keys/clicks/copies. |
| live1006 | 29 → 30 | Agent Memory Repo — Cognition | 17.0 s | 28.31% | Exceeds 20%, even with no keys/clicks/copies. |
| live1006 | 220 → 221 | T3 Code (Nightly) | 46.8 s | 20.33% | Exceeds 20%; new row also has a click and is excluded by origin classification. |

These observations already passed baseline dedup. The measurement demonstrates
no caption-cost or latency saving on these sessions; it does not support the
audit's suggested approximate halving. Fake-backed tests demonstrate that an
eligible pair makes one caption call rather than two. No paid information-loss
evaluation or live/replay latency measurement was performed.

Baseline DBs lack sightings and store aggregate input counts, so this is an
observation-level estimate, not a reconstruction of every capture. The estimator
refuses newer databases with sightings rather than silently omitting their
navigation barriers. It opens checkpointed databases with `mode=ro&immutable=1`
and query-only SQL, refusing nonempty WALs to avoid incomplete snapshots.

Reproduce from the repository root:

```bash
PYTHONPATH=. .venv/bin/python tools/measure_caption_reuse.py --db bench/private/g46/base.db --manifest bench/private/g46/manifest.jsonl
PYTHONPATH=. .venv/bin/python tools/measure_caption_reuse.py --db bench/private/live1006/base.db --manifest bench/private/live1006/manifest.jsonl
```

## Verification

- `.venv/bin/python -m pytest -q`: **364 passed** in the shared workspace.
- Caption reuse tests alone: **42 passed**.
- `.venv/bin/ruff check discrete tests tools/measure_caption_reuse.py tools/replay_session.py`: **passed**.
- `git diff --check`: **passed**.

Tests cover exact navigation/session identity, short/empty text, threshold
boundaries, missing frames/hashes, media, engagement, copying and provenance,
one-slot concurrent legacy replay-shaped jobs, restart/pending recovery, source
expiry, cumulative text drift, collapsed navigation/input, engagement arriving
during a caption wait, stale provenance cleanup, and byte-for-byte read-only
offline measurement with no model calls. The repository code-review guidance was
applied to the scoped changes; no unresolved correctness findings were identified.
