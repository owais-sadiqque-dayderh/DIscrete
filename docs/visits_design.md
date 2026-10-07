# Content and visits (ME-02; design only)

Unchanged content currently has one observation row, one episode, and an interval
extended by later sightings. A revisit can therefore contribute attention without
becoming a new experience. No representation change is implemented in this patch.

Where this happens today:

- `ObservationDraft.dedup_key` (`models.py:47`) covers app, title, URL and text,
  not the occurrence; `add_observation` looks it up with no session or time bound
  (`store/observations.py:74`).
- `_bump` (`store/observations.py:53`) records a `sightings` row (migration
  `m008_sightings`, `store/db.py:385`) but extends the original observation's end
  time and adds the new input counts to it.
- The duplicate path in `app.py:143` returns before segmentation, and segmentation
  only picks up observations with no episode (`store/observations.py:40`), so a
  day-two revisit never enters a day-two episode.
- Claim times, modality and confidence then follow the original row: an engaged
  revisit can relabel an old passive view, and recurrence counts the first day only.

Attention provenance already reads sightings individually (`store/provenance.py`),
which is the occurrence model this design generalises.

## Proposed schema

- `contents(id, content_key, revision, app, site, url, window_title, screen_text,
  caption, page_title, text_source, created_at)`: immutable content revisions.
  Caption/extraction and embedding work are cached by content revision and model.
  A content key is a deduplication candidate, not proof of occurrence identity.
- `visits(id, content_id, session_id, started_at, ended_at, keys, clicks, scrolls,
  clipboard, origin, episode_id, source_sighting_id UNIQUE)`: a bounded foreground
  occurrence with its own modality and episode. A return after the visit gap or on
  a later day creates another visit even when content is identical. Do not extend
  a visit across an absence. A content revision change ends the previous visit.
- Evidence gains `visit_id` and `content_revision`; quotes remain field/offset
  references into the cited immutable content. Claim support membership records
  every contributing visit independently of a bounded displayed quote sample.
  Action confirmations carry occurrence identity; seeing one confirmation twice
  must not imply two completed actions.

## Migration and rollout

Add the tables and nullable evidence references in a numbered, write-locked,
transactional migration. Preserve observation IDs through an explicit legacy
mapping. Backfill content revisions from observations and visits from timestamped
sightings, preserving per-sighting interactions and sessions where known. Never
infer continuous dwell or a session across missing history. Legacy evidence with
no unambiguous sighting maps to a marked legacy occurrence; do not automatically
copy that claim to every historical revisit.

Make the backfill idempotent via unique source IDs. Retain legacy reads during a
staged rollout, then switch segmentation, drafting, provenance, retrieval windows,
retention and privacy lineage to visits. Queue captions/vectors once per content
revision, while drafting receives visit IDs, times and engagement. Erasure must
remove shared content only when no eligible visit depends on it. Validate encrypted
stores and undo/history references before retiring the old mapping.

## Dreaming recurrence

Count distinct eligible episodes and local days from support visits, rather than
content-row starts or quote counts. Repeated frames in one visit do not increase
recurrence. Identical content encountered in separate sessions/days can support a
recurring exposure or project, but does not itself prove a habit, preference, or a
repeated completed action. Retraction removes the visit's support and revalidates
patterns/profile descendants. Replay tests should cover a two-day revisit, an
engaged revisit after passive reading, same-confirmation repeat views, independent
repeat actions, timezone boundaries, and removal of one shared-content visit.
