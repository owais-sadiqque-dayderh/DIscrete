# Attention provenance

The graph answers why questions about visits and attention, including activities
that never produced a completed-action claim. Building it and inspecting it use
no models. Normal answers still require the independent support verifier.

Research informed the design:

- Chromium stores a referring visit (`visits.from_visit`) separately from the URL
  and records link, typed, generated/search, reload and redirect transitions.
  This motivates visit IDs rather than edges between deduplicated page rows.
  [visit database](https://chromium.googlesource.com/chromium/src/+/4d54f0a8e47abbf6d5d401c7d09a43ec2994d130/components/history/core/browser/visit_database.cc),
  [transition definitions](https://chromium.googlesource.com/chromium/src/+/main/ui/base/page_transition_types.h).
- Recall exposes timestamped snapshots through a timeline and search. Its timeline
  is useful evidence organization; chronological proximity alone is insufficient
  to establish why an action occurred.
  [Microsoft's description](https://support.microsoft.com/en-us/windows/ai/ai-features/retrace-your-steps-with-recall).
- Screenpipe combines event-triggered frames, accessibility/OCR text, and separate
  keyboard, mouse and clipboard events. That supports joining text to interaction
  signals while preserving the raw evidence.
  [Screenpipe architecture](https://docs.screenpipe.com/architecture).
- Rewind was also investigated as a timeline analogue. Its original technical
  documentation is no longer available from the original product domain; the
  successor's official announcement says Rewind capture was disabled in December
  2025. No undocumented Rewind causal-graph capability is assumed.
  [successor announcement](https://www.limitless.ai/).

Migration 13 adds `attention_provenance` and `provenance_inputs`. Edges reference
`sightings.id` at both ends, cascade with deleted observations/sightings, and store
kind, confidence, gap, and a literal quote plus field name from each end. Kinds are
`navigated_from`, `navigation`, `search`, `clipboard`, `message_share`,
`media_mention`, `typed_navigation`, `app_switch`, `state_selection`, and
`state_context`, and `same_site_context`. Bare succession on a host supplies only
the nearest contextual predecessor, with confidence 0.30, below the causal walk
threshold. Navigation requires a captured link/page label, referrer URL, or
interaction signal. A model's `navigated_from` assertion alone receives 0.35;
corroboration by source-screen evidence, a referring URL or a payload-matched
interaction is required
to raise it to 0.78. Model caption/referrer evidence is capped at 0.78, even when
corroborated. A clipboard count does not establish its contents: a payload must occur
in the destination's input/content, rather than a browser sidebar.

All persisted model strings, including caption, title, referrer and entities, are
redacted before truncation and storage, including page-field backfills. Inference
scrubs its inputs and rejects any edge whose quote still contains a detectable
secret or is no longer a literal endpoint substring. This guard also applies to
legacy rows and stored edges. Debug rendering and writer context scrub titles and
other metadata as well.

Drafting and maintenance build missing/changed destinations in bounded batches.
Each pass reads one shared snapshot and extracts terms, intents and URL indexes
once. Per-destination candidates and term frequencies use sliding temporal windows;
batch boundaries do not change inference. Writes/checkpoints share a transaction
per batch. An explicit backfill covers all ready destinations in one pass.
Changed source observations invalidate existing edges and make nearby destinations
eligible again, including destinations with no previous edges. Backfill is
restartable and model-free:

```sh
discrete why "why did I open that report?" --chain --backfill
```

The resolver uses lexical effect identity, action citations, profile/search roles,
explicit quoted dialogs and optional local clock anchors. A clock without a date
does not silently become today's date. Visits are filtered by their own sighting
times; a deduplicated observation's widened interval cannot anchor a return visit.
Legacy sightings do not store their own session ID; a return visit's session is
left unknown rather than borrowing the observation's first session.
Matching effects are grouped into sittings: consecutive matches more than
`DISCRETE_WHY_SITTING_GAP_S` apart (default 3 hours) start a new sitting. Select the
most recent sitting before the question time (`resolve_why(..., reference_at=...)`,
defaulting to now), then its first matching arrival. Later returns in the same
sitting have derivative causes and must not displace that arrival. This cutoff
is applied before the candidate/visit pool limits, so later visits cannot crowd
out a historical question's matches. Relevance ties in the candidate pool prefer
recent observations. With no time context, multiple separated plausible sittings
return an explicit ambiguity listing their first arrival times. "Again", "the last
time", "most recently" and "this time" select the latest matching arrival instead.
Equally ranked visits at an indistinguishable time also return an explicit ambiguity
with candidate dates/times and sighting IDs. The writer asks which visit, with no
causal chain asserted. Explicit intervals and clock anchors remain authoritative.
It follows high-confidence incoming edges with depth and elapsed-time limits,
revalidating both quotes and visit order. Fresh empty checkpoints also count as
processed: only missing/stale destinations reached by the chosen backward walk
are inferred, with shared extracted features and no query-time writes.
A message, post or typed intent ends the
walk; an actual share can lead backward from a post to a DM. Multiple copied
references are retained independently. Unsupported chain records, unavailable
verification or a verification timeout cannot become an answerable chain. Without
a connecting edge or an existing verified action trail, why remains unknown.

Dialogs may get two state records: the source chosen for import and a recent
foreground sighting of that application. Recent foreground use is evidence that
the app *may* still have been open, not a process-liveness measurement. The writer
must hedge that inference. Sources come exclusively from stored fields; the graph
cannot recover an unreadable product name or an uncaptured click.

Defaults: 600 seconds per edge, 6 parents, batches of 200; backward walks stop
after 8 edges or 1800 seconds and discard confidence below 0.55. Effect localization
examines at most 300 visits and allows 180 seconds around a clock anchor.
`DISCRETE_PROVENANCE=false` retains the previous action-trail/heuristic path.
All settings are typed in `config.py` and listed in `.env.example`.
