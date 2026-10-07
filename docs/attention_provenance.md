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
`state_context`. Generic same-site transitions have lower confidence than quoted
links. A clipboard count does not establish its contents: a payload must occur
in the destination's input/content, rather than a browser sidebar.

Drafting and maintenance build missing/changed destinations in bounded batches.
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
It follows high-confidence incoming edges with depth and elapsed-time limits,
revalidating both quotes and visit order. A message, post or typed intent ends the
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
