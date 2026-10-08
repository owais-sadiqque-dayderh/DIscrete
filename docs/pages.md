# Pages and documents

Migration `m014_pages` adds `pages` and `observations.page_id`. Pages have a unique
canonical URL, site, latest title, deterministic kind, first/last sighting, visits,
estimated foreground dwell (`dwell_s`), active dwell, and an engagement flag.

HTTP identities lowercase/normalize hosts and default ports, remove `www.`,
fragments and known tracking parameters, and sort the remaining query parameters.
Identity-bearing parameters such as search `q`, video `v`, and document IDs remain.
Schemes and path case remain meaningful. URL patterns and the structured listing
flag classify article, repo, video, product, chat, file, listing, search, doc, other.

Capture already puts AXDocument in the observation URL. Local paths become lexical
file URLs; no file is opened or resolved. Without a URL, a nongeneric document/file
window title becomes an app-scoped `doc://` or `file-title://` identity. These title
identities are approximate: homonymous documents in the same app can collide and
renames can split them. Vision-generated `page_title` can label a document but
cannot establish its identity. Collection windows are listings.

Every sighting lasts until the next user sighting, capped by `DWELL_GAP_CAP_S`; the
last sighting uses `DWELL_LAST_S`. Active dwell uses the existing attention rule:
the sighting had keys, clicks, scrolls, or clipboard activity. A visit begins when
the preceding user sighting belongs to a different page, or the gap reaches
`PAGE_VISIT_GAP_S` (default 300 seconds). Consequently a collapsed duplicate still
contributes dwell and can establish a return visit. Intervals clip dwell and count
visits whose start falls in the interval, including the actual next sighting even
when it lies beyond the requested interval.

Triggers invalidate changed observations and neighboring sightings. Store
transactions drain those queues before committing and recompute only affected
pages with indexed neighbor lookups. Capture, delayed enrichment, discards,
deletion, sighting updates, and forget/undo therefore update the materialized
objects atomically. Near-duplicate collapse cannot cross different page identities.

`discrete pages --backfill` derives objects from legacy observations/sightings,
without models, and is idempotent. `--since 7d`, `--kind repo`, and `--top N` filter
the report. Migration alone does not bulk backfill legacy observations. A legacy
store whose original visit times were lost before sightings existed cannot recover
those times; backfill uses the sightings available, never a widened observation
interval as foreground dwell.

The existing aggregate/enumerate planner intents carry guarded page operations for
“what did I read most”, “which repos do I return to”, “which docs did I edit”, and
“how long on X”. They bypass planning, embeddings, reranking, verification, and
writing models and produce deterministic output. Reading is described as estimated
foreground time; document typing is an edit signal, not proof of saving. Other
questions use the normal retrieval/evidence path.

Dreaming and the profile receive repeat visits and dwell as context for interests
and habits. Page totals are not citable claim evidence. Existing episode/day gates,
normal memory evidence, and independent verification remain required.

Denied apps/sites, nonuser observations, and evidence of forgotten claim lineages
receive no page link; unreferenced pages are removed. Backfill does not resurrect
forgotten evidence. Undo restores eligible links. Privacy is checked before copying
metadata, and configured redaction applies to titles and URLs. Changed privacy
settings invalidate existing page links before the next store transaction commits.
The table and invalidation queues use the existing SQLCipher database connection,
so encryption covers them and WAL automatically.

Typed settings: `DISCRETE_PAGES=true`, `DISCRETE_PAGE_VISIT_GAP_S=300`, and
`DISCRETE_PAGE_QUERY_TOP=10`; foreground estimation also uses the existing dwell
settings. Disabling pages disables the model-free routes and page-context input.
