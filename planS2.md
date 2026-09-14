# Stage 2 — Data Processing: Design Plan

**Status: design record, revised in place with a visible revision log.**
This file is self-contained: every mechanism, threshold and rationale needed to
implement Stage 2 is stated here. Design changes are made directly in this
file and recorded in the Revision log (what changed, why, and the evidence).
Implementation bugs — cases where the design was right but the code was
wrong — belong in `insights.md`. Sections changed by a revision carry a
`[Rev N]` marker. Stage 1 is specified in `planS1.md`; Stage 3 in `planS3.md`.

---

## Revision log

### Rev 1 — 2026-09-14 (architecture audit against the real WorkAA store)

The evidence is one real 70-minute session (89 observations → 64 propositions,
123 entities, 2190 evidence links, 1808 entity mentions), plus a survey of
current memory systems: mem0, Zep/Graphiti, Hindsight, A-MEM, Letta, GUM,
lattice, and tada's memory wiki. Full numbers are in `audit.md`.

**The central finding.** The prior design had a runaway feedback loop. A
dedup key containing a generated caption let near-identical screens through.
Mixed batches of those observations produced near-identical drafts. Each draft
matched the claim drafted moments earlier as "similar". The revise rule then
merged N claims into **one** claim that had to "drop nothing factual", with
no length bound. Result: one lineage reached **version 45 in 35 minutes**,
mean claim text grew **84 → 1192 characters**, and 89 observations collapsed
into **6 live propositions**. One of those was a 1001-character run-on linked
to 80 of 89 observations. Rev 1 removes every link of that chain.

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 2.1 | §2 | **The dedup key excludes the VLM caption.** It is built from stable signal only: app, window title, URL and verbatim `screen_text`. A frame perceptual-hash tier runs first, in Stage 1. | 84/89 observations had `dup_count = 1`. Fifteen rows whose entire text was the word "discrete" were stored separately, because run-to-run caption rewording dominated the shingle set. |
| 2.2 | new §3 | **Episode segmentation replaces fixed-size batches.** Claims are drafted once per *episode*, a contiguous stretch of one activity, instead of per 5–15-observation batch. | Batches mixed unrelated activities. The 5-minute-gap "session" swallowed 70 minutes of work as one unit. Event-segmentation work (EM-LLM, ICLR 2025) and lattice's session-grouped synthesis both segment into coherent events before abstracting. |
| 2.3 | new §4 | **Claims are typed** as `event` (dated, immutable), `fact` (durable) or `preference` (a belief that strengthens or weakens with evidence). Each claim is ≤ 240 characters and states one proposition. | One untyped table held "editing Artboard 3 right now" and "prefers monochrome palettes" under one merge rule, so the ephemeral claims absorbed the durable ones. Hindsight (91.4 % LongMemEval) separates objective facts, experiences and opinions. Zep separates episodes from semantic facts. |
| 2.4 | §6 | **The online write path never rewrites claim text.** Its operations are ADD, REINFORCE (link evidence only) and INVALIDATE (bi-temporal close on contradiction). The revise cooldown is retired. | Text rewriting was the accretion engine. mem0 moved from ADD/UPDATE/DELETE to **ADD-only** extraction and reports +21 points on LoCoMo and +27 on LongMemEval (vendor-reported). A controlled ablation found verbatim memory beating LLM-rewritten artifacts by 16–22 points (arXiv 2601.00821). The cooldown also never executed in the prototype (see `insights.md` §7.1). |
| 2.5 | new §7 | **Consolidation runs offline and periodically** ("sleep-time"). It merges and *splits* claim clusters (N inputs → a set of outputs, each bounded), induces durable facts and preferences from accumulated events, refreshes entity pages, and lints the store. | GUM's original revise step returns a *set* and permits split, with "typically 1–2 sentences". The prototype dropped both. Letta's sleep-time agents, tada's memory ingest/lint cycle and lattice's layered induction all do abstraction offline, over a window, as new nodes rather than rewritten ones. |
| 2.6 | §6.1 | **Relation labels add `contradicts` and `refines`.** Contradiction closes the old claim's validity interval and never deletes it. | The prior identical/similar/unrelated scheme had no way to express "this used to be true", which is the one thing bi-temporality exists for. This follows Zep's and Graphiti's edge invalidation. |
| 2.7 | §8 | **Confidence is derived from evidence, not asked of the model.** The LLM-assigned `decay` field is removed; each claim kind carries a fixed half-life instead. This resolves the prior open item. | 60/64 claims had confidence exactly 8.0 and 56/64 had decay exactly 4.0: the values in the prompt's JSON example. Verbalized LLM confidence is documented to saturate at a few high values. |
| 2.8 | §5, §9 | **Evidence is bounded.** A claim keeps at most 12 verified supports. Weak co-occurrence links are dropped once 3 verified supports exist. Quotes come from the drafting call and are verified deterministically; the separate evidence-mapping LLM call is removed. | The live blob claim held 58 verified supports and 22 co-occurrence links over an 89-observation corpus. A verified quote proved the text existed; it did not prove relevance. |
| 2.9 | §10 | **Entities: salience gate, recurrence gate, and a real resolution cascade.** The cascade is exact → normalized → string-fuzzy → embedding plus keyword candidates → LLM adjudication. `kind` is an attribute, never identity. Mentions require the name to actually appear. This **reverses** the prior "embedding instead of fuzzy matching" decision. | 106/123 entities never got a summary. 90 % were noise: gallery logo names, `art`, `after`, `chart`, `Artboard 2`. The real duplicates were string variants embeddings handle badly: `TOOLS.design`/`TOOL.design`, `Grainient`/`Graident`, `logofoilo.com`. Distinct brands (`Grainient`, `Grabient`, `Gradientos`) risk false merges under embedding similarity. Batch-level mention attribution linked the user entity to 88 % of all rows. Zep resolves with embedding plus full-text candidates and an LLM decision; Hindsight combines string, co-occurrence and temporal similarity. |
| 2.10 | new §11 | **Index coverage is an invariant.** Every live row has its full-text and embedding entries, written in the same transaction or through a durable outbox. Embeddings are computed by a local model. | Only 27/64 propositions and 0 entities were embedded. The largest live claim had no vector, so two answerable questions were logged as "knowledge gaps". |
| 2.11 | §4, §6 | **Claims carry `modality`** (`engaged` / `passive` / `stated`), computed from Stage 1 interaction counts. Passive evidence can never invalidate engaged or stated claims. | This separates "did" from "viewed" (GUM's own warning). It is also a layered defence against memory poisoning from on-screen text: a page the user merely viewed cannot overwrite what they did. |
| 2.12 | §3.3 | **Session summaries become episode summaries**, plus a daily digest produced in consolidation. | One summary covered 4 of 89 observations. Its app list repeated a model typo, and its text summarized the system's own error message as user activity. |
| 2.13 | new §12 | **A retention policy is specified.** | The superseded-history table otherwise grows without bound; the prior plan left it out of scope. |

---

## 1. Purpose and scope

Stage 1 produces a stream of `Observation` rows: ground-truth app/window/URL
metadata, verbatim `screen_text`, a VLM caption, working entities and
interaction counts. Stage 2 turns that stream into a durable, queryable model
of the user:
- deduplicated observations;
- **episodes**, which are coherent stretches of one activity, each with a summary;
- typed **claims** about the user, with verified evidence;
- a resolved **entity** set with pages;
- the indexes Stage 3 reads.

Stage 2 does not answer queries.

### Pipeline  `[Rev 1]`

```
Observation (Stage 1)
   │
   ▼
[1] Dedup (stable-key exact → MinHash/LSH on screen_text)  → duplicate: bump dup_count, stop
   │
   ▼
[2] Episode segmentation (online; boundary on app/site shift, semantic shift, idle, max length)
   │   (episode closes)
   ▼
[3] Episode drafting — ONE call per episode:
      title + summary, event claims, fact/preference candidates, entities,
      evidence quotes
   │
   ▼
[4] Deterministic validation: length/atomicity, quote verification, modality, injection check
   │
   ▼
[5] Online merge (fact/preference only): recall candidates → classify
      same → REINFORCE   |   contradicts → INVALIDATE + ADD   |   refines / unrelated → ADD
   │
   ▼
[6] Entity extraction + resolution + mention recording (surface-form verified)
   │
   ▼
[7] Index write (FTS + local embedding) in the same transaction / outbox
   ═══════════════════════════════════════════════════════════════════
[8] Consolidation (offline, periodic): cluster refine (N→set), fact/preference
    induction from events, entity pages, daily digest, lint + reconciliation
```

---

## 2. Observation-level deduplication  `[Rev 1]`

Dedup runs synchronously at ingest. Frame-level duplicates, meaning the same
screen by perceptual hash, are already caught in Stage 1 (`planS1.md` §5) and
arrive only as `dup_count` bumps.

**Stable key.** `key_text = app ⏎ window_title ⏎ url ⏎ screen_text`, normalized
(Unicode NFKC, whitespace collapsed, case-folded). **The caption is never part
of any dedup key.** It is generated text that is reworded run to run for an
identical screen, so including it makes identical moments look different.

### 2.1 Tier 1 — exact

SHA-1 of `key_text`. On a match the existing row's `dup_count` increments,
`ended_at` / `updated_at` refresh, and processing stops.

### 2.2 Tier 2 — near-duplicate via MinHash + LSH

`screen_text` is shingled into word 3-grams. A MinHash signature (128
permutations) is inserted into and queried against a `MinHashLSH` index covering
all retained history (library: `datasketch`). An LSH hit with estimated
Jaccard ≥ `DEDUP_JACCARD_THRESHOLD` (default **0.80**) **and** an identical
`(app, site)` pair is a duplicate. The index is in-process and rebuilt from
storage at startup.

**Short-text guard.** When `screen_text` has fewer than 30 shingles, the
Jaccard estimate is too coarse at a 0.80 threshold. Such rows are duplicates
only if tier 1 matches, or if normalized `screen_text` is identical and `app`
and `window_title` match.

Why MinHash rather than a bit-sketch (SimHash): MinHash estimates Jaccard
directly, which is the quantity the threshold is expressed in, and its LSH
index reaches all history in sublinear time instead of a fixed recent window.

### 2.3 New observation

A surviving row is inserted. Its hash and signature are indexed, the token
document-frequency table is incremented from `screen_text` and `caption` only
(see `planS3.md` §5), and the row is handed to segmentation.

---

## 3. Episode segmentation  `[Rev 1 — new section]`

### 3.1 Why

The unit that claims and summaries are built from must be one coherent
activity. A fixed count of consecutive observations mixes activities, and a
silence-bounded session is far too coarse: one session in the prototype was
70 minutes of three different tasks. Cognitive event-segmentation research,
and LLM memory systems built on it (EM-LLM), place boundaries where
predictability drops. The implementable analogue here is a shift in app/site
or in content.

### 3.2 Boundary rule (online, evaluated per new observation)

A new episode starts when any of these hold:
1. the Stage 1 session changed, or an idle transition occurred;
2. the `(app, site)` pair changed **and** stayed changed for at least
   `EPISODE_SWITCH_MIN_S` (default **90 s**). A quick alt-tab to check
   something and back does not split an episode; a sustained switch does. The
   boundary is placed retroactively at the switch point;
3. semantic shift: the cosine similarity between the new observation's
   embedding (caption + first 1000 characters of `screen_text`) and the running
   episode centroid falls below `EPISODE_SHIFT_SIM` (default **0.55**) for two
   consecutive observations, even within the same app (e.g. switching projects
   in the same editor);
4. the episode reached `EPISODE_MAX_S` (default **1800 s**): forced boundary.

An episode **closes** at its boundary, or after `EPISODE_IDLE_CLOSE_S` (default
**300 s**) with no new observation. Closing triggers drafting (§5) exactly once.

### 3.3 Episode record

`id, session_id, started_at, ended_at, apps[], sites[], observation_ids[],
title (≤ 80 chars), summary (≤ 600 chars), entity_ids[], engaged_fraction`.

`engaged_fraction` is the share of the episode's observations with any
interaction (`planS1.md` §8). The episode summary replaces the prior
per-session summary. A **daily digest**, a short narrative over the day's
episodes, is written by consolidation (§7.4).

---

## 4. The claim model  `[Rev 1 — new section]`

| Field | Meaning |
|---|---|
| `id`, `lineage_id`, `version` | Identity and version chain (bi-temporal history). |
| `kind` | `event`: a dated thing the user did or saw ("signed in to Supabase with Google"). Never merged, never invalidated. `fact`: durable state ("uses Adobe Illustrator for logo work"). `preference`: a belief about taste or habit ("prefers monochrome palettes"), strengthened or weakened by evidence. |
| `text` | One proposition about the user, at most `CLAIM_MAX_CHARS` (default **240**) and at most 2 sentences. |
| `context` | When or where it applies ("while working on the Dooran logo"). |
| `reasoning` | Why the model proposed it. Stored for audit, **not** full-text indexed (see `planS3.md` §5). |
| `modality` | `engaged` (some supporting observation had user input), `passive` (all supporting observations were viewing-only), `stated` (user testimony, e.g. an answered knowledge gap). Computed in code from evidence (§5.2), never taken from the model. |
| `occurred_from`, `occurred_to` | **Event time**: for `event`, the episode interval; for `fact`/`preference`, the first and last supporting evidence time. |
| `valid_from`, `valid_to` | World-validity interval. `valid_to` is set when the claim is contradicted, or superseded by consolidation. |
| `created_at` | Transaction time. |
| `superseded_by` | Pointer to the replacing row, if any. |
| `status` | `live`, `superseded` (replaced by a consolidation rewrite), `invalidated` (contradicted). |
| `confidence` | Derived from evidence (§8). Never written by a model. |

**Atomicity check (code):** a claim is rejected, and the drafting call
re-prompted once, if it exceeds the length bound or joins independent clauses
about different objects. The proxy is: more than 2 sentences, or more than 3
distinct entities joined by "and" or "while". Non-destructive supersession
(old rows keep `valid_to` and `superseded_by`; evidence and mentions are carried
forward) is retained from the prior design unchanged. What changed is *when*
it happens: only in consolidation (§7) and on contradiction (§6.2), never as a
per-batch rewrite.

---

## 5. Episode drafting  `[Rev 1]`

### 5.1 The call

When an episode closes, **one** LLM call receives the episode's observations in
time order. For each observation it gets: id, time, `app`/`site`/`window_title`,
caption, interaction counts, and a `screen_text` excerpt. Excerpts are capped at
`DRAFT_TEXT_CHARS_PER_OBS` (default **1200**), and lines repeated across the
episode are sent once. All screen-derived text is delimited as untrusted data
(`planS1.md` §10). The call returns:
- `title`, `summary` for the episode;
- up to `DRAFT_MAX_EVENTS` (default **5**) `event` claims;
- up to `DRAFT_MAX_DURABLE` (default **5**) `fact`/`preference` candidates;
- for each claim: `text`, `context`, `kind`, `entities[]`, and
  `evidence: [{observation_id, quote}]`.

Schema examples in the prompt use type placeholders, never literal values.

### 5.2 Deterministic validation (no model involved)

1. **Length/atomicity:** see §4.
2. **Quote verification:** each `quote` is normalized (whitespace collapsed,
   case-folded) and accepted only if it is a literal substring of the cited
   observation's normalized `screen_text` or caption. A failed quote is
   replaced by the highest term-overlap window of that observation's real text.
   A claim with no observation reaching even a computed-snippet overlap of
   `EVIDENCE_MIN_OVERLAP` (default **0.2**) with its content words is dropped as
   unsupported. (This retains the prior quote-verification design, which worked
   in the prototype: 1380/1380 verified links carried a real quote. The
   separate evidence-mapping LLM call is folded into drafting.)
3. **Modality:** `engaged` if any cited observation has non-zero keys, clicks or
   clipboard; otherwise `passive`.
4. **Injection check:** a claim whose text contains an imperative sentence
   copied from `screen_text` (trigram containment ≥ 0.8 against any single
   sentence of the evidence) is dropped and counted.
5. **Subject check:** a claim must be about the user or the user's activity.
   Claims whose subject is solely third-party content ("the page says X is the
   best tool") are dropped.

`event` claims are written immediately with `status = live`. `fact` and
`preference` candidates go to the online merge (§6).

---

## 6. Online merge — ADD, REINFORCE, INVALIDATE  `[Rev 1]`

### 6.1 Candidate recall and classification

For each fact/preference candidate, the top `CANDIDATE_LIMIT` (default **8**)
live claims of the **same kind** are recalled. Recall uses the hybrid
lexical + semantic + RRF described in the prior design and retained here:
field-weighted BM25 over claim `text`/`context`, cosine similarity over claim
embeddings, fused with `RRF_K = 60`. It uses the same shared indexes as
Stage 3, not a parallel index. One LLM call classifies each candidate's relation
to the draft:

- **same**: the same proposition, with no new information;
- **refines**: the same subject, adding detail or scope;
- **contradicts**: the two cannot both be currently true;
- **unrelated**.

### 6.2 Operations

| Relation | Operation |
|---|---|
| `same` | **REINFORCE**: the draft's verified evidence links are added to the existing claim, and `occurred_to` extends. The claim text is untouched. |
| `contradicts` | **INVALIDATE + ADD**: the old claim gets `valid_to = draft.occurred_from` and `status = invalidated`, and the draft is added with a link `invalidates = old.id`. **Guard:** a `passive` draft may never invalidate an `engaged` or `stated` claim; in that case the draft is added and both are flagged for consolidation. |
| `refines` | **ADD** with `refines = candidate.id`. Consolidation (§7.1) folds refinement chains later. Stage 3 shows only the newest member of a refinement chain. |
| `unrelated` / no candidates | **ADD** as a new lineage, version 1. |

**The online path never rewrites claim text.** That is the structural fix for
the accretion loop: a claim cannot grow from absorbing drafts, so it cannot
become an attractor that matches ever more drafts. The prior
`INSIGHT_REVISE_COOLDOWN_S` mitigation, a rate limiter on rewrites, is retired
along with the rewrites.

---

## 7. Consolidation (offline, periodic)  `[Rev 1 — new section]`

Consolidation runs every `CONSOLIDATION_INTERVAL_S` (default **3600 s**), when
the user goes idle, and once at the end of each day. It can be re-run safely:
every step can be repeated without duplicating output.

### 7.1 Cluster refine — N inputs → a bounded set

Live fact/preference claims are clustered by `refines` links, and by claim
embedding similarity ≥ `CONSOLIDATE_SIM` (default **0.82**) within the same kind.
For each cluster with ≥ 2 members, one call returns a **set** of claims that are
clear and non-redundant. It may *merge*, *split*, or *drop as duplicate*, as
GUM's original revise step allowed. Code enforces:
- every output ≤ `CLAIM_MAX_CHARS` and atomic (§4);
- output count ≤ input count + 2;
- every input is accounted for, as kept, merged into an output, or dropped as a
  duplicate of an output;
- each output lists the input ids it derives from, and **inherits evidence only
  from those inputs**. There is no blanket carry-forward of the whole cluster's
  evidence;
- no output introduces an entity absent from its inputs' evidence.

Inputs that are replaced get `status = superseded` and `superseded_by`. Outputs
continue the lineage of their primary input. A call that fails validation
changes nothing: a failed consolidation never loses information.

**Lineage ceiling:** a lineage past `LINEAGE_MAX_VERSIONS` (default **12**) is
reported by lint as an anomaly. Its latest version is frozen, and subsequent
related drafts ADD new lineages. A lineage reaching version 45 was the
prototype's clearest failure signal, and nothing reported it.

### 7.2 Durable induction from events (layered abstraction)

Over a rolling window (`INDUCTION_WINDOW_DAYS`, default **7**), event claims and
episode summaries are grouped by shared entities and passed to an induction
call. It proposes fact/preference candidates *supported by several episodes*,
each citing the episode and observation evidence it rests on. The candidates go
through §5.2 validation and §6 merge like any other draft. This is the second
layer of a lattice: durable understanding is **added as new claims pointing
down to evidence**, never produced by lengthening existing claims. The
open-source `latticing` library implements this layered induction and should
be evaluated as the implementation before building one (§14, open item 2).

### 7.3 Entity pages

See §10.6.

### 7.4 Daily digest

At day end, a short narrative over the day's episode titles and summaries,
stored as a `digest` row keyed by date. Stage 3 uses it for "what did I do
yesterday"-style questions.

### 7.5 Lint and reconciliation

A deterministic, code-first pass. It computes metrics in code and uses a model
only where a judgment is needed, following tada's memory-lint pattern:
- index coverage reconciliation (§11) — any gap is repaired and counted;
- entity duplicate sweep: re-runs resolution (§10.4) pairwise over entities
  created since the last run;
- claims with zero verified supports older than `UNSUPPORTED_TTL_DAYS` (default
  **14**) → `status = superseded` with reason `unsupported`;
- lineage ceiling anomalies (§7.1);
- backlog report: episodes not yet drafted, outbox depth, failed calls by type.

---

## 8. Confidence — derived, not verbalized  `[Rev 1]`

Asked for a number, the model copied the prompt's example value on 94 % of
claims. Verbalized LLM confidence is documented to saturate at a few high
values regardless of accuracy. Confidence is instead computed from evidence:

```
w(modality) = stated 1.0 | engaged 0.8 | passive 0.4
E = Σ over distinct supporting episodes of w(that episode's best evidence modality)
    + 0.5 × (distinct supporting days − 1)
confidence = 1 − exp(−E / CONFIDENCE_SCALE)            (CONFIDENCE_SCALE default 3.0)
```

A single passive sighting gives about 0.13. Engaged evidence in three episodes
on two days gives about 0.63. User testimony alone gives about 0.28, and
testimony also sets `modality = stated`, which Stage 3 treats as authoritative
for its subject.

**Staleness** is not a stored model guess. Each kind has a half-life used by
Stage 3 as a recency prior: `event` none (it is dated), `fact`
`FACT_HALF_LIFE_DAYS` = **180**, `preference` `PREF_HALF_LIFE_DAYS` = **90**,
measured from `occurred_to`. The prior LLM-assigned `decay` field is removed.
Its scale was also the inverse of GUM's, a latent source of silent inversion.

The Stage 3 evaluation harness checks that confidence is not degenerate: no
single value may hold more than 50 % of live claims.

---

## 9. Evidence links  `[Rev 1]`

- `supports` links carry a verified quote (§5.2). Each claim keeps at most
  `EVIDENCE_MAX_SUPPORTS` (default **12**). When the cap is exceeded, the links
  kept are the most recent 6 plus the 6 whose quotes have the highest summed
  IDF (most specific). The others are removed from the claim, and their
  observations remain.
- `cooccurs` links (a weak "was in the same episode" link) are written only
  for a claim that has no verified support. Once a claim has ≥ 3 supports, they
  are dropped.
- Upgrade-only: a `supports` link is never downgraded to `cooccurs`.
- In consolidation, outputs inherit evidence only from the inputs they derive
  from (§7.1), then the cap applies.

A claim linked to most of the corpus is not well evidenced; it is imprecisely
evidenced. These bounds keep "why do you think this?" answerable with a handful
of specific quotes.

---

## 10. Entities  `[Rev 1]`

### 10.1 Why entities exist

"What do you know about the billing service" is thing-shaped, not
claim-shaped. An entity layer gives the store an "everything linked to X" axis
alongside "everything matching this text". This rationale is unchanged.

### 10.2 Extraction

Candidate names come from three sources:
1. the Stage 1 VLM's **working entities** per observation, meaning names the
   user is working with, not everything visible;
2. the drafting call's claim entities (§5.1);
3. a local zero-shot NER pass (GLiNER) over `screen_text`, with the label set
   `{person, organization, project, software, website, file, repository,
   product}` and score ≥ `NER_MIN_SCORE` (default **0.5**). The score is the
   salience signal the prior design lacked.

### 10.3 Salience and recurrence gates

- A name from sources 1 or 2 is **salient**: the user was working with it.
- A name only from source 3 is a **candidate**. It is recorded in
  `entity_candidates(name_norm, count, first_seen, last_seen)` and is promoted
  to an entity only after appearing in ≥ `ENTITY_MIN_EPISODES` (default **2**)
  distinct episodes.
- **Stop-entity floor:** names under 3 characters, pure numbers, single
  common-dictionary words (word-frequency Zipf ≥ 5.0 without internal capitals
  or digits), generic category words (`tech`, `saas`, `ai`, `design`), and
  document-chrome names (`Untitled-1`, `Artboard 2`) are never promoted.
- Scrubbed placeholders (`planS1.md` §9.2) are never entities.

This is what keeps a gallery of logo names the user scrolled past out of the
model of the user.

### 10.4 Resolution cascade

`kind` (person/org/project/…) is an **attribute**. It is **never part of
identity**. Aliases live in a child table `entity_aliases(entity_id,
alias_norm)` with a unique index. Resolution of a candidate name:

1. **Exact:** `alias_norm` lookup, where `alias_norm` = NFKC, case-fold,
   collapsed whitespace, stripped surrounding punctuation.
2. **Normalized key:** additionally strip non-alphanumerics, reduce web
   addresses to the registrable domain's label (`logosystem.co` → `logosystem`),
   and strip a small list of generic suffixes (`font`, `app`, `inc`, `ltd`,
   file extensions). A match is an auto-merge.
3. **String-fuzzy:** `rapidfuzz` `WRatio` over normalized keys, both at least
   5 characters. At ≥ **96** it auto-merges. At 88–96 it goes to adjudication
   (step 5).
4. **Semantic candidates:** cosine ≥ **0.80** against entity name embeddings,
   plus BM25 over entity names and page summaries. Candidates go to
   adjudication.
5. **LLM adjudication:** one call sees the candidate name with its mention
   context and each candidate entity's page summary, and answers
   same/different. It follows Zep's resolution step. If no model is available,
   no merge happens: a duplicate entity is cheap to merge later, and a false
   merge is not.

**Why this reverses the prior embedding-only decision:** the prototype's real
duplicates were string-level: typos, suffixes, domain variants, OCR errors
(`OpenC1aw`). String similarity handles those precisely, and embeddings of
short proper nouns do not. Embeddings also score distinct brands with similar
spelling (`Grainient` / `Grabient` / `Gradientos`) as near-identical, which is
exactly the irreversible false merge the prior plan warned about. Embeddings
remain useful for meaning-level aliases ("the auth service" ↔ "authentication
layer"), which is why they feed candidates to a decision step rather than
deciding alone.

### 10.5 Mentions and co-mention edges

- An entity→observation mention is recorded **only if** one of the entity's
  aliases occurs (normalized substring) in that observation's `screen_text`,
  caption, `window_title` or `url`. The prior design attributed every draft
  entity to every observation in the batch; that linked the user entity to 88 %
  of all rows and made the edge meaningless.
- An entity→claim mention requires the alias to occur in the claim text or in
  one of its verified quotes.
- `mention_count` is computed from the mention table (a query or view), never
  stored as a counter.
- `entity_edges(a, b, weight, last_seen)`: incremented once per episode in which
  both entities are mentioned. This is cheap, and it is the substrate for
  Stage 3's optional graph expansion.

### 10.6 Entity pages

In consolidation, each entity whose mentions changed since its last refresh
gets its page regenerated. Pages are refreshed in order of distinct-episode
count, so the most-used entities come first. A page is `kind`, a summary
(≤ 800 chars), and a list of its key live claims by id. It is synthesized only
from live claims and episode summaries mentioning the entity. Pages are the
prior "dossiers"; what changed is that only salience- and recurrence-qualified
entities get them, in usage order.

---

## 11. Index coverage invariant  `[Rev 1 — new section]`

**Invariant:** every live observation, episode, claim and entity has (a) its
full-text index row and (b) exactly one embedding for the current embedding
model.

- Full-text rows are maintained by triggers, which are already transactional.
- Embeddings are computed **in the write path by a local embedding model**
  (a Qwen3-Embedding-0.6B- or EmbeddingGemma-300M-class model on CPU/MLX,
  milliseconds per row) and written in the same transaction as the row. If the
  embedder is unavailable, an `index_outbox(ref_kind, ref_id)` row is written in
  the same transaction instead, and a worker drains it.
- INVALIDATE and supersede remove the old row from the live dense index and add
  the new one in the same transaction.
- A startup reconciliation plus the lint pass (§7.5) repair any drift. The
  store card (`planS3.md` §9) reports coverage %. The evaluation harness fails
  if coverage is under 100 %.
- The embedding model's identifier is stored per vector. Changing the model
  schedules a full re-embed; mixed-model vectors are never compared.

Why a local embedder: it makes the transactional write possible (no network in
the commit path), it removes a per-row API cost, and it keeps screen text from
leaving the machine for indexing. Where the vectors live (plain BLOB plus a
brute-force scan, or `sqlite-vec`) is a `planS3.md` §4 decision.

---

## 12. Retention  `[Rev 1 — new section]`

| Data | Retention |
|---|---|
| Frame JPEGs | `FRAME_TTL_DAYS` = **3**, then deleted. |
| Observation `screen_text` | `SCREEN_TEXT_TTL_DAYS` = **90**, then cleared (caption and metadata kept). DF counts are decremented on clear. |
| Episodes, digests | Kept indefinitely. |
| Live claims, entities | Kept indefinitely. |
| Superseded/invalidated claims | Kept indefinitely as a lineage skeleton (text, validity interval, pointers). Evidence links beyond 3 per superseded row are pruned after 180 days. |

Every purge updates DF and mention accounting in the same transaction.

---

## 13. Full tunable reference  `[Rev 1]`

| Name | Default | Section |
|---|---|---|
| `DEDUP_JACCARD_THRESHOLD` | 0.80 | 2.2 |
| `DEDUP_MINHASH_PERMUTATIONS` | 128 | 2.2 |
| `EPISODE_SWITCH_MIN_S` | 90 | 3.2 |
| `EPISODE_SHIFT_SIM` | 0.55 | 3.2 |
| `EPISODE_MAX_S` | 1800 | 3.2 |
| `EPISODE_IDLE_CLOSE_S` | 300 | 3.2 |
| `CLAIM_MAX_CHARS` | 240 | 4 |
| `DRAFT_TEXT_CHARS_PER_OBS` | 1200 | 5.1 |
| `DRAFT_MAX_EVENTS` / `DRAFT_MAX_DURABLE` | 5 / 5 | 5.1 |
| `EVIDENCE_MIN_OVERLAP` | 0.2 | 5.2 |
| `CANDIDATE_LIMIT` | 8 | 6.1 |
| `RRF_K` | 60 | 6.1 |
| `CONSOLIDATION_INTERVAL_S` | 3600 | 7 |
| `CONSOLIDATE_SIM` | 0.82 | 7.1 |
| `LINEAGE_MAX_VERSIONS` | 12 | 7.1 |
| `INDUCTION_WINDOW_DAYS` | 7 | 7.2 |
| `UNSUPPORTED_TTL_DAYS` | 14 | 7.5 |
| `CONFIDENCE_SCALE` | 3.0 | 8 |
| `FACT_HALF_LIFE_DAYS` / `PREF_HALF_LIFE_DAYS` | 180 / 90 | 8 |
| `EVIDENCE_MAX_SUPPORTS` | 12 | 9 |
| `NER_MIN_SCORE` | 0.5 | 10.2 |
| `ENTITY_MIN_EPISODES` | 2 | 10.3 |
| Fuzzy auto-merge / adjudicate band | 96 / 88–96 | 10.4 |
| Entity embedding candidate threshold | 0.80 | 10.4 |
| `FRAME_TTL_DAYS` / `SCREEN_TEXT_TTL_DAYS` | 3 / 90 | 12 |

Retired by Rev 1: `MIN_BATCH`, `MAX_BATCH` (replaced by episodes),
`INSIGHT_REVISE_COOLDOWN_S` (no online rewrites),
`ENTITY_RESOLUTION_SIMILARITY_THRESHOLD = 0.85` (replaced by the cascade),
`ENTITY_DOSSIER_MIN_MENTIONS` (replaced by the recurrence gate and
usage-ordered refresh), the LLM `decay` field, and `SESSION_SUMMARIES`
(replaced by episode summaries and the daily digest).

---

## 14. Resolved decisions and open items

### Resolved (Rev 1)

1. The dedup key never contains generated text. MinHash+LSH over `screen_text`,
   with a short-text guard.
2. Episodes, not fixed batches, are the drafting unit.
3. Typed claims (`event` / `fact` / `preference`), bounded length, one
   proposition each.
4. The online path is ADD / REINFORCE / INVALIDATE, with no text rewriting.
5. Rewriting happens only in offline consolidation, N→set, bounded, validated,
   with evidence inherited per derivation.
6. Confidence is derived from evidence; the kind half-life replaces the LLM
   `decay`.
7. Entities: salience and recurrence gates; the exact → normalized → fuzzy →
   semantic → LLM cascade; kind is not identity; surface-verified mentions.
8. The index coverage invariant uses a local embedder and an outbox.
9. Retained unchanged from the prior design: literal-substring quote
   verification with a computed-snippet fallback; non-destructive bi-temporal
   supersession with lineage and version; hybrid lexical + semantic + RRF
   candidate recall over shared indexes; `origin = agent` exclusion.

### Open

1. **Segmentation thresholds** (`EPISODE_SHIFT_SIM`, `EPISODE_SWITCH_MIN_S`)
   are initial values. They are calibrated on a few hand-labelled days of real
   captures before being treated as settled.
2. **`latticing` versus in-house induction for §7.2.** Evaluate the library
   against the claim model here (typed, bounded, evidence-linked). Adopt it if
   its output can be mapped onto §4 without losing evidence links.
3. **The consolidation model tier.** Consolidation is offline and
   latency-insensitive, so it can use a stronger model than the online path.
   The choice is made by the Stage 3 harness.
4. **Graph use of `entity_edges`** is decided in Stage 3. Stage 2 only
   maintains the edges.

---

## 15. Out of scope for Stage 2

- Query answering, ranking and grounding: `planS3.md`.
- Capture, triggers, text extraction and privacy gates: `planS1.md`.

## 16. Overall assessment

The prior design's components were individually reasonable: quote
verification, bi-temporal history, hybrid recall. Combined, they produced a
runaway. Rev 1 changes the store's *write semantics* so that runaway cannot
happen by construction. Online writes only add, reinforce or close claims.
Text is rewritten only offline, into bounded sets. Understanding deepens by
adding a layer, not by lengthening a row. The empirical direction of the field
agrees: mem0's move to ADD-only, verbatim memory outperforming rewritten
artifacts, Zep's episode/semantic split with invalidation, and Hindsight's
fact/belief separation. The remaining open items are calibrations and library
choices, not structural gaps.
