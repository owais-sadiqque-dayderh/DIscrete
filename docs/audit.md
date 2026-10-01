# DIscrete — Architecture Audit (2026-09-14)

> **Routing status (2026-09-14).** Every finding in this document has been
> routed. Design findings are applied as **Rev 1** of `planS1.md`, `planS2.md`
> and `planS3.md`, each listed in that plan's Revision log. Implementation bugs
> are recorded in `insights.md` §7. This file remains the evidence record the
> revisions cite. Where its recommendations differ from the plans, the plans
> win: Rev 1 went further after a web survey. For example, entity PII detection
> uses a GLiNER2-PII-class model rather than Presidio alone, and the online
> write path is ADD-only rather than a bounded revise.

An adversarial review of `planS1.md`, `planS2.md`, `planS3.md`, checked against
the real WorkAA store at `~/.cache/workaa/workaa.db` (a 70-minute recorded
session) and against the reference implementations in `gum/`, `lattice/`,
`napsack/`, `tada/`.

This is a findings document, not a decision record. Nothing here edits the
frozen plans; where a finding should change a plan, it says so and the change
belongs in a new record.

---

## 0. The evidence base

One real session, 2026-09-12 21:08:19 → 22:18:11 UTC (70 minutes), captured by
WorkAA with essentially the pipeline planS1/planS2 specify.

| Table | Rows | Notes |
|---|---|---|
| `observations` | 89 | 88 `origin=user`, 1 `agent`; avg `dense_text` **150 chars** |
| `propositions` | 64 | **only 6 live**; 58 superseded |
| `entities` | 123 | **106 have an empty summary**; 111/123 `kind='topic'` |
| `evidence_links` | 2190 | 1380 `supports`, 810 `cooccurs` |
| `entity_mentions` | 1808 | 14.7 per entity, over 153 total referenceable rows |
| `session_summaries` | 1 | covers **4 of 89** observations |
| `embeddings` | 200 chunks | 85/89 observations, **27/64 propositions**, **0 entities** |
| `gaps` | 4 | all `status=open`; 2 of them were answerable from the store |
| `token_df` | 1255 | entire corpus vocabulary |

Everything below is traced to this data unless marked as a design-only concern.

---

## 1. The headline finding: one causal chain produced almost every defect

The failures are not independent. They are one chain, and fixing any single
link without the others will not hold.

```
  VLM caption is part of the dedup key
        │
        ▼  (captions are re-worded every run, so near-identical screens hash apart)
  Dedup catches ~nothing — 84/89 rows have dup_count=1;
  15 separate rows whose entire dense_text is the single word "discrete"
        │
        ▼  (batcher sees 15 near-identical observations instead of 1)
  15 near-identical proposition DRAFTS
        │
        ▼  (each one "similar"-matches the claim drafted 40 seconds earlier)
  REVISE_PROMPT: "merge into ONE claim ... drop nothing factual"
        │
        ▼  (N inputs → 1 output that must retain all N, with no length bound)
  Monotonic accretion: 45 versions in 35 minutes, text 89 → 1001 chars
        │
        ▼  (a longer claim lexically/semantically matches MORE future drafts)
  Runaway attractor: 89 observations collapse to 6 live propositions,
  one of which holds 90% of the session and is linked to 80 of 89 observations
```

Measured growth of lineage `a090102c` (mean text length by version across all
lineages): **84 → 233 → 281 → 389 → 496 → 648 → 734 → 984 → 1192 chars** at
versions 1, 2, 3, 5, 7, 10, 14, 20, 27.

The terminal state, verbatim — this is the single most important live
proposition in the store:

> *"Owais is actively engaged in a comprehensive design project titled
> 'Japaneasy - 01 - Unlimited Gradients and AI-Generated Backgrounds' in Adobe
> Illustrator 2026, where he is focusing on editing the text 'discrete' across
> multiple artboards, specifically on Artboard 3. He is utilizing a gradient
> from the 'Chaotic Gradients' collection on the Grainient website, exploring
> resources from TOOLS.design and Grainient for gradient design tools, managing
> collections in the Logosystem app, and browsing design resources on the
> Logofolio website for logo design inspiration. Additionally, he is
> incorporating AI-generated backgrounds into his design work... including logo
> designs for 'Dooran' and 'HyperFlow' that target the Tech, SaaS, and Web3 &
> Crypto industries..."*

That is a session summary wearing a proposition's clothes. It is not a claim,
it cannot be superseded meaningfully, it cannot be contradicted, and it matches
every future draft about any of the eight topics it absorbed.

**`planS2.md` §5.2 specifies this exact merge rule and does not bound it.** The
`INSIGHT_REVISE_COOLDOWN_S` cooldown is presented as the mitigation; §5 below
shows it is both insufficient in principle and broken in the implementation it
was written from.

---

## Stage 1 — Data Collection

### S1-A ▸ `dense_text` is ~150 characters. The corpus is 70 minutes of screen time and ~22 KB of text. — **HIGH**

`planS1.md` §6.2 describes `dense_text` as "the substantive on-screen text —
article body, code, message text, search results." Measured: mean 150 chars
(keyboard-triggered 166, mouse-triggered 142). Fifteen observations have
`dense_text` = `"discrete"`. Several are three lines of Illustrator panel
labels.

The entire retrieval corpus is ~1255 distinct tokens. Every downstream IDF,
coverage-floor and BM25 tuning in planS3 is being set against a vocabulary that
small. This is the root reason retrieval is fragile, and no amount of reranking
fixes it.

**Why it happens:** the VLM is asked to *select* substantive text and
*exclude chrome*, and it is conservative — it returns a summary of the text
rather than the text. planS1 §6.2 asserts that supplying OCR ground truth turns
this into "select and organize what's substantive from text already extracted."
The data says the model still heavily abridges.

**Recommendation:** store the raw Apple Vision OCR blob as its own column
(`ocr_text`) alongside the VLM's curated `dense_text`, and index both — OCR at
low BM25 weight, curated text at high. OCR is free and local; there is no cost
argument for discarding it. This also gives dedup a stable key (S1-B) and gives
IDF a real vocabulary.

### S1-B ▸ The dedup key includes the generated caption, so dedup cannot work. — **CRITICAL** (root of the chain in §1)

Both the WorkAA implementation and `planS2.md` §2.1/§2.2 build the dedup key
from `app + window_title + url + caption + dense_text`. `caption` is
free-form VLM output that varies run-to-run for an identical screen:

- obs 26: *"editing text in Adobe Illustrator, specifically the word 'discrete' on Artboard 2"*
- obs 59: *"working on a text design in Adobe Illustrator, specifically editing the word 'discrete'"*
- obs 64: *"designing a text graphic in Adobe Illustrator, focusing on the word 'discrete'"*

All three have `dense_text = "discrete"`. All three stored as separate rows.
Trigram Jaccard over the combined string lands well under 0.80 because the
caption dominates the token count. Result: **84 of 89 rows have `dup_count=1`.**

Note this is a Stage 1 finding even though dedup lives in Stage 2 — the defect
is that Stage 1 hands Stage 2 a record whose only high-volume field is
non-deterministic.

**Recommendation (three parts, cheap):**
1. Dedup on **stable signal only**: `app + window_title + url + ocr_text`.
   Never on `caption`.
2. Add a **perceptual image hash** (`imagehash` pHash/dHash, ~10 lines, or the
   same primitive `tada` uses) on the downscaled frame as a tier-0 check
   *before* any text work. Two screenshots of the same paused screen are
   pixel-near-identical; that is the cheapest and most reliable duplicate
   signal available and the pipeline currently throws it away.
3. Only then run the MinHash/LSH tier planS2 §2.2 specifies.

### S1-C ▸ `app` is a free-text generated field and is full of typos. — **HIGH**

25 distinct `app` values for 89 observations, where the true count is ~8:

| Cluster | Observed variants |
|---|---|
| Logosystem | `logosystem.co` (4), `logosesystem.co` (2), `logosytem.co` (1), `Logosystem` (1), `Logosytem` (1) |
| Grainient | `Grainient` (3), `Graident` (2), `Grainient / Adobe Illustrator` (1) |
| Logofolio | `Logofolio` (2), `logofoilo.com` (2) |
| ASCII Magic | `Ascii Magic` (7), `ASCII Magic` (4) |
| Arc | `Arc` (8), `Arc Browser` (2) |
| compound | `LogoFolio and ASCII Magic` (1) |

`planS1.md` §6.2 defines `app` as "the foreground application **or website
name**" — that instruction is what invites the model to overwrite a
ground-truth application name with a guessed site name. §6.3's AX override then
cannot help, because AX returns `Arc` and the model wants `MyFonts`, and the
plan's own field definition says the model is right.

Consequences: `store_card`'s "most-seen applications" is wrong; the one session
summary that exists lists `logosesystem.co` as an app the user used; `app` is
BM25-weighted at 3.0 and is feeding typos into the index and into `token_df`.

**Recommendation:** split into two fields. `app` = **AX only, never
model-written**, a closed vocabulary. `site` = the model's/extension's reading
of the web property, normalized to a registrable domain
(`tldextract`, one line) so `logosytem.co`/`Logosystem`/`logosystem.co` all
collapse. Then adopt the transcript's own recommendation and put
`aw-watcher-web` in front of both for browsers — `planS1.md` §13 defers this,
but the typo cluster above is the concrete cost of deferring it.

### S1-D ▸ The before/after frame pair causes cross-app contamination. — **MEDIUM**

`planS1.md` §5 mandates a before/after pair so enrichment can describe a
*change*. Observed side effect: when a window change lands inside the debounce
window, the two frames are different applications and the model fuses them.

- obs 30 caption: *"working with an ASCII art generator **and then viewing a collection of logos**"* — `app` recorded as `Ascii Magic`.
- 12 rows carry `app = "Adobe Illustrator 2026"` with `window_title = "ASCII Magic Editor, Free Online Image..."` — a browser tab title attached to a desktop app.
- `app` values `"LogoFolio and ASCII Magic"` and `"Grainient / Adobe Illustrator"` are the model honestly reporting that it saw two apps.

planS1 does not mention this failure mode. **Recommendation:** abort/re-arm the
pending capture if the AX foreground app at the *after* frame differs from the
*before* frame. This is the same guard as the transcript's "mid-generation
focus-drift detection" (§4 of the research pass), applied to capture. Cheap,
deterministic, and it also removes a class of bad `app` values from S1-C.

### S1-E ▸ No sensitive-content policy anywhere in Stage 1. — **HIGH** (product risk, not a bug)

The store contains, verbatim and unredacted:
- `dayderhowaissadique@gmail.com` — stored both in `observations.dense_text` and promoted to a first-class **entity**.
- `slqhpbeqtikghlwlzta.supabase.co` — a live Supabase project ref, also an entity.
- The full contents of a Google OAuth consent screen (obs 39).

`planS1.md` §6.5 drops captures the model marks `useful:false`; there is no
concept of *sensitive*. For a product whose entire premise is continuous screen
recording, this is a first-class design gap, and the three plans are silent on
it. Password managers, banking, health, private DMs, `.env` files in an editor,
and OAuth screens are all in scope of the capture loop today.

**Recommendation (Stage 1, before implementation):**
- An **app/URL denylist** (`ACTIVITY_EXCLUDE_APPS`, `CAPTURE_EXCLUDE_URL_PATTERNS`), suppressing capture entirely, not filtering after the fact. `tada` and ActivityWatch both ship this pattern.
- A **secret-shaped-string scrubber** on `ocr_text`/`dense_text` before storage — `detect-secrets` or `gitleaks`' rule set, or Microsoft **Presidio** for PII (emails, phone, card, SSN). Presidio is the mature off-the-shelf answer here and matches the "don't reinvent" principle.
- A hard rule that scrubbed spans never reach the entity extractor.

### S1-F ▸ The two open questions in §12 should be closed now, not "before or during implementation."

- **Audio**: `planS3.md` §3.1 promises to answer "what did I actually do this morning." Meetings and calls are a large share of that, and Stage 1 captures none of it. If audio is out of scope, the product claim needs narrowing; if not, it is a Stage 1 component, not a footnote. Worth noting the machine already runs **screenpipe**, which does exactly this (24/7 screen + audio + OCR, local, with a REST API) — a serious "already built" candidate for the whole capture tier rather than a component.
- **Multi-monitor**: capturing only the mouse's monitor means the passive-session blind spot that §2.3 exists to close reopens on any second display. At minimum, capture the AX-focused window's monitor, not the mouse's.

### S1-G ▸ Heartbeat design chose the option the research pass recommended against. — **LOW**

`planS1.md` §11 resolves the heartbeat recapture as fixed-count (Option A). The
research pass in the transcript explicitly recommended Option D (content-driven
with a ceiling and floor). With OCR now running on every heartbeat anyway (§6.4),
the OCR-diff signal Option D needs is **already computed and free** — the only
argument for Option A (implementation simplicity) largely evaporates. Worth
revisiting; low urgency.

---

## Stage 2 — Data Processing

### S2-A ▸ The revise rule is an unbounded accretion loop. — **CRITICAL**

This is §1's chain, and it is the most important thing in this document.

The prompt WorkAA ran (and which `planS2.md` §5.2 describes faithfully) says:

> "merging several overlapping claims ... into **ONE** claim"
> "Keep everything true and specific from the inputs; **drop nothing factual**."
> "Do not average two different facts into a vaguer one. If they really are
> different facts, **keep the stronger and note the other in `reasoning`**."

N inputs → exactly 1 output → which must retain all N facts → with no length
bound and no escape hatch. Growth is structurally guaranteed.

**GUM, which this was derived from, does not do this.** Its `REVISE_PROMPT`
returns a **set**, and explicitly permits *split*, *add*, and *remove*, with the
instruction *"Keep each proposition clear and concise (typically 1-2
sentences)."* Discrete's lineage dropped the N:M shape and the length bound, and
kept only the merge. That is the regression.

**Recommendation — four changes, in priority order:**
1. **Restore N:M revision.** The revise call must be allowed to return
   *multiple* propositions (merge *and* split), the way GUM does. A single
   claim that has absorbed eight topics should split back into eight.
2. **Hard length bound**, enforced in code, not prompt: reject/re-prompt any
   proposition over ~240 chars or ~2 sentences.
3. **Separate episodic from semantic claims** (see S2-B). The blob formed
   because a "right now I'm on Artboard 3" claim and a "Owais works in
   Illustrator" claim live in the same table under the same merge rule.
4. **A lineage version ceiling** as a backstop: past ~10 versions, force
   *create-new* instead of revise, and let the old lineage retire. A lineage
   hitting v45 is a bug signal, and nothing currently reports it.

### S2-B ▸ There is no hierarchy. Everything is flat, so abstraction has nowhere to go but "longer." — **CRITICAL (architectural)**

Discrete has one claim layer. `lattice/` — which is already sitting in this
repo, is a published method (Zhao, Lam, Yang, Bernstein, arXiv 2604.07629), and
is `pip install latticing` — is built around exactly the thing Discrete is
missing:

| Layer | Lattice | Discrete today |
|---|---|---|
| L0 | raw observations | `observations` ✅ |
| L1 | per-session insights, synthesized across a window | `propositions` ❌ *and* `session_summaries` ❌ (both degenerate — see S2-F) |
| L2+ | cross-session patterns, induced from L1 | **does not exist** |

The key structural property: in lattice, generalization **adds a node at a new
layer with edges down**. It never rewrites the layer below. That is precisely
what makes it immune to S2-A's runaway — an L1 node is bounded by its window,
and cross-session insight is a *new row*, not a *longer row*.

Discrete's propositions table is being asked to hold, simultaneously:
- "Owais is editing the text 'discrete' on Artboard 3" (episodic, minutes)
- "Owais explores fonts on MyFonts, dafont, and Adobe Fonts" (session-level)
- "Owais prefers monochrome over sepia palettes" (durable preference)

These need different merge rules, different retention, and different retrieval
weights. Putting them in one table with one similarity-merge rule is why the
episodic ones swallowed the durable ones.

This is also the axis on which the mem0 / Zep comparison actually bites.
Discrete already has the bi-temporal machinery Zep/Graphiti has (`valid_to`,
`superseded_by`, `lineage_id`) — which is genuinely good and ahead of mem0 — but
bi-temporality only pays off when rows have a *type* and an *expected
lifetime*. Right now every row is the same type, so the invalidation machinery
has nothing to key off and 91% of the table is dead history after 70 minutes.

**Recommendation:** add an explicit `kind` to propositions
(`episodic` / `activity` / `preference` / `trait`), derived from — and this is
the natural, already-paid-for use — the `decay` field. Episodic claims are
never revised, only expired. Only `preference`/`trait` claims enter the
revise-and-supersede path. Then add an L2 pass (weekly/periodic) that induces
durable claims *from* accumulated episodic ones rather than by rewriting them.
`latticing` can be adopted directly for that L2 pass rather than hand-built.

### S2-C ▸ The revise cooldown is dead code — it raises `TypeError` on every evaluation. — **CRITICAL (implementation), HIGH (spec)**

`planS2.md` §5.2 rests the anti-thrash guarantee entirely on
`INSIGHT_REVISE_COOLDOWN_S`. The code it was written from:

```python
cooling_down = similar and any(
    (utcnow() - p.created_at).total_seconds() < Tunables.INSIGHT_REVISE_COOLDOWN_S
    for p in similar
)
```

`utcnow()` returns `datetime.now(timezone.utc)` (**aware**). `Proposition.created_at`
is `DateTime(timezone=True)` with `server_default=func.now()`, and SQLite has no
timezone type, so SQLAlchemy hands back a **naive** datetime. Verified directly
against the real database:

```
created_at = datetime.datetime(2026, 9, 12, 21, 43, 12)
EXCEPTION: TypeError can't subtract offset-naive and offset-aware datetimes
```

The exception is swallowed by the per-draft `except Exception` a few lines
below. The guard has never once evaluated.

Two things follow:
1. **The mitigation planS2 §5.2 relies on has never been exercised.** Its
   effectiveness is unmeasured, not "measured and adequate."
2. **Even if it worked, it is a rate limiter, not a fix.** At 300 s over an
   8-hour day it permits ~96 revisions of one lineage. The blob still forms; it
   forms slower. S2-A's structural fixes are required regardless.

Also: the file's mtime (2026-09-13 04:00 local = 22:30 UTC) is *after* the last
proposition in this session (22:00 UTC), so the guard was added as a fix
immediately after this data was produced and was never run against anything.
This is exactly the process failure `insights.md` §6 describes, recurring.

**Recommendation:** store all timestamps as UTC-aware and normalize on read
(a single `_aware()` helper at the model layer), and add a regression test that
asserts the cooldown actually suppresses a second revise. More generally: any
guard a frozen spec depends on needs a test, or the spec is asserting something
unverified.

### S2-D ▸ Entity mentions are batch-level — the exact bug `insights.md` §2 says was fixed for evidence, never fixed for entities. — **HIGH**

`planS2.md` §7.2, verbatim: entity names "are additionally attributed to
**every observation in the batch** that produced the draft."

That is `insights.md` §2's described failure, restated as the design. The code
does `for observation in batch: for name in draft.entities: add_mention(...)`,
so a 15-observation batch with 10 entities writes 150 mentions with no check
that the entity appears in that observation.

Measured result: **1808 mentions across 153 referenceable rows.** `Owais` is
linked to 135 of 153 (88%). `Dooran` to 103. `logosystem.co` to 103. Every
entity is "mentioned" almost everywhere, so the entity→observation edge carries
no information and `ENTITY_DOSSIER_MIN_MENTIONS` gates on a meaningless number.

**Recommendation:** an entity→observation mention requires the surface form (or
a known alias) to actually appear in that observation's text. This is a
substring check — the same cheap, deterministic verification planS2 §6.2
already applies to evidence quotes, applied one table over. No LLM call.

### S2-E ▸ Entity resolution: the plan adopts the wrong tool for the observed failure population. — **HIGH**

`planS2.md` §7.3 adopts embedding similarity at 0.85 and explicitly argues
*against* fuzzy string matching ("it cannot connect 'the auth service' to 'our
authentication layer'"). The real duplicate population says the opposite.

**Actual duplicate/near-duplicate entities in the store:**

| Type | Examples |
|---|---|
| Typos / OCR errors | `TOOLS.design`/`TOOL.design`, `Grainient`/`Graident`, `Logofolio`/`logofoilo.com`, `OpenC1aw` (for OpenClaw) |
| Suffix variants | `Silkscreen`/`Silkscreen font`, `Monotype Foundry`/`Monotype Foundry Fonts`, `ascii-magic_1.png`/`ascii-magic-1.png` |
| Domain variants | `grainient.supply`/`graineint.supply`/`Grainient` |
| **Same name, different `kind`** | `Owais`×2, `Workaa`×2, `logosystem.co`×2, `MyFonts`×2, `Adobe Fonts`×2, `dafont.com`×2, `Adobe Illustrator`×2, `ASCII Magic`×2, `Logofolio`×2, `HyperFlow`×2, `off chart`×2, `Dooran logo`×2 |

Three separate defects here:

1. **The identity key includes `kind`.** The table constraint is
   `UNIQUE (norm_name, kind)` and `upsert_entity` looks up on
   `(norm_name, kind)`. The LLM guesses `kind` inconsistently, so the same
   entity forks into two rows with two separate mention histories — including
   the user themself. `planS2.md` §7.3 specifies a kind-free normalized-name
   lookup, which would fix it, but never says the `kind` column must be
   dropped from the identity key, and the schema carries it forward silently.
   **`kind` must be an attribute, never part of identity.**
2. **Embeddings are the wrong instrument for this population.** These are
   string-level variants. Worse, short proper-noun strings embed badly: the
   store contains `Grainient`, `Grabient`, `Gradient Art`, `Gradientify`,
   `Gradientos`, `Gradiientos` — several of which are *genuinely different
   products* and would plausibly clear 0.85 cosine against each other. So an
   embedding-only cascade both **misses the typos it should catch** and
   **risks false merges the plan itself calls "expensive and not easily
   reversed."**
3. **Substring alias matching is actively harmful.** `entity_by_name` falls
   back to `Entity.aliases.ilike('%{norm}%')` over a concatenated alias blob,
   so a 3-char entity like `art` matches any entity with `Artboard` in its
   aliases. The store contains entities named `AI`, `art`, `after`, `chart`,
   `diet`, `indie` — all live grenades for this code path.

**Recommendation — a real cascade, cheapest and most precise first:**
`exact normalized` → **`rapidfuzz`** (`token_set_ratio` ≥ 90, plus a
domain-suffix/plural normalizer) → embedding (≥ 0.88, and only for candidates
≥ 6 chars) → LLM adjudication for the narrow band that only embeddings matched.
`rapidfuzz` is C++-fast, has no model dependency, and is purpose-built for
exactly the typo/suffix class that dominates here. The transcript's research
pass already surfaced it; planS2 declined it on reasoning the data contradicts.

### S2-F ▸ Entity extraction has no salience filter; 90% of entities are noise, and the dossier budget goes to the noise. — **HIGH**

123 entities from 89 observations. A representative sample of what is stored as
a first-class entity about the user:

> `PROFIT`, `YEBUAH`, `ZARIRA`, `Flukie`, `Woonaz`, `minimin`, `arora`, `RYNO`,
> `spreadit`, `conflow`, `Hellenist`, `NorthKai`, `Wavro`, `DRUMROLL`, `DÉMPO`,
> `Bouri`, `Primus`, `DECO`, `NORTON`, `Chicken Dinner`, `art`, `after`,
> `chart`, `diet`, `indie`, `AI`, `Tech`, `Gaming`, `SaaS`, `Artboard 1`,
> `Artboard 2`, `Artboard 3`, `Untitled-1`, `discrete`

Most of these are **logo names in a gallery the user scrolled past** on
logosystem.co. The extractor has no notion of whether a proper noun is
*something the user relates to* versus *something that was rendered on screen*.

And the dossier budget is spent on precisely the wrong rows. **106 of 123
entities have an empty `summary`.** The 17 that got an LLM-synthesized dossier
include `Tech` (523 chars), `SaaS` (522), `indie` (510), `Lore` (513),
`Web3 & Crypto` (473) — while `Adobe Illustrator`, `Owais`, `Grainient`,
`MyFonts` got nothing.

This matters directly for Stage 3: `planS3.md` §3.1 calls entities "cheapest to
query, highest precision when it hits," and §3.3 builds a whole
redundancy-suppression mechanism against entity dossier bodies. **86% of entity
rows have no body at all.** The entity kind contributes almost nothing to
retrieval today, and §3.3 is solving a problem that does not yet exist.

Compounding this, `mention_count` is a denormalized counter that has already
drifted from the link table on **every single row**: `Owais` 85 vs. 135 actual,
`Dooran` 63 vs. 103, `logosystem.co` 67 vs. 103 — all undercounts. The dossier
gate reads the wrong number.

**Recommendation:**
- Gate entity creation on **recurrence**: a proper noun becomes an entity only
  after appearing in ≥ 2 *distinct, deduplicated* observations. Nearly every
  name in the list above appears exactly once.
- Add a **stop-entity floor**: reject single common words, ≤ 3 chars, and
  pure-generic category terms.
- **Drop `mention_count`** and compute it from `entity_mentions` (there is an
  index on `entity_id`; this is free). A denormalized counter that is wrong on
  100% of rows should not gate an LLM call.
- Adopt **GLiNER** as the first-pass extractor, as the transcript's research
  recommended and `planS2.md` §11 left open. It is local, CPU-friendly,
  zero-shot, returns a *typed* entity with a *confidence score* — which is the
  salience signal that is currently missing entirely — and it removes entity
  extraction from the drafting prompt, where it is currently competing for the
  model's attention with claim-writing.

### S2-G ▸ `confidence` and `decay` carry zero information, because the prompt hands the model the answer. — **MEDIUM**

| `confidence` | rows | | `decay` | rows |
|---|---|---|---|---|
| 7.0 | 1 | | 2.0 | 1 |
| **8.0** | **60** | | 3.0 | 2 |
| 9.0 | 3 | | **4.0** | **56** |
| | | | 5.0 | 3 |
| | | | 6.0 | 2 |

94% of propositions have `confidence = 8.0`; 88% have `decay = 4.0`. The cause
is in the prompt: the JSON schema example is
`{"proposition": "...", "confidence": 8, "decay": 4, ...}` — literal values, not
placeholders. The model is copying the example.

So `planS2.md` §11's open item ("the `decay` field is written but never
consumed") understates the problem. The field is not merely unused — **it is
uncalibrated, so any future consumer of it would be reading noise.** And it is
the field S2-B wants to key episodic-vs-durable classification off.

Also worth flagging as a live inversion risk: Discrete's convention is
*1 = permanent, 10 = ephemeral*. **GUM's is the exact opposite**
(*"1 (short-lived) to 10 (long-term relevance)"*). These prompts are derived
from GUM's. `planS3.md` §8 assigns knowledge-gap answers `decay = 1.0` meaning
"permanent." Any future copy of GUM prompt text, or any contributor reading the
GUM paper, silently inverts the field. Rename it (`durability`, higher = longer)
or document the inversion loudly.

**Recommendation:** replace literal values in prompt schemas with
`<int 1-10>`; add 2-3 few-shot examples spanning the range; and validate
calibration with a distribution check in the eval harness. Better still, derive
confidence from measurable signal (verified-supports count, `dup_count`,
distinct-session count) rather than asking the model for a number.

### S2-H ▸ Evidence links are quote-verified but not relevance-verified, and carry-forward makes them grow without bound. — **HIGH**

The quote-verification design (planS2 §6.2) is genuinely good and works: 1380 of
1380 `supports` links carry a non-empty quote, 810 of 810 `cooccurs` links
correctly carry none.

But nothing bounds *how many* observations may support one claim. Live
proposition 64 has **58 `supports` + 22 `cooccurs` = 80 links across a
corpus of 89 observations (90%)**. "Verified" here means the quote is real, not
that the observation is evidence for the claim. Mean links per proposition is
34; mean per observation is 26.

This interacts with §5.2's carry-forward rule: every revise copies forward all
prior links *and* adds the entire new batch. Over 45 versions, evidence count
grows monotonically alongside text length. The carry-forward rule is correct in
isolation (it fixes WorkA's real data-loss bug) but combined with unbounded
revision it converges on "this claim is supported by everything."

**Recommendation:** cap `supports` links per proposition (keep the top-K by
quote specificity / IDF weight of the quoted span), and make `cooccurs` a
bounded fallback that is *dropped* once ≥ 3 `supports` links exist rather than
retained forever. A claim with 80 pieces of evidence is not better-evidenced
than one with 5; it is less precisely evidenced.

### S2-I ▸ Nothing maintains index coverage across supersession — and this is why the two answerable questions failed. — **CRITICAL**

`planS2.md` §5.2 mandates carrying evidence links and entity mentions forward on
supersede. It says **nothing** about the embedding index, and `planS3.md` §4
assumes "a precomputed embedding index for that kind" is current.

Measured:
- **27 of 64 propositions embedded.** Max embedded `ref_id` is 62.
- **Proposition 64 — the single largest live proposition, holding ~90% of the session's knowledge — has no embedding at all.** It was created at 22:00:11; the last embedding write was 22:00:01.
- **Zero entities are embedded**, so `planS2.md` §7.3's adopted entity-resolution-by-embedding has no index to run against.
- 85 of 89 observations embedded.

Now look at the `gaps` table. All 4 gaps are `status=open`, `hits=1`. Two of
them are **answerable from the store**:

| Gap | Asked | Store actually contains |
|---|---|---|
| #3 | *"what have I been working on today"* | prop 64 (the whole session) |
| #4 | *"what design tools have I used"* | prop 19 lists Adobe Fonts, MyFonts, dafont.com; prop 64 lists Illustrator, Grainient, Logosystem |

Neither has meaningful lexical overlap with the stored text — "design tools"
does not appear. They are *exactly* the class of plainly-worded question
`planS3.md` §5 introduces the dense-hit bypass for, and §5 uses **"what design
tools have I used" as its literal worked example**. The bypass could not fire,
because the dense index did not contain the rows that held the answer.

So: a retrieval failure was recorded as a knowledge gap, and the system
concluded it needed to ask the user for information it already had.

**Recommendation:**
- Make index coverage an **invariant with a test**: no live proposition, entity
  or observation may lack an embedding row; a startup + maintenance backfill
  job reconciles it; the store card reports coverage %.
- `supersede()` must enqueue the new row for embedding in the same transaction,
  the same way it carries evidence links.
- `planS3.md` §8 must distinguish **"store has nothing"** from **"retrieval
  found nothing"**. Before queuing a gap, run a cheap verification pass (e.g.
  entity-name lookup, or a full-table scan on a corpus this small) — and if it
  finds evidence, that is a retrieval bug to log, not a question to ask the
  user. As written, the gap mechanism silently converts recall bugs into user
  homework.

### S2-J ▸ Session summarization is degenerate at both ends. — **MEDIUM**

One summary row, covering **4 of 89 observations**, for a 70-minute period that
`SESSION_GAP_S=300` collapsed into 2 sessions. Its `apps` field reads
`"Arc, Terminal, logosesystem.co"` — propagating S1-C's typo — and its text
includes *"he encountered a note indicating that the dense index was empty"*,
i.e. the system summarizing its own error message as user activity.

Two independent problems:
1. **Coverage**: the maintenance pass evidently ran once early and never caught
   up. Same class as S2-I — no coverage invariant.
2. **Granularity**: a 5-minute-gap-bounded session is not a useful narrative
   unit. 70 minutes of continuous work is one row. "What did I do this morning"
   needs *topic-segmented* sub-sessions, not silence-bounded ones. This is the
   same "one row trying to represent too much" failure as the proposition blob,
   and it has the same fix: layered synthesis (S2-B), with segmentation driven
   by app/entity shift rather than by input silence.

---

## Stage 3 — Retrieval and Evaluation

Stage 3 is the best-reasoned of the three documents. Most of its findings are
**premise failures inherited from Stages 1–2**, not internal defects.

### S3-A ▸ Stage 3's tuning constants are being set against a corpus too small to support them. — **HIGH**

`COVERAGE_FLOOR = 0.12`, `COVERAGE_STRONG = 0.55`, `ECHO_THRESHOLD = 0.55`,
`DENSE_HIT_THRESHOLD = 0.5` are all presented as resolved defaults. They are
IDF-derived, and the entire corpus vocabulary is **1255 tokens over 89
observations**. At that size most content words sit at DF 1–3, so IDF is
near-maximal for nearly everything and coverage is dominated by sampling noise.

Worse, the highest-DF tokens are pipeline boilerplate, not user content:

```
user 89   owais 64   claim 54   specific 53   selecting 52   design 52
owais's 50   including 49   designs 48   evidence 48   exploring 48
supported 47   specifically 47   project 46   browsing 46   dooran' 44
```

`user` appears in **89 of 89** rows (the VLM caption template is "The user
is..."). `claim`, `evidence`, `supported` are the *insight pipeline's own prompt
vocabulary* leaking into `propositions.reasoning`, which is FTS-indexed at
weight 2.0. `dooran'` shows a tokenizer apostrophe artifact.

**Recommendation:**
- Exclude generated boilerplate from the DF table — or better, compute DF over
  `ocr_text` (S1-A) rather than over model-written prose. DF should measure the
  *world*, not the prompt.
- Treat the floors as **corpus-size-dependent**, not constants: below some row
  count, disable the coverage floor entirely and rely on dense + cross-encoder.
  There is currently no cold-start story, and every user's first week is
  cold-start.

### S3-B ▸ BM25 field weights are backwards for content questions. — **MEDIUM**

Inherited weights: `caption 8.0`, `title 5.0`, `url 4.0`, `app 3.0`,
`dense_text 1.0`. `planS3.md` §4 restates this as design ("an observation's
caption is weighted far higher than its raw dense text").

Given S1-A (dense_text averages 150 chars) and S1-C (`app` is typo-polluted),
metadata outweighs content 12:1 — and one of the metadata fields is actively
wrong. These weights are asserted, never measured. With `ranx` adopted (§10.2),
they are exactly the kind of thing that should be swept and gated, not frozen.

### S3-C ▸ §3.3's entity/proposition redundancy suppression is premature. — **LOW**

It is well-reasoned and the narrow scoping is right. But 86% of entity rows have
empty bodies (S2-F), so there is nothing to suppress against yet. Build it after
S2-F lands; measure the actual overlap rate before tuning
`ENTITY_PROP_OVERLAP_THRESHOLD = 0.6`.

### S3-D ▸ The §4.1 rejection of a vector DB is right, but §4.2's trigger is the wrong metric. — **LOW**

Declining Qdrant/miniCOIL/ColBERT at this scale is correct and well-argued. But
§13 flags the `sqlite-vec` trigger as undecided while framing it as a *latency*
question. At 200 chunks, latency will never be the trigger. The real one is
**index maintenance complexity** — S2-I shows the hand-rolled embedding table
already fails its coverage invariant at 200 rows. `sqlite-vec` is worth adopting
early not for speed but because it puts vectors under the same transaction as
the rows they describe, which is the actual defect. Reframe §4.2's trigger from
"scan is slow" to "coverage invariant is hard to maintain."

### S3-E ▸ The mandatory cross-encoder is the right call, with one caveat. — **LOW**

§6.1's reasoning is sound and the accepted local-weights tradeoff is stated
honestly. Caveat: on a corpus where ~90% of live propositions are one
1001-char run-on blob, a cross-encoder truncating at 512 tokens will score that
blob against every query on its first ~350 words. S2-A's length bound is a
prerequisite for the reranker to behave; note the dependency.

### S3-F ▸ The eval harness cannot be trusted until the store is stable — and `insights.md` already predicted this. — **HIGH**

`insights.md` §4 records that the golden set produced a **fake regression twice**
because it is tied to whatever is currently in the database. Nothing in
`planS3.md` §10 addresses that; it adopts `ranx` and statistical-significance
gating, which are both improvements, but sharper statistics on an unstable
substrate produce confident wrong answers faster.

`ranx` significance testing also needs a case count that the current golden sets
(15–23 cases, per `insights.md`) cannot support — a paired t-test / Fisher
randomization over 20 queries has very little power, so most real regressions
will read as "not significant" and pass the gate. That is a *worse* failure mode
than the old tolerance band, which at least failed loudly.

**Recommendation:**
- Version the golden set **against a store snapshot hash**. A case whose
  supporting rows are no longer present must be reported as `stale`, not as a
  miss. This single change removes the fake-regression class entirely.
- Target ≥ 100 cases before enabling significance gating; below that, keep the
  tolerance band and report the metric delta.
- Add two harness checks that directly target the defects above:
  **(a) answerable-gap detection** — for every gap the system queues, verify by
  brute force that the store truly lacks the answer (S2-I); **(b) calibration**
  — assert `confidence`/`decay` are not single-valued (S2-G).

---

## 2. Off-the-shelf substitutions, consolidated

Per the "don't reinvent what's already optimized" principle. Ordered by
value-per-effort.

| Concern | Today / planned | Recommended | Why |
|---|---|---|---|
| Near-duplicate frames | Trigram Jaccard over text incl. caption | **`imagehash`** (pHash) tier-0 + MinHash on stable text | Fixes the root of §1's chain. Duplicate *screens* are a pixel problem, solved before any text exists. |
| Entity resolution | embedding-only @ 0.85 (planS2 §7.3) | **`rapidfuzz`** cascade → embedding → LLM | The observed duplicates are typos and suffixes, which is rapidfuzz's exact job; embeddings both miss them and risk false merges on short brand names. |
| Entity extraction | inline in the drafting LLM call | **GLiNER** first pass | Local, free, typed, **and returns a confidence score** — the salience signal that's missing. planS2 §11 already flags this as open. |
| PII / secrets | *nothing* | **Presidio** (PII) + **`detect-secrets`** (keys/tokens) + app/URL denylist | S1-E. Non-negotiable for a screen recorder. |
| Domain normalization | none | **`tldextract`** | Collapses `logosytem.co`/`logosystem.co`/`Logosystem` in one line. |
| Vector storage | hand-rolled `embeddings` table + numpy scan | **`sqlite-vec`** | Not for speed — for transactional coupling, which is the actual failure (S2-I). |
| L2 / cross-session synthesis | *does not exist* | **`latticing`** (already in-repo, `pip install latticing`) | The structural answer to S2-A/S2-B. |
| Full capture tier | hand-rolled | evaluate **screenpipe** (already installed) | 24/7 screen + **audio** + OCR, local, REST API. Directly answers planS1 §12's open audio question. Worth a real build-vs-buy comparison before writing Stage 1 code. |
| Eval metrics | hand-rolled → `ranx` | **`ranx` ✅ (keep)** + snapshot-versioned cases | Adoption is right; the substrate problem (S3-F) is what needs fixing. |

Explicitly **agreeing** with the plans' rejections: Qdrant, miniCOIL, ColBERT,
and BEIR are all correctly declined for this scale and this acceptance
criterion. That reasoning in `planS3.md` §4.1/§10.3 is sound and should stand.

---

## 3. What to do, in order

**Tier 0 — nothing else matters until these land.** Every one is traceable to a
measured failure, not a hypothesis.

1. **S1-B + S1-A** — dedup on stable signal (OCR/AX, never caption), add pHash,
   store raw OCR. *Breaks the chain at its root.*
2. **S2-A** — N:M revision (restore GUM's split), hard length bound, lineage
   version ceiling. *Stops the runaway.*
3. **S2-I** — embedding-coverage invariant, backfill job, `supersede()` writes
   the vector transactionally. *Stops the store from silently losing its own
   answers.*
4. **S2-C** — fix the tz bug; add the regression test the spec's guarantee
   depends on.
5. **S1-E** — sensitive-content policy before any further capture.

**Tier 1 — these make the model of the user actually usable.**

6. **S2-B** — proposition `kind` (episodic/activity/preference/trait) with
   different merge rules per kind; L2 layer via `latticing`.
7. **S2-E + S2-F** — rapidfuzz cascade, drop `kind` from entity identity,
   recurrence gate, GLiNER, compute `mention_count`.
8. **S2-D** — entity mentions require a substring hit.
9. **S1-C + S1-D** — split `app`/`site`, AX-only `app`, abort on focus drift.

**Tier 2 — measurement and tuning, once the substrate is stable.**

10. **S3-F** — snapshot-versioned golden set, ≥100 cases before significance
    gating, answerable-gap and calibration checks.
11. **S2-G** — fix prompt-schema literals; recalibrate or derive
    confidence/decay; resolve the GUM decay inversion.
12. **S3-A + S3-B** — corpus-size-dependent floors; sweep BM25 weights with
    `ranx` instead of freezing them.
13. **S2-H** — cap evidence links per proposition.
14. **S1-F, S1-G, S3-C, S3-D** — close the open questions; defer §3.3 until
    dossiers exist; reframe the `sqlite-vec` trigger.

---

## 4. What is genuinely good and should not be touched

Stated plainly, because most of this document is criticism and the ratio would
otherwise be misleading.

- **Quote-verified evidence** (planS2 §6.2). Works exactly as specified —
  1380/1380 supports links carry a real quote, 810/810 cooccurs correctly carry
  none. Deterministic, cheap, catches a whole hallucination class. Keep.
- **Non-destructive bi-temporal supersession** (planS2 §5.2). The machinery is
  right and is ahead of mem0; it just needs typed rows (S2-B) to pay off.
- **Structured `RetrievalResult` and a grounding decision that is a pure
  function of counts, never re-parsed text** (planS3 §7). A real correctness
  property with a real incident behind it.
- **First-class unanswerable eval cases** (planS3 §10.1). The single most
  valuable idea in the whole lineage.
- **The dense-hit bypass on the relevance floor** (planS3 §5). Correct
  reasoning; it simply needs a dense index that exists (S2-I).
- **Declining Qdrant / miniCOIL / ColBERT / BEIR** (planS3 §4.1, §10.3).
  Correctly argued from this system's actual scale and acceptance criterion.
- **Triple-grounded enrichment with deterministic AX precedence** (planS1 §6.3).
  The right architecture; §6.2's "or website name" clause (S1-C) is the one
  thing undermining it.
