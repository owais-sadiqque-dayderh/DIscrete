# Insights — carried forward from WorkA / WorkAA / WorkAB

Plain-language notes on the concepts and mistakes from the last build, so
DIscrete doesn't have to relearn them by breaking again.

> **How this file is used (standing rule for the whole development phase).**
> Findings are routed by kind:
> - **Design / pipeline / plan-level issues** are fixed in `planS1.md`,
>   `planS2.md` or `planS3.md` directly, each with an entry in that plan's
>   *Revision log* stating what changed and why.
> - **Implementation bugs** (the design was right, the code was wrong) are
>   recorded here, with symptom, evidence, root cause and the rule that
>   prevents a repeat.
>
> Sections 1–5 describe mechanisms carried forward from the prototypes; where a
> plan revision later changed a mechanism, a note says so. Section 7 is the
> running bug log.

---

## 1. Non-destructive bi-temporal "supersede" instead of delete

**The problem it fixes.** WorkA's merge logic (`_handle_similar`) worked like
this: two claims look similar → LLM writes a merged version → **delete both
old rows** → rebuild the merged claim's evidence from a fresh query capped at
`limit=5`. Two things went wrong every single time this ran: the delete
cascaded and killed the evidence links pointing at those rows, and the
`limit=5` rebuild meant a claim that had 27 pieces of evidence came back with 5. Do this a few hundred times and you get exactly what the audit found: 85%
of all propositions ever created were gone, and there was no way to see what
a claim used to say.

**What it is.** "Bi-temporal" just means every row tracks two timestamps: when
it was true in the world, and when it was true _in the database_. Concretely,
a `Proposition` row never gets deleted. When a claim needs updating:

- the **old row stays in the table**, but gets `valid_to = now` and
  `superseded_by = <new row's id>` set on it (it's now "closed", not gone)
- a **new row is inserted** with the updated text, sharing the same
  `lineage_id` (a stable thread-id linking every version of "this claim over
  time") and `version = old.version + 1`
- **every** evidence link the old row had gets copied onto the new row (not
  re-queried, not capped — copied, so nothing is dropped), preferring a
  `supports` link over a weaker `cooccurs` link if both exist for the same
  observation

**Why this instead of just "don't delete, only insert new":** because you
still need to know which version is _current_ (`valid_to IS NULL`) without
scanning the whole lineage every time you retrieve, and you still want full
history queryable on demand (`workaa lineage <id>`).

**Pros:** nothing is ever silently lost; you can audit "what did we believe
about X, and when did that change"; retrieval always reads only current rows
so it's not slower.
**Cons:** the propositions table grows forever (every edit is a new row, old
ones just sit there) — needs a retention/archival story eventually, which
WorkAA never built. Also more code complexity than a plain UPDATE.

---

## 2. Quote-verified evidence links

**The problem it fixes.** WorkA linked evidence at the _batch_ level: if 5
screenshots got processed together into 2 claims, **every one of those 5
observations got linked to both claims**, regardless of whether a given
screenshot had anything to do with a given claim. A link meant "was in the
same batch," not "actually supports this." That's why asking "what supports
this claim about X" could return a screenshot of something unrelated
(spaceship.com, in one real case).

**What it is.** Two parts:

1. Each `EvidenceLink` row has a `relation` field — either `supports` (an LLM
   actually read the observation and the claim and confirmed a real
   connection) or `cooccurs` (weaker — just "these were near each other in
   time," used as a fallback when nothing has been verified yet, e.g. on
   import from an old dataset).
2. When the LLM claims "observation #12 supports this because it says
   `<quote>`," WorkAA **checks that the quoted text actually appears in
   observation #12's stored text** before saving the link. If the LLM
   hallucinated a quote that isn't really there, the code throws it away and
   computes a real snippet from the observation instead — it never trusts the
   model's citation blindly.

**Why:** an LLM can confidently invent a quote that sounds plausible but isn't
real. Verifying against the actual stored text is a cheap, deterministic check
that catches that class of hallucination completely — no second LLM call
needed, it's just a substring check.

**Pros:** provenance you can actually trust — "click into this claim" shows
you the real evidence, not batch noise; you can tell verified facts from
weak/imported ones and weight retrieval accordingly.
**Cons:** costs an extra LLM call per claim (`_map_evidence`) that batch-level
linking didn't need; imported data from the old system is honestly labeled
`cooccurs` forever unless you re-run `remap-evidence` (an explicit, separate
LLM pass) to upgrade it.

---

## 3. SimHash + trigram Jaccard dedup

> **Superseded in design by `planS2.md` Rev 1 (§2).** On the real WorkAA store
> this cascade almost never fired (84/89 rows `dup_count = 1`), because the key
> included the VLM caption, which is reworded run to run. That is a *design*
> flaw and is fixed in the plan: stable-signal key, perceptual-hash tier in
> Stage 1, MinHash+LSH. The explanation below is kept as history.

**The problem it fixes.** 93% of WorkA's observations shared an identical
300-character prefix — near-identical screenshots (same paused video, same
article scrolled slightly) each got stored as a brand-new full-size row. That
inflates the database, and worse, it inflates every word's document frequency
(DF), which is the denominator retrieval ranking depends on — with fake
duplicate counts, "rare" words stop looking rare and ranking degrades.

**What it is**, in order, on every new observation:

1. **Exact-hash check first** (SHA1 of the normalized text) — catches byte-–
   identical repeats instantly, cheapest possible check.
2. **SimHash prefilter** — a 64-bit fingerprint built from trigram shingles of
   the text, compared to recent observations via Hamming distance (how many
   bits differ). This is deliberately _loose_ — it's just a fast way to throw
   out obviously-unrelated candidates before doing the expensive check. Note:
   the newest 5 observations are _always_ compared regardless of the SimHash
   filter, because back-to-back screen captures are where almost all real
   duplication happens, and a loose-but-imperfect prefilter shouldn't be
   allowed to miss the most likely case.
3. **Trigram Jaccard on survivors** — the real decision. Break both texts into
   overlapping 3-word chunks (trigrams), measure `|intersection| / |union|`
   between the two sets. If ≥ 0.80 (the same threshold `tada` uses), it's a
   duplicate.
4. On a duplicate: **don't insert a new row** — bump a `dup_count` counter on
   the existing row and refresh its timestamp. The count itself becomes a
   confidence signal later ("this was seen 40 times" is stronger evidence than
   "seen once").

**Why two stages instead of just Jaccard on everything:** Jaccard requires
comparing full trigram sets, which is expensive if you check every new
observation against every old one. SimHash reduces that to comparing two
64-bit integers first, so the expensive check only runs on plausible
candidates.

**Pros:** collapsed 52.9% of the real corpus on import, directly fixed the DF
inflation; cheap to compute; a duplicate count becomes a useful signal instead
of noise.
**Cons:** 0.80 is a threshold tuned by hand against `tada`'s value — it's an
approximation, not a proof, and can under- or over-collapse on data that looks
different from what it was tuned on (an early test used one-line documents,
where the trigram set was too small for 0.80 to mean the same thing it does
on 4KB screen transcripts — a real bug this project already hit once).

---

## 4. The eval harness + `golden/cases.yaml` + regression gate

**The problem it fixes.** This is the single most important lesson from the
whole history. WorkA had **no way to measure whether a change helped or
hurt**. The only feedback loop was "type one question, eyeball the answer,
tweak the code." That loop is exactly how you end up with
`if "statement.pdf" in lower and "navier" in lower: score += 12.0` hardcoded
into the ranker — a fix that made _one_ test question look perfect while
silently making every other question worse, because nothing was checking the
other questions.

**What `golden/cases.yaml` is.** A hand-reviewed list of test questions
against your real store, each one of two kinds:

- **Answerable** — a question with a known correct answer, e.g. "what lean
  theorem prover was I using" → `expect_any: ["Lean", "theorem prover"]`. The
  harness checks whether those expected strings actually show up in what got
  retrieved/answered.
- **Unanswerable** — a question the store genuinely has no data for (a flight
  booking, a salary figure, a person who was never mentioned). These exist
  specifically to catch confabulation: **a system that makes something up for
  every unanswerable question is worse than one that correctly says "I don't
  know," even if the first one "sounds" more helpful.**

**What `workaa eval` measures**, per run:

- `hit@8` — did a correct answer appear anywhere in the top 8 retrieved
  results?
- `mrr` (mean reciprocal rank) — same idea, but weighted by _how high_ the
  correct hit ranked (1st place scores 1.0, 8th place scores 0.125)
- `precision` — of everything shown, what fraction was actually relevant?
- `mode_accuracy` — did the system correctly decide "I have a confident
  answer" vs "I only have weak evidence" vs "I have nothing"?
- `refusal_accuracy` — on the unanswerable cases specifically, did it
  correctly refuse instead of guessing? (This is the number that caught the
  original bug: it came back **0.0** on the very first real run — perfect
  retrieval, but zero honest refusals, because nothing enforced a relevance
  floor.)

**The regression gate.** `baseline.json` is a saved snapshot of these numbers
from a known-good run. Every future `workaa eval` run is diffed against it: if
any tracked metric drops by more than a small tolerance, the command **exits
with code 2** — a real failure signal a script or a pre-commit hook can act
on, not just a number to eyeball.

**Why this matters as a rule going forward:** "run eval before and after every
retrieval change" was rule #2 in WorkAA's own `plan.md`. It's cheap (~0.1s per
run, lexical-only) and it is the only thing standing between "I think this
change helped" and actually knowing.

**Pros:** turns retrieval quality into a number you can track over time
instead of a vibe; catches regressions immediately instead of days later; the
unanswerable cases specifically prevent optimizing for "sounds confident" over
"is honest."
**Cons — and these are real, not hypothetical, they already happened twice:**

- The golden set is tied to whatever's _currently in your database_. If the
  database resets or changes a lot (which happens often in dev), the golden
  set goes stale and reports a **fake regression** — this happened twice in
  the real history (WorkAA once, WorkAB once), both times producing a scary
  "hit@8 collapsed from 1.0 to 0.28" that was actually just "the test
  questions no longer match what's in the store," not a code problem.
- Regenerating the golden set (`--generate`) used an LLM to write new cases
  from the current store, and **overwrote `cases.yaml` in place** instead of
  writing to a separate file — once actually wiping out 5 hand-written
  unanswerable test cases with no backup, because the directory had no git
  history at the time to recover them from. (In DIscrete, this is now simply
  a "never overwrite golden files in place, always diff before replacing"
  rule enforced by real commits.)
- It's a small, self-authored sample (15–23 cases) written by reading the
  store, not blind — so the absolute numbers aren't meaningful on their own,
  only the _trend_ relative to the last baseline is.

---

## 5. AX (Accessibility API) — used in _two different places_, not just Bubble

> **Changed in design by `planS1.md` Rev 1 (§6.1).** The data layer now also
> reads the focused window's *text* through AX (with window-cropped OCR as a
> fallback). It no longer reads only app/URL/title metadata, and the VLM no
> longer transcribes. The table below describes the prototype.

You're not quite right that AX is only for Bubble. It's the same underlying
macOS API (`accessibility.py`), but it's called from two different places for
two different jobs:

**A. Data layer (`capture/screen.py`) — `window_context()`**
Every time a screenshot is taken, AX is asked one lightweight question: _which
app, which URL, which window title is this?_ This is metadata only — it does
**not** read the actual text content of whatever field you're in. It runs
alongside the screenshot+vision-caption pipeline, not instead of it. The
screenshot+VLM call is still what reads/describes on-screen _content_; AX just
tags that observation with exact, ground-truth app/URL/title instead of
whatever the vision model guesses from pixels.

**B. Bubble (`bubble/service.py`) — `read_focused_text()` / `capture_focus()`**
When you hit Option+Tab, Bubble needs to know _what's actually typed in the
field you're focused on, right now_. Originally (in WorkA) this meant a
screenshot + a 15–25 second vision call just to re-read text the OS already
has. WorkAA's fix: ask AX directly for `AXValue`/`AXSelectedText` — the
field's literal text content, no image involved, ~5ms instead of 15–25s. Falls
back to a screenshot+vision read only when an app hides its real text behind
its own custom rendering (Google Docs, Figma/Canvas-style apps don't expose
`AXValue` properly).

**So, corrected mental model:**
| | Data layer (screen.py) | Bubble (service.py) |
|---|---|---|
| What it asks AX for | app / URL / window title (metadata) | the actual text inside the focused field |
| Runs | on every screen capture | only when you press the hotkey |
| Replaces/supplements | tags the VLM-described observation | replaces a slow vision call with a fast direct read |

Both are real, both matter, and they solve two unrelated problems: one is
"label this observation correctly," the other is "read what the user is doing
right now, fast."

---

## 6. Process insights — why this file exists

The technical fixes above were all good. What actually broke the project
wasn't the architecture — it was how the _changes_ to the architecture were
made. In order, what happened and what it teaches:

1. **No git for four days of live-patching WorkAA.** Every real fix (the
   YouTube blind spot, an IntegrityError race, a stale eval set, the IDF floor
   rejecting plain-English questions) was a direct edit to the running
   codebase with no commit. When something broke and you asked for "the old
   WorkAA back," there was **nothing to revert to** — the "old good version"
   only existed in memory, not in any recoverable state. Git was only
   initialized _after_ this happened.
2. **A destructive regen (`--generate`) overwrote a hand-authored file in
   place** instead of writing to a new file for review — this is the kind of
   mistake version control exists specifically to make non-fatal, and it
   still cost real work because there was no history to fall back on.
3. **Two parallel, uncommitted variants existed at once** (WorkAA and WorkAB,
   the latter developed _inside_ WorkA's own directory, still showing as
   untracked/modified files in `git status` today). Without branches, there
   was no clean way to compare them, merge the good parts of each, or even
   answer "which one is current" without re-reading both by hand.
4. **The plan.md progress log stopped being updated for 4 days** while real
   fixes kept happening. When you asked "where did we diverge from the
   original plan," the honest answer was: nowhere, architecturally — but it
   _felt_ like drift because nothing recorded what changed, so there was no
   way to distinguish "intentional evolution" from "someone went off-script"
   without re-deriving it from scratch (which is exactly what that session had
   to do).
5. **A config value silently reverted** (`.env` model settings flipping back
   to an unreliable free-tier model after having been fixed in an earlier
   session) — another symptom of no durable, diffable record of what state was
   supposed to be true.

**The rule this sets for DIscrete:** every change is a commit, on a branch,
tied to a spec — not a live edit to whatever's running. `plan.md`-equivalent
docs get updated in the same commit as the code change they describe, not
"later." Destructive regenerations always go to a new file first. This isn't
about the architecture changing — the architecture was sound — it's about
making every change recoverable and legible after the fact, which is the one
thing that was actually missing.

**Addendum (2026-09-14 audit).** One more instance of the same pattern. The
revise-cooldown guard that `planS2.md` relied on was added to WorkAA *after*
the session whose runaway it was meant to stop (file mtime 22:30 UTC; last
proposition 22:00 UTC). It was never exercised, and it throws on every call
(§7.1). Rule added: **any mitigation a plan depends on ships with a test that
exercises it**. A plan may not describe a guard as working if no test has run
it.

---

## 7. Implementation bug log

Each entry: what happened, the evidence, the root cause, and the rule that
prevents it in DIscrete. Evidence comes from `~/.cache/workaa/workaa.db` and the
WorkAA source, as of the 2026-09-14 audit (details in `audit.md`).

### 7.1 Revise cooldown never executed (naive vs. aware datetimes)

- **Symptom:** one proposition lineage reached version 45 in 35 minutes,
  despite a 300 s revise cooldown in the code.
- **Evidence:** `utcnow()` returns `datetime.now(timezone.utc)` (aware).
  `Proposition.created_at` is `DateTime(timezone=True)` with
  `server_default=func.now()`, but SQLite stores no offset, so SQLAlchemy
  returns a naive datetime. Run against the real DB: `TypeError: can't subtract
  offset-naive and offset-aware datetimes`.
- **Root cause:** mixed naive and aware datetimes, with no single conversion
  point.
- **Rule:** every timestamp is timezone-aware UTC in Python. A SQLAlchemy
  `TypeDecorator` attaches UTC on read and converts on write. No bare
  `datetime.now()` or `func.now()` comparisons. A test asserts that a time-based
  guard actually suppresses the second action.

### 7.2 A broad per-item `except Exception` hid 7.1

- **Symptom:** the cooldown's `TypeError` produced a log warning, and the
  pipeline carried on as if nothing were wrong.
- **Root cause:** `except Exception` around the entire per-draft body, which
  counted one error string and continued.
- **Rule:** catch narrow, expected exceptions only (network, parse). Anything
  else increments a per-type error counter that is shown on the store card, and
  the test suite asserts **zero** unexpected errors in a full pipeline run over
  a fixture.

### 7.3 Entity identity included the LLM-guessed `kind`

- **Symptom:** 12 entities existed twice, including the user
  (`Owais`×2, `Workaa`×2, `logosystem.co`×2, `MyFonts`×2, `Adobe Illustrator`×2, …),
  each with its own mention history.
- **Root cause:** `UNIQUE(norm_name, kind)`, and `upsert_entity` looked up on
  `(norm_name, kind)`. The model assigned `kind` inconsistently (`topic` vs
  `tool` vs `organisation`).
- **Rule:** identity keys never include model-guessed attributes. Identity is
  the entity id plus a unique normalized-alias table (`planS2.md` §10.4).

### 7.4 Alias matching by substring `ILIKE` over a concatenated blob

- **Symptom:** risk of arbitrary merges. Short entity names (`art`, `ai`,
  `after`, `chart`) match any entity whose alias blob contains them (`art` ⊂
  `Artboard 2`).
- **Root cause:** `Entity.aliases.ilike(f"%{norm}%")` over a single text
  column.
- **Rule:** exact match on a normalized `entity_aliases` table. Fuzzy matching
  only through the explicit, thresholded cascade.

### 7.5 Prompt JSON examples with literal values were copied by the model

- **Symptom:** 60/64 propositions had `confidence = 8.0`; 56/64 had
  `decay = 4.0`.
- **Root cause:** the prompt's output example was
  `{"confidence": 8, "decay": 4, …}`.
- **Rule:** schema examples use type placeholders (`<int 1-10>`, `<string>`),
  never literal sample values. A test renders each prompt and fails on numeric
  literals inside the output-schema block. (The design change of no longer
  asking the model for confidence at all is in `planS2.md` §8.)

### 7.6 Denormalized `mention_count` drifted on every row

- **Symptom:** `Owais` 85 vs. 135 actual mentions, `Dooran` 63 vs. 103; every
  entity undercounted. The dossier gate read the wrong number.
- **Root cause:** a counter incremented on some code paths and not others.
- **Rule:** derived counts are computed from the source table (query or view),
  never stored as independent counters unless a reconciliation job and test
  keep them equal.

### 7.7 Superseded rows were never embedded; the embedding backlog stalled

- **Symptom:** 27/64 propositions and 0/123 entities had embeddings. The
  largest live proposition (id 64) had none. Two answerable questions were
  recorded as knowledge gaps.
- **Root cause:** embeddings were written by a separate maintenance pass that
  `supersede()` never enqueued for, and nothing reported the backlog.
- **Rule:** every write that creates a live row enqueues or writes its
  embedding in the same transaction. The design-level invariant is
  `planS2.md` §11. A test asserts 100 % coverage after a pipeline run.

### 7.8 Session summarization ran once and never caught up

- **Symptom:** 1 summary row covering 4 of 89 observations.
- **Root cause:** `sessions_without_summary(limit=5)` on a maintenance cadence,
  with no backlog visibility and no catch-up guarantee.
- **Rule:** every periodic job reports its backlog on the store card, and a
  test drives the job to empty over a fixture.

### 7.9 Model output overrode AX ground truth for `app` / `url`

- **Symptom:** `app` values full of model spellings (`logosesystem.co`,
  `Graident`, `logofoilo.com`).
- **Root cause:** `screen.py` took the VLM's `app` / `url` first and fell back
  to AX (`data.get("app") or ctx.get("app")`). This is the reverse of the
  intended precedence.
- **Rule:** ground-truth precedence lives in one function with table-driven
  tests. (In the revised design the model does not produce these fields at all:
  `planS1.md` §6.2–6.3.)

### 7.10 Tokenizer kept trailing apostrophes

- **Symptom:** `token_df` contained `dooran'` (df 44) as a separate token from
  `dooran`.
- **Root cause:** the tokenizer split on whitespace and punctuation but did not
  strip possessive or trailing quote characters.
- **Rule:** the tokenizer strips possessives (`'s`, `’s`) and leading or
  trailing quotes. Unit tests cover straight and curly quotes.

---

*Entries 7.11–7.26 come from the review of the v1 build (commit `a235668`,
generated from the plans in another tool) and its first live run, 2026-09-14.
Each entry is pinned by a test in `tests/test_pipeline.py` where one is
feasible.*

### 7.11 One missing pyobjc constant disabled all of Accessibility

- **Symptom:** `app`, `window_title`, `url` and `site` were empty on 17/17
  observations of the first live run.
- **Evidence:** `from ApplicationServices import kAXManualAccessibilityAttribute, …`
  raises `ImportError` (pyobjc does not export it). The module-level
  `except ImportError: _HAS_AX = False` then made every AX function return
  empty, silently.
- **Root cause:** a batch import guarded by a single capability flag. An
  optional attribute took down a mandatory capability.
- **Rule:** AX attribute names are plain strings. Window identity comes from the
  Quartz window list, which needs neither AX nor a run loop. `doctor` prints the
  live window context, and a test asserts that it resolves.

### 7.12 OCR ran in *fast* mode; `text_source` was mislabeled

- **Symptom:** garbled text on every capture ("Ed￿* View Swes Tab$ Archfve",
  "news.ycofflbinalor.com").
- **Evidence:** `setRecognitionLevel_(1)  # accurate`. But
  `VNRequestTextRecognitionLevelAccurate = 0` and `Fast = 1`. The same frame
  read cleanly at level 0 with language correction (1.2 s vs 0.24 s). Rows with
  no AX text were also labeled `ax+ocr`.
- **Rule:** use named constants and never trust comments on magic numbers. A test
  OCRs a rendered string.

### 7.13 Frames were full-screen; the window crop was never implemented

- **Symptom:** OCR and captions included the menu bar, the dock and the
  overlapping terminal that was running the daemon (observations 1, 2 and 5
  summarized its own log).
- **Root cause:** `planS1.md` §5/§6.1 specify a crop to the focused window. The
  frame loop grabbed `monitors[1]`.
- **Rule:** `capture/frames.py` grabs the window's bounds on its own display.
  Self-windows are classified by title (`planS1.md` Rev 3.4).

### 7.14 Interaction counts were wrong in three ways

- **Symptom:** `clicks` up to 3652 per capture, `keys = 0` on keyboard
  captures, `clipboard` always 0. Every observation was therefore "engaged",
  and the modality signal carried nothing.
- **Root causes:**
  - Mouse *moves* were counted as clicks and armed captures.
  - Counts were snapshotted when the capture armed, not over its interval.
  - Clipboard triggers never incremented a counter.
  - A drift abort skipped the counter reset.
- **Rule:** counting happens on the event-loop thread, inside the pending
  capture it describes. Moves are neither triggers nor interaction.

### 7.15 Idle transitions captured the screen

- **Symptom:** the user going away produced a capture of an unattended screen.
- **Root cause:** the idle poller fired an ordinary trigger. `planS1.md` §2.7
  says idle affects sessions only.
- **Rule:** idle has its own callback. It rotates the session, closes the
  episode, and suspends heartbeats.

### 7.16 Episodes were never drafted

- **Symptom:** 2/2 episodes of the first run stayed `open`, and 0 claims
  existed.
- **Root causes:**
  - Tracker state lived only in memory, nothing closed the current episode on
    shutdown, and `discrete process` handled only `ready`.
  - The switch rule closed on the *first* observation of a new app whenever the
    old app had lasted ≥ 90 s, and assigned that observation to the old episode.
  - Each observation was embedded twice, double-weighting the centroid.
  - Idle close stamped `ended_at = now`, adding 300 s to every episode.
  - Drafting ran inline in the ingest loop and in the draft loop, with no guard
    against doing it twice.
- **Rule:** an explicit lifecycle with an atomic `ready → drafting` claim
  (`planS2.md` Rev 3.1). Tests cover alt-tab absorption, the retroactive split
  and orphan recovery.

### 7.17 Quote verification could be bypassed

- **Symptom (code):** a claim with no `quote_obs_id` was stored with no
  evidence at all. The fallback check compared an observation's text against
  its *own* tokens, which says nothing about the claim.
- **Rule:** a claim needs ≥ 1 verified evidence link, or it is dropped. A
  substitute quote is the best window *per field* for the claim's content
  words, so it never spans two fields.

### 7.18 Batch-level mentions came back (a repeat of §2)

- **Symptom (code):**
  - `_attach_entities` extracted names from every observation in the episode and
    linked them to every claim.
  - It created entities immediately, with no recurrence gate.
  - It never checked that the name occurred in the claim.
  - Every capitalized OCR word became a candidate, at one embedding call each.
- **Rule:** a mention requires the alias in the claim text or a verified quote.
  Entities come only from promotion (≥ 2 episodes). Promotion back-fills
  mentions. A test pins all three.

### 7.19 Confidence was off-scale and reset on reinforcement

- **Symptom (code):**
  - `create_claim` defaulted to `len(evidence) × 0.5`, capped at 10.
  - Reinforcement *replaced* confidence with a one-episode value.
  - Superseding set 0.5 or 0.3.
  - The formula summed *unique* modalities rather than one term per episode.
- **Rule:** confidence is recomputed from evidence rows inside the transaction
  on every evidence change. A test asserts it rises monotonically across
  reinforcements.

### 7.20 Consolidation kept only the first output

- **Symptom (code):** `apply_consolidation` looped over outputs and `break`-ed
  after the first. It superseded every cluster member, so splits were lost. No
  output validation ran.
- **Rule:** one transaction. Each output inherits evidence from its own inputs,
  inputs no output uses stay live, and a call that fails validation changes
  nothing. A test pins it.

### 7.21 Retention deleted rows, and evidence cascaded away

- **Symptom (code):** `purge_old_observations` deleted observations older than
  90 days. `ON DELETE CASCADE` removed their evidence, and the
  unsupported-claim lint then superseded the claims. Retention would have erased
  the memory it was retaining.
- **Root cause:** `planS2.md` §12 says *clear* `screen_text`. The code deleted
  the row.
- **Rule:** retention blanks text and frames only, and decrements DF. A test
  asserts that evidence survives.

### 7.22 The outbox never embedded episodes or entities

- **Symptom:** an episode outbox row sat forever, and coverage never reached 100%.
- **Root cause:** the text builder handled only observations and claims.
  Failures were not counted, and dead references were never discarded.
- **Rule:** all four kinds are handled, dead or non-live references are
  discarded, and failures count toward `OUTBOX_MAX_ATTEMPTS`. A test asserts
  coverage = 1.0 for every kind.

### 7.23 Retrieval ranking was mostly unranked

- **Symptom (code):**
  - Dense candidates were kept in insertion order, not by similarity.
  - `Hit.score` was never set, so recency and MMR multiplied zero.
  - `_filter_interval` returned its input.
  - BM25 field weights were not applied.
  - Claim materialization reloaded all live claims per hit (O(N²)).
  - `brute_verify` could never mark a hit strong, so every thin query became a
    "knowledge gap".
- **Rule:** a cached normalized matrix per kind with argsort, fused RRF as the
  base score, field-weighted `bm25()`, and batch materialization. Tests cover
  dense ranking and an end-to-end retrieve.

### 7.24 The query planner trusted the model and dateparser

- **Symptom:** on the live store, three of four real questions returned
  **empty**. The utility model returned `needs_store: false` and free-text
  intents, and `dateparser` turned "on hacker news" into a date six days back.
- **Rule:** tests pin both. The design change is `planS3.md` Rev 3.1–3.2.

### 7.25 A junk digest was generated at startup

- **Symptom:** today's digest read "Sure! Please provide the details or
  summaries of the episodes…".
- **Root cause:** consolidation ran immediately on start. The digest prompt was
  built from zero episode summaries, and the stored row then blocked the day.
- **Rule:** consolidation runs on its persisted interval, no digest is written
  without summaries, and digests track the episode count (`planS2.md` Rev 3.2).

### 7.26 The tests passed while nothing worked end to end

- **Symptom:** 9/9 tests passed on a build that stored no claims, no entities,
  and no app names.
- **Root cause:** the tests exercised helpers (hashing, a store round-trip),
  never the pipeline. There was no fake model backend. The eval set was 3
  unanswerable cases compared against an empty baseline.
- **Rule:** every guard a plan relies on has a behavioural test over a fake
  backend (`tests/conftest.py`). A stage counts as done only after a real-data
  run: `discrete replay-workaa` plus `discrete eval --cases
  golden/workaa_replay.yaml`, and `discrete doctor` plus a short live `run`.
  This extends §6's addendum: tests must exercise the *pipeline*, not only its
  parts.


### 7.27 UI chrome became entities

- **Symptom:** a 4-observation live run promoted 18 entities, among them
  "Dashboard\nHome", "Projects\nFilter", "Unnamed Unnamed" and
  "Users/owaissadiqque".
- **Root cause:** the Title-Case pattern matched `\s+`, so spans crossed line
  breaks between separate AX labels. An app's sidebar and menu labels recur in
  every episode, so they pass the ≥2-episode recurrence gate by construction.
  Filesystem path fragments matched the repo pattern.
- **Rule:** the structural pass works line by line and keeps only
  identifier-shaped patterns (inner caps, files, domains, handles, repos).
  Multi-word proper names come from the VLM and claim sources. `promotable`
  rejects multi-line names, repeated-token echoes, and paths under
  filesystem roots.

### 7.28 An agent chat on screen was modelled as the user's own work

- **Symptom:** the live run's claims read "The user is improving OCR accuracy
  with language correction" and similar. That was the coding agent's
  transcript in Conductor, which the user was only watching.
- **Root cause:** `classify_origin` recognises DIscrete's own output, not
  other agents' output. The claims were tagged `passive`, which is correct but
  does not stop them from being retrieved as the user's activity.
- **Rule (open):** see `planS1.md` open items. Until an agent-transcript
  classifier exists, `SELF_APPS` / `SELF_WINDOW_PATTERNS` let a user exclude
  agent chat windows, and retrieval surfaces modality so the writer can say
  "viewed".

### 7.29 A new TLS handshake on every rerank

- **Symptom:** each rerank call cost roughly 150–300 ms of connection setup.
- **Root cause:** `OpenAIBackend.rerank` opened a fresh `httpx.AsyncClient`
  per call.
- **Rule:** keep a persistent client per backend. Latency-critical paths also
  never await independent API calls in sequence: the query plan and the query
  embedding now run concurrently.

### 7.30 PyObjC turned the overlay's Python helpers into selectors

- **Symptom:** the overlay process died at import with `BadPrototypeError:
  '_layout' expects 0 arguments`.
- **Root cause:** every method defined on an `NSObject` subclass is converted
  to an Objective-C selector, and a name without a trailing underscore
  declares zero arguments.
- **Rule:** UI state lives in a plain Python class. Only the minimal
  callback target (the `tick:` timer target) subclasses `NSObject`.

### 7.31 The caret AXValue came back as a tuple, not a struct

- **Symptom:** every Bubble hotkey press crashed the event-tap callback with
  `AttributeError: 'tuple' object has no attribute 'location'` in
  `_caret_rect` → `capture_focus`, so no snapshot was ever taken.
- **Root cause:** `AXValueGetValue(rng, kAXValueCFRangeType, None)` bridges the
  `CFRange` as a plain `(location, length)` tuple on this pyobjc version, not
  an object with `.location`. The code assumed the object form; the same
  assumption sat on the `CGPoint`/`CGSize`/`CGRect` reads.
- **Rule:** never touch a bridged AXValue struct field directly. Normalize
  through `_range_location` / `_point_xy` / `_size_wh` / `_rect_xywh`, which
  accept both the object and (nested) tuple forms and return `None` on
  anything malformed, so a hotkey press can never crash the tap.

### 7.32 A browse of many pages collapsed into one before/after

- **Symptom:** opening a burst of Hacker News articles produced a single
  observation for the whole session, not one per article.
- **Root cause:** every trigger — clicks, scrolls, *and* each `window_change`
  — funnelled into one shared `_pending` slot and one debounce timer, and each
  new trigger cancelled and rescheduled that timer. While the user kept moving
  faster than the debounce, the pair never flushed; the first page's `before`
  and the last page's `after` were finally stored as one "moment." The debounce
  *interval* was not the cause — the single-slot coalescing model was.
- **Rule:** a navigation to a different window/app is a capture *boundary*, not
  another debounce reset. On `app_switch`/`window_change` to a different key,
  `_rotate` finalizes the article in flight immediately and a fresh pair opens
  on the new window (planS1 §5, Rev 4). This generalises the focus-drift guard
  from "abort at flush" to "segment at the boundary." A rotated capture reads
  its text from the frame it froze (OCR), because the live AX tree is already
  the *next* page — same-process tab navigation cannot be read retroactively.

### 7.33 Browser pages had no identity

- **Symptom:** most Arc rows were titled "Space 1" with no URL. Navigating
  between pages inside a tab triggered no capture at all.
- **Root cause:** Arc titles its windows by Space, not by page. The poller's
  change key was (pid, window title), so in-tab navigation was invisible to
  it. URLs came from AppleScript, which takes 0.25 s per call in Arc and
  3.7 s in Chrome, and returns nothing for a stale capture.
- **Rule:** page identity comes from the tab's `AXWebArea` (`AXURL`,
  `AXTitle`). The lookup is cached, takes about 3 ms per poll, and is re-found
  when the title changes or every 2 s. The change key is (pid, title, URL).
  AppleScript is only a fallback.

### 7.34 "What did I open from X" cannot be answered by matching

- **Symptom:** "list all the articles I opened in hacker news" returned the
  Hacker News listing pages themselves.
- **Root cause:** a page opened from a listing never mentions the listing.
  Top-k and keyword matching can only find the listing.
- **Rule:** list mode returns a time-ordered trail. It holds the pages that
  match the subject (★), plus the pages visited in the same app within
  `ENUMERATE_TRAIL_S` after a match (→). Visits collapse by URL, or by title
  when the URL is missing. The writer decides which pages answer the question.

### 7.35 A zero-width field became a query

- **Symptom:** in Google Docs, Bubble typed a paraphrase of random captions.
  The log showed `query="​"`.
- **Root cause:** Docs renders its text on a canvas, and its AX value is a
  zero-width space. `str.strip()` keeps U+200B, so the field looked non-empty.
  An empty query then retrieved arbitrary records, and the writer paraphrased
  them.
- **Rule:** invisible characters are stripped from field text. A field
  without words falls back to the vision read. The engine raises `EmptyField`
  instead of retrieving on nothing.

### 7.36 Query scaffolding and captions matched everything

- **Symptom:** list-mode results were full of unrelated rows (a todo "list"),
  and every page was marked as the source.
- **Root cause:** the request's own words ("list", "opened", "articles",
  "below") were searched as if they were the subject. VLM captions also
  mention the previous page ("switched from Hacker News to…").
- **Rule:** scaffolding words are stripped before search, and the stripped
  subject is embedded. A named source matches on page identity (title, site,
  URL), scored per query string. Content matches are used only when no
  identity matches anywhere.

### 7.37 Rotated captures captioned the wrong page

- **Symptom:** the "Blueprint | Founders, Inc." row was captioned as a Google
  Doc, and another row as a Finder folder.
- **Root cause:** a rotated capture took the latest cached frame. The poller
  notices a navigation up to `POLL_S` after the frame cache already holds the
  next page. The "before" frame at arm time also showed the page just left.
- **Rule:** keep a small ring of recent (frame, context) pairs and use the
  newest frame whose window key matches. Drop the capture rather than caption
  another window's frame. The caption prompt describes only the last
  screenshot, and names the page.

### 7.38 List answers were summarized prose, not article names

- **Symptom:** "list all the articles I opened in hacker news" produced a
  paragraph about Arc, OpenRouter and Finder — or listed "Hacker News" three
  times — instead of article titles like WorkAA would.
- **Root cause:** three compounding gaps, all on the read path:
  1. Arc titles in-tab pages "Space 1", so enumerate dedup collapsed every
     article into one row and the writer only saw generic window names.
  2. The writer's `grounded` instruction asks for "the most specific facts" as
     prose; even with an `enumerate` mode, the LLM still merged/summarized.
  3. Listing pages (HN index) were returned as items instead of the trail
     pages opened after them; captions held the real names ("Linux From
     Scratch", "frank-386") but were never used for labels.
- **Rule:** `page_label` / `page_identity` in `textkit.py` derive display names
  from captions and URLs when the window title is generic. `shape_enumerate_items`
  keeps only navigation-away rows for opened-item questions and drops listing
  index pages. `format_enumerate_answer` renders the list deterministically —
  no LLM on the enumerate write path.
