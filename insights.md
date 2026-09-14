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

