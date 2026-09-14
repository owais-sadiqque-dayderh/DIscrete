# Stage 3 — Retrieval and Evaluation: Design Plan

**Status: design record, revised in place with a visible revision log.**
This file is self-contained. Design changes are made directly in this file and
recorded in the Revision log (what changed, why, and the evidence).
Implementation bugs belong in `insights.md`. Sections changed by a revision
carry a `[Rev N]` marker. Upstream stages: `planS1.md` (capture) and
`planS2.md` (processing).

---

## Revision log

### Rev 1 — 2026-09-14 (architecture audit against the real WorkAA store)

Evidence: the real WorkAA store (see `audit.md`), especially its `gaps` table.
All 4 logged "knowledge gaps" were still open. Two of them (*"what have I been
working on today"*, *"what design tools have I used"*) were **answerable from
the store**; the dense index simply lacked the rows holding the answer.
External evidence: memory benchmarks (LongMemEval, LoCoMo), a controlled
verbatim-versus-extracted memory ablation (arXiv 2601.00821), and the
retrieval designs of Zep, Hindsight and tada.

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 3.1 | §3 | **Verbatim content is a first-class retrieval target, not a fallback.** The record kinds are entities, claims, **episodes** (which replace sessions) and observations. Episodes and observations are always searched; session gating is removed. | In a controlled ablation, verbatim memory beat LLM-extracted artifacts by 15.9 points on LoCoMo and 22.0 on LongMemEval-S. Extracted artifacts helped only *alongside* verbatim text, never instead of it. tada's predictor retrieves over raw captions. Stage 2 Rev 1 also made `screen_text` verbatim and kept it. |
| 3.2 | §2 | **Real temporal understanding.** Time expressions parse to explicit intervals, replacing a 5-value `time_hint` enum. Two intents are added: `temporal` and `aggregate`. | Temporal reasoning is one of LongMemEval's five core abilities, and the category with the largest memory-system gains. "Last Tuesday afternoon" cannot be expressed as `this_week`. Hindsight runs a dedicated temporal retrieval channel. |
| 3.3 | §4.4 | **Bounded time-window queries switch to exhaustive mode.** Every episode in the interval is returned, ordered by time; there is no top-k. | "What did I do this morning" is a *listing* question. Top-k retrieval structurally drops parts of the answer. A day's episode summaries fit easily in context. |
| 3.4 | §4.5 | **`aggregate` queries route to a structured lookup** over entities by kind and claims in a time window, before text search. | "What design tools have I used" is an aggregation over entity kind `software`. Text search for "design tools" matched nothing in the prototype, because no stored text contained that phrase. |
| 3.5 | §5 | **Term statistics are computed over world text only** (`screen_text`, captions, titles), never over model-written reasoning. The coverage floor is disabled on a cold-start corpus. | The prototype's top document-frequency tokens were pipeline vocabulary: `user` (89/89 rows), `claim`, `evidence`, `supported`. The corpus had 1255 tokens in total, too small for IDF floors to be anything but noise. |
| 3.6 | §5.3 | **Recency prior per kind** (claim half-lives from Stage 2; no prior for facts about fixed things), and **MMR diversity** at assembly. | tada's retriever applies exponential time decay plus MMR; Zep ships an MMR reranker. Near-duplicate hits of one activity otherwise fill top-k. |
| 3.7 | §6 | **Concrete reranker candidates**, chosen by the harness. Reranker input is bounded (claim or summary plus its best snippet). | The mandatory cross-encoder decision stands. Its behaviour depended on Stage 2 no longer producing 1000-character claims. |
| 3.8 | §8 | **Verify before queuing a gap.** A retrieval miss is distinguished from genuine store absence, and every miss becomes a regression case automatically. | The prototype turned its own recall failures into questions for the user. |
| 3.9 | §7.1 | **Retrieved content is returned marked as untrusted data, with claim modality attached.** | Memory poisoning: stored text originated from arbitrary web pages. The product layer must never execute instructions found in memory, and must know "viewed" from "did". |
| 3.10 | §10 | **Evaluation gains a question taxonomy, snapshot-versioned golden sets, a minimum case count before significance gating, baselines, and system-health checks.** | The golden set produced fake regressions twice when the store changed. A significance test over 15–23 cases has little power, so real regressions pass as "not significant". Memory benchmarks compare against full-context baselines. |
| 3.11 | §4.2 | **The `sqlite-vec` trigger is reframed** from query latency to transactional coupling, with its pre-v1 status noted. | At prototype scale latency is never the problem. Index drift was (27/64 claims embedded). |
| 3.12 | §4.1 | **BM25 field weights re-baselined** with verbatim text weighted up. The weights are a sweep target, not frozen constants. | Prior weights gave metadata a 12:1 advantage over content, and one metadata field (`app`) was typo-polluted. |

### Rev 2 — 2026-09-14 (user decisions before implementation)

| # | Section | Change | Why |
|---|---|---|---|
| 2.1 | §6.1 | **Rerank stays mandatory but is API-served.** The cascade is OpenRouter's `/rerank` endpoint (Cohere Rerank family) → LLM listwise through the utility tier → fused order. The Rev 1 local cross-encoder is removed. | User decision: no locally hosted models. OpenRouter serves Cohere Rerank models through one `/rerank` endpoint. The accepted cost is per-query API cost and network latency. |
| 2.2 | §2, §13 | The on-device utility tier option (Apple Foundation Models / MLX) is removed. The utility tier is an API model. | Same decision. |
| 2.3 | §4.1 | Query embeddings use the same embedding API model as Stage 2. | Same decision (`planS2.md` Rev 2). |

---

## 1. Purpose and scope

Stage 3 turns a raw query (typed text, a field's current contents, or a
question) into:
1. a structured, ranked set of hits drawn from the store's record kinds;
2. a decision about what posture is safe to answer from (confidently,
   cautiously, or not at all);
3. a measurement of whether (1) and (2) work, so every change is judged by
   numbers.

Stage 3 does not write the answer or drive the hotkey, tool-calling and
insertion loop; that is the product layer. Its output is a structured
`RetrievalResult` that nothing downstream ever re-parses from rendered text.
Re-parsing rendered text is a proven failure mode: a format change once zeroed
every hit count while real evidence was on screen.

### Pipeline  `[Rev 1]`

```
Raw query
   │
   ▼
[1] Query understanding → QueryPlan (intent, keyword queries, entities, time interval)
   │
   ├── intent = aggregate ─────────► [A] Structured lookup (entities by kind, claims in window)
   ├── bounded interval ≤ 1 day ───► [T] Exhaustive episode listing for the interval
   │
   ▼
[2] Recall per kind: BM25 + dense (+ temporal filter, + optional 1-hop entity expansion) → RRF
   │
   ▼
[3] Relevance: echo suppression, IDF coverage (world text), relevance floor (off at cold start)
   │
   ▼
[4] Assembly: recency prior, claim↔evidence pairing, entity/claim redundancy suppression, MMR
   │
   ▼
[5] Mandatory rerank (rerank API → LLM listwise → fused order)
   │
   ▼
[6] Grounding decision (pure function of counts) → edit / grounded / thin / empty / continuation
   │
   ▼
[7] On thin/empty: verification pass → store_lacks (queue gap) | retrieval_miss (log + eval case)
   │
   ▼
Structured RetrievalResult (untrusted-marked) + GroundingDecision → product layer
```

---

## 2. Query understanding  `[Rev 1]`

A `QueryPlan` has these fields:
- `raw`;
- `intent`: `edit` / `recall` / `temporal` / `aggregate` / `world` /
  `continuation` / `mixed`;
- `queries`: 2–10 short keyword strings;
- `entities`;
- `interval`: an explicit `[start, end)` in local time, or null;
- `entity_kind_filter`: for aggregate queries;
- `needs_store`.

**Two sources, unioned (retained):**
1. **Structural pass**, with no model call:
   - Capitalized spans, quoted spans, file-ish tokens, dotted identifiers and
     internal-capital words become entity candidates.
   - Edit-verb phrases with no question mark mean `edit`.
   - Long text with no question and no interrogative means `continuation`.
   - `[Rev 1]` Time expressions are parsed into an interval by a
     temporal-expression parser (`dateparser`, with its search mode for
     embedded expressions; `ctparse` as an alternative). Examples: "this
     morning", "last Tuesday", "two weeks ago", "since Monday", "in August". A
     parsed interval sets `intent = temporal` unless another intent is stronger.
   - `[Rev 1]` Aggregation cues ("which/what … have I used", "list", "all the",
     "how many") set `intent = aggregate`. The noun they govern is matched
     against the entity kind vocabulary (`software`, `person`, `project`,
     `website`, …) to fill `entity_kind_filter`.
2. **LLM plan**: one call to the utility model tier, returning the same shape.
   Its output is unioned with the structural plan; any failure falls back to
   the structural plan alone.

No hardcoded synonym tables (retained): hand-written expansions only cover
phrasings someone already anticipated.

---

## 3. Record kinds  `[Rev 1]`

| Kind | What it is | Role |
|---|---|---|
| **Episodes** | A coherent stretch of one activity, with title, summary and time interval (Stage 2 §3). | The primary narrative unit. Answers "what was I doing", and anchors temporal questions. |
| **Observations** | Verbatim `screen_text` plus caption, app, site, title and time. | Primary evidence. Searched directly, not only as a fallback; the verbatim layer is what extracted memory loses. |
| **Claims** | Typed (`event` / `fact` / `preference`), bounded, evidence-linked (Stage 2 §4). | Precise answers to fact- and preference-shaped questions. Carry derived confidence and modality. |
| **Entities** | Resolved entities with pages (Stage 2 §10). | Thing-shaped questions, and aggregation (§4.5). |
| **Digests** | One narrative per day (Stage 2 §7.4). | "What did I do yesterday / last week". Used in exhaustive mode (§4.4). |

All kinds are searched on every recall-type call; the prior session gating is
removed. Episodes are cheap (a few dozen per day), and verbatim observations
are the kind evidence says must not be demoted.

**Claim↔evidence pairing:** a claim hit is assembled together with its best
verified quote, as one unit. An observation hit already cited by an included
claim is folded into that claim's unit instead of being listed separately.

**Entity/claim redundancy suppression (retained, deferred):** a claim hit whose
trigram-Jaccard against an included entity page body is ≥
`ENTITY_PROP_OVERLAP_THRESHOLD` (default **0.6**) is dropped. It is enabled
only once ≥ 50 % of entities with ≥ 3 mentions have pages. Until then there is
nothing for it to act on: in the prototype, 86 % of entities had no page.

---

## 4. Recall  `[Rev 1]`

### 4.1 Channels and fusion

Each kind runs three kinds of ranked list, all fused with **Reciprocal Rank
Fusion** (`RRF_K = 60`):
- one **field-weighted BM25** list per plan query string (SQLite FTS5);
- one **dense** list from the query embedding, using the same embedding API
  model as Stage 2 §11;
- `[Rev 1]` when `interval` is set, candidates outside it are filtered out
  (episodes, observations, event claims), or down-weighted by distance for
  facts and preferences whose `occurred_from..occurred_to` does not intersect
  the interval.

Initial BM25 field weights `[Rev 1]` (sweep targets, §10.6):

| Kind | Field weights |
|---|---|
| Observation | `screen_text` 2.0 · `caption` 4.0 · `window_title` 3.0 · `site` 2.0 · `app` 1.0 |
| Episode | `title` 5.0 · `summary` 3.0 |
| Claim | `text` 6.0 · `context` 2.0 · `reasoning` not indexed |
| Entity | name/aliases 6.0 · page summary 2.0 |

A dense hit counts as *independent evidence* (for the relevance floor, §5) only
at cosine ≥ `DENSE_HIT_THRESHOLD` (default **0.5**). `CANDIDATE_POOL` (default
**60**) candidates per kind survive fusion.

**Optional graph expansion** `[Rev 1]` (off by default, `GRAPH_EXPAND`): entities
matched by the plan are expanded one hop over Stage 2's `entity_edges`, and the
neighbours' claims join the pool as an additional RRF list. This is the cheap
analogue of the spreading activation or personalized PageRank used by
Hindsight and HippoRAG for associative questions ("what else was I doing
around the Dooran logo"). It is enabled only if the harness shows a gain on
multi-episode cases.

### 4.2 Retrieval engine: stay in SQLite  (retained, trigger reframed)

The prior evaluation stands. A vector-database service, a sparse-neural
lexical model and a late-interaction model were rejected for a single-user
corpus of thousands to tens of thousands of rows: a brute-force cosine scan is
milliseconds at this size, and the other models' advantages are covered by the
dense leg and snippet extraction.

`[Rev 1]` **Trigger for `sqlite-vec` (or libSQL's native vector type):** adopt
it when keeping vectors transactionally coupled to their rows becomes
error-prone in the BLOB-table design, or when a brute-force scan exceeds 50 ms
at p95. The coupling trigger is the likelier one; the prototype's failure was
coverage, not speed. Note that `sqlite-vec` is pre-v1 with breaking changes
expected: pin the version and keep a BLOB-scan fallback path.

### 4.3 Tunables

| Name | Default |
|---|---|
| `RRF_K` | 60 |
| `DENSE_HIT_THRESHOLD` | 0.5 |
| `CANDIDATE_POOL` | 60 |
| `GRAPH_EXPAND` | off |

### 4.4 Exhaustive mode for bounded intervals  `[Rev 1]`

When `interval` is set, its length is ≤ `EXHAUSTIVE_MAX_SPAN` (default
**1 day**), and the intent is `temporal` or `recall` with timeline cues, every
episode in the interval is returned in time order. Each comes with its title,
summary and top claims, plus the digest if the day is closed. Ranking is
skipped. For listing questions, completeness is the correct objective, and
the total fits in a few thousand tokens. Beyond one day, digests are listed
exhaustively and episodes are retrieved top-k.

### 4.5 Aggregate lookup  `[Rev 1]`

For `intent = aggregate`, before any text search, entities are selected by
`entity_kind_filter` and by the interval (entities with mentions in the
interval), ordered by distinct-episode count. Each is returned with its page
summary and a count. Live claims of the matching kind in the interval are
added. Text recall then runs as usual and is fused below the structured
results. This answers "what design tools have I used" from the entity layer
even though no stored text contains the phrase "design tools".

---

## 5. Relevance scoring and the relevance floor  `[Rev 1]`

- **`echo_overlap`** (retained): trigram-Jaccard between a hit and the text
  already in the field being typed into. Above `ECHO_THRESHOLD` (default
  **0.55**) the hit is dropped as the user's own text read back.
- **`coverage`** (retained): the IDF-weighted fraction of the query's
  discriminative terms that a hit contains.
  - `[Rev 1]` Document frequency is computed only over world text: observation
    `screen_text`, captions and titles. Model-written `reasoning` and prompt
    vocabulary are never counted. Possessives and trailing punctuation are
    stripped by the tokenizer.
- **Relevance floor** (retained): a hit is dropped if `coverage <
  COVERAGE_FLOOR` (default **0.12**) and it was not an independent dense hit.
  A hit is **strong** if `coverage ≥ COVERAGE_STRONG` (default **0.55**) or it
  is an independent dense hit, and it is not an echo.
- `[Rev 1]` **Cold start:** while the corpus has fewer than `COLD_START_ROWS`
  (default **2000**) observations, the coverage floor is disabled. Hits are
  admitted on dense or BM25 presence, and strength is decided by the
  rerank relevance score (§6) at ≥ `RERANK_STRONG` (calibrated by the harness). IDF
  estimates on a tiny vocabulary are noise, and every user starts at cold
  start.

### 5.3 Recency prior and diversity  `[Rev 1]`

- Before reranking, each claim's fused score is multiplied by
  `0.5^(age_days / half_life)`. The age is measured from `occurred_to`, and the
  half-life comes from `planS2.md` §8: `event` none, `fact` 180, `preference`
  90. Episode and observation scores get a mild prior with
  `RECENCY_HALF_LIFE_DAYS` (default **30**). No prior applies when an explicit
  interval is set.
- After reranking, the final list is selected by **MMR**
  (`MMR_LAMBDA` default **0.7**, similarity = cosine of item embeddings), so
  several near-identical hits from one activity cannot fill top-k.

### 5.4 Tunables

| Name | Default |
|---|---|
| `ECHO_THRESHOLD` | 0.55 |
| `COVERAGE_FLOOR` | 0.12 |
| `COVERAGE_STRONG` | 0.55 |
| `COLD_START_ROWS` | 2000 |
| `RECENCY_HALF_LIFE_DAYS` | 30 |
| `MMR_LAMBDA` | 0.7 |
| `ENTITY_PROP_OVERLAP_THRESHOLD` | 0.6 |

---

## 6. Reranking  `[Rev 1]`

### 6.1 Mandatory rerank stage — API-served  `[Rev 2]`

The pool surviving §5 (up to `RERANK_POOL` = **50** per kind) is reranked on
every call. The rerank order decides order within the included set; the
relevance floor still decides inclusion.

Provider cascade (the first available wins):
1. **Rerank API:** OpenRouter's `/api/v1/rerank` (Cohere Rerank family;
   default `cohere/rerank-4-fast`, configurable through `RERANK_LM_*`), or any
   endpoint with the same request shape.
2. **LLM listwise** through the utility tier (the §6.2 mechanism), when no
   rerank API is configured.
3. **No rerank:** the fused and coverage ordering, when both fail. A failure
   here degrades order; it never removes a hit or blocks a result.

**Bounded input (retained from Rev 1).** Each document sent is the hit's unit
text, truncated to `RERANK_MAX_TOKENS` (default **384**, approximated as 4
characters per token):
- a claim: its text plus its best quote;
- an episode: its title, summary and best snippet;
- an observation: its caption plus the best `screen_text` window.

**Accepted tradeoff (Rev 2):** the Rev 1 local cross-encoder is replaced by a
per-query API cost and network latency (on the order of 100–300 ms), by user
decision (no locally hosted models). Rerank is skipped for pools of 3 hits or
fewer, where it cannot change much.

### 6.2 Optional LLM listwise rerank (retained, off by default)

`LLM_RERANK` (default **false**) lets one utility-tier call reorder and demote
within the reranked pool. It can never add hits, and any failure
leaves the order unchanged.

---

## 7. Grounding decision (retained) and output contract  `[Rev 1]`

A pure function of the structured counts computed during scoring, never of
re-parsed text:
- **`edit`**: a self-contained text operation; no lookup.
- **`grounded`**: at least one strong hit, a primary-source observation, an
  exhaustive-mode listing (§4.4) with at least one episode, or an aggregate
  lookup with at least one entity.
- **`thin`**: only weak hits.
- **`empty`**: nothing clears the weak bar.
- **`continuation`**: continue a draft; evidence is optional.

### 7.1 Output contract  `[Rev 1]`

Every hit in `RetrievalResult` carries:
- `kind`, `id`, text, and time;
- for claims: `modality`, `confidence` and `status` (`planS2.md` §4, §8);
- `untrusted: true` on all screen-derived text.

The product layer must present memory to its models as quoted data, never as
instructions, and must phrase `passive` claims as "you viewed…", not "you
did…". Stored text originated in third-party content, and a memory store that
executes what it remembers is the memory-poisoning attack surface.

---

## 8. Knowledge gaps  `[Rev 1]`

When grounding is `thin` or `empty`, a **verification pass** runs before any
gap is queued:
1. an exact entity lookup of every plan entity and alias;
2. a brute-force scan: BM25 over *all* rows with every plan query, plus a
   full dense scan with no pool limit. This is cheap at single-user scale.

- If verification finds a hit that would have been strong, the outcome is
  **`retrieval_miss`**. Nothing is queued for the user. The query, the missed
  row ids and the pipeline stage that dropped them are logged, and the case is
  appended to `golden/auto_cases.pending.yaml` for review (§10.4). A retrieval
  miss is a bug report, not a question for the user.
- Otherwise the outcome is **`store_lacks`**, and a gap is queued as before.
  An answered gap becomes a claim with `modality = stated`. Its confidence is
  derived per `planS2.md` §8, and `stated` is authoritative for its subject.
  This replaces the prior "confidence 10, decay 1" constants.

---

## 9. Store card  `[Rev 1]`

A computed description of the store, cached for 120 s, consumed by whatever
process plans queries. It contains:
- date coverage;
- rows per kind;
- the supports-versus-cooccurs ratio;
- the most-used apps (by `bundle_id`) and the most-mentioned entities;
- `[Rev 1]` **health**: index coverage % per kind (must be 100), the
  consolidation and outbox backlog, Stage 1 drop and degrade counters by
  reason, the confidence distribution's top-value share, the lineage-ceiling
  anomaly count, and the `retrieval_miss` count over the last 7 days.

---

## 10. Evaluation  `[Rev 1]`

### 10.1 Case format (retained, extended)

A case has:
- `question`;
- `expect_any`: verbatim distinctive substrings;
- `expect_none`: contamination checks;
- `unanswerable`;
- `[Rev 1]` `category`: one of the taxonomy in §10.2;
- `[Rev 1]` `snapshot`: the store snapshot id the case was written against;
- `[Rev 1]` `support_ids`: the row ids whose content makes the case answerable.

### 10.2 Category taxonomy  `[Rev 1]`

The categories are adapted from LongMemEval's five abilities, plus one specific
to screen memory:

| Category | Example |
|---|---|
| `extraction` | "Which font did I pick on dafont?" |
| `multi_episode` | "What have I worked on for the Dooran logo across the week?" |
| `temporal` | "What was I doing Tuesday afternoon?" |
| `knowledge_update` | "Which palette am I using now in ASCII Magic?" (monochrome, not the earlier sepia) |
| `aggregate` | "What design tools have I used?" |
| `engaged_vs_passive` | "Did I use Logofolio, or only look at it?" |
| `abstention` | Unanswerable questions (retained, first-class). |

Metrics are reported per category. An aggregate score hides exactly the
category a change breaks.

### 10.3 Metrics and gating

- **`ranx`** (retained) computes hit@k, MRR and NDCG@k from `Qrels`/`Run`
  objects built from `expect_any` matches.
- Retained and still hand-built: `expect_none` contamination,
  `mode_accuracy`, `refusal_accuracy`.
- `[Rev 1]` **Gating rule:** with ≥ `EVAL_MIN_CASES_FOR_SIG` (default **100**)
  non-stale cases, a regression is a statistically significant drop
  (`EVAL_REGRESSION_ALPHA` 0.05, paired test in `ranx`). With fewer cases, a
  regression is a drop larger than `EVAL_TOLERANCE` (default **0.05**) on any
  category's primary metric. A significance test over ~20 cases lets most real
  regressions pass as "not significant", which is worse than a tolerance band
  that at least fails loudly.

### 10.4 Golden sets are versioned against store snapshots  `[Rev 1]`

- Each evaluation run records the store snapshot id. This is a content hash
  over the live row ids and versions of every kind.
- A case whose `support_ids` are no longer live in the current store is
  reported as **`stale`**. It is excluded from metrics and never counted as a
  miss. This removes the fake-regression class that the golden set produced
  twice when the store changed.
- Generated cases (bootstrapped from real records, retained) and auto-captured
  `retrieval_miss` cases (§8) are written to `*.pending.yaml` and merged only
  after review. Golden files are never overwritten in place.

### 10.5 Baselines  `[Rev 1]`

Every full evaluation also reports two baselines, so the value of the
architecture is measured rather than assumed:
1. **Full-context:** for temporal and extraction cases whose answer lies within
   one day, a model is given every episode summary and caption for that day,
   with no retrieval. Memory-system papers routinely compare against
   full-context. If the pipeline does not beat this on short horizons, the
   short-horizon path should simply be exhaustive (§4.4).
2. **Verbatim-only:** retrieval over observations and episodes only, with no
   claims or entities. This measures what the extracted layers add; per the
   verbatim-versus-extracted evidence, they must earn their place.

### 10.6 System-health checks and model selection  `[Rev 1]`

The harness **fails** on any of the following, independent of retrieval
metrics:
- index coverage under 100 % for any kind;
- one confidence value holding more than 50 % of live claims;
- any lineage past `LINEAGE_MAX_VERSIONS`;
- entity duplicate rate over 5 %, measured by the Stage 2 lint sweep;
- a verified `retrieval_miss` on any non-stale golden case.

The same harness is the selection mechanism for every model or weight choice
left open in any stage: the API VLM (`planS1.md` §13), the embedding and
rerank models (§6.1), the utility tier (§2), consolidation's model tier, and
the BM25 field weights (§4.1), swept and compared with `ranx`.

### 10.7 BEIR (retained)

BEIR is not the regression set; it answers a different question. It remains an
optional sanity check for a candidate embedder or reranker in isolation.

### 10.8 Tunables

| Name | Default |
|---|---|
| `EVAL_K` | 8 |
| `EVAL_REGRESSION_ALPHA` | 0.05 |
| `EVAL_MIN_CASES_FOR_SIG` | 100 |
| `EVAL_TOLERANCE` | 0.05 |

---

## 11. Full tunable reference

| Name | Default | Section |
|---|---|---|
| `RRF_K` | 60 | 4.3 |
| `DENSE_HIT_THRESHOLD` | 0.5 | 4.3 |
| `CANDIDATE_POOL` | 60 | 4.3 |
| `GRAPH_EXPAND` | off | 4.1 |
| `EXHAUSTIVE_MAX_SPAN` | 1 day | 4.4 |
| `ECHO_THRESHOLD` | 0.55 | 5.4 |
| `COVERAGE_FLOOR` / `COVERAGE_STRONG` | 0.12 / 0.55 | 5.4 |
| `COLD_START_ROWS` | 2000 | 5.4 |
| `RECENCY_HALF_LIFE_DAYS` | 30 | 5.4 |
| `MMR_LAMBDA` | 0.7 | 5.4 |
| `ENTITY_PROP_OVERLAP_THRESHOLD` | 0.6 | 5.4 |
| `RERANK_POOL` | 50 | 6.1 |
| `RERANK_MAX_TOKENS` | 384 | 6.1 |
| `LLM_RERANK` / `LLM_RERANK_POOL` | false / 24 | 6.2 |
| `EVAL_*` | see 10.8 | 10.8 |

Retired by Rev 1: `SESSION_FALLBACK_FLOOR` (sessions are replaced by episodes,
which are always searched) and the `time_hint` enum (replaced by `interval`).

---

## 12. Resolved decisions

1. Grounding and gap handling stay in Stage 3 (retained).
2. `[Rev 1]` Kinds are entities, claims, episodes, observations and digests.
   Verbatim is first-class; session gating is retired.
3. The SQLite FTS5 + dense + RRF recall stack is retained. A vector-DB service,
   sparse-neural and late-interaction models remain rejected at this scale.
   `[Rev 1]` The `sqlite-vec` trigger is transactional coupling or p95 > 50 ms.
4. `[Rev 2]` Rerank stays mandatory but is API-served (OpenRouter rerank → LLM
   listwise → fused order), with bounded input. No local model.
5. `ranx` metrics are retained. `[Rev 1]` Significance gating applies only at
   ≥ 100 cases; golden sets are snapshot-versioned; there are category metrics,
   baselines and health checks.
6. `[Rev 1]` Temporal intervals, exhaustive mode for bounded intervals, and
   structured aggregate lookup.
7. `[Rev 1]` Verify before queuing a gap; retrieval misses become eval cases.
8. `[Rev 1]` The output is untrusted-marked and carries modality.

## 13. Open items

1. **The exact aggregation-cue and entity-kind vocabulary** (§2). This is an
   implementation-time list, validated by the `aggregate` eval category.
2. **Graph expansion** (§4.1) is enabled only on measured multi-episode gain.
3. **Which utility-tier API model** is good enough for query planning and
   relation classification at the lowest cost. This is decided by the harness.
4. **`RERANK_STRONG` calibration** for cold-start strength decisions (§5).

## 14. Out of scope

- The product and UI layer: the hotkey, the tool-calling research loop,
  streaming writing, field insertion, and how gaps are surfaced to the user.
- Capture (`planS1.md`) and processing (`planS2.md`).

## 15. Overall assessment

The core of the prior design is sound and retained:
- structured results instead of re-parsed text;
- a relevance floor with a dense bypass;
- a grounding decision that is a pure function of counts;
- first-class unanswerable cases;
- a mandatory rerank stage;
- declining heavyweight retrieval infrastructure at this scale.

Rev 1 corrects what the real data exposed. The retrieval target now includes
the verbatim layer that extracted memory loses. Listing and aggregation
questions get structurally appropriate paths instead of top-k text search.
Term statistics measure the world, not the prompt. The system can tell its own
recall failures apart from genuine ignorance. And the evaluation now measures
per category, against baselines, on snapshot-stable golden sets, with health
checks that would have caught every prototype failure listed in `audit.md`.
