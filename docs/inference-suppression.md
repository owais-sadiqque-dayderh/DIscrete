# Inference feedback and bridge integration

Migration 016 adds `suppressions.scope` (`kind` or `inference`) and nullable
`retired_after`. Keeping scope separate from kind preserves the original feedback
kind for auditing and preserves forget/erase's existing kind-specific gates.
Inference scope applies only to preference, pattern and fact, never events.
Existing reject rows and suppression snapshots inside undo events are upgraded.
Revert removes the event's rows; redo restores scope, timestamp and vector exactly.

Public APIs:

```python
async def reject_inference(self, text: str, *, kind: str = "preference",
                           claim_ids: Sequence[int] = (), reason: str = "") -> int
async def retire_inference(self, text: str, *, kind: str = "preference",
                           claim_ids: Sequence[int] = (), reason: str = "") -> int
async def suppressed(self, kind: str, text: str, vector=None, *,
                     claim_ids: Sequence[int] = (),
                     observation_ids: Sequence[int] = ()) -> bool
```

All feedback leaves source memories untouched. `reject_inference` is permanent
until reverted. `retire_inference` blocks a candidate unless its cited live claims
have supporting user observations with `started_at` strictly greater than the
feedback timestamp. Newly verified observation IDs can be passed directly by the
drafting pipeline. Processing old evidence again, changing claim timestamps, or
adding co-occurrence evidence does not revive it. Revival is per candidate; the
retirement row remains, so subsequent suggestions citing only old evidence stay
blocked. New supporting evidence must support the candidate, not just mention its
subject. No expiry timer can revive an idea solely because time passed.

Matching tries normalized exact wording (and erased hashes), existing Jaccard
0.80, inference token-set ratio 92 with content-token Jaccard at least 0.50, and
cosine similarity 0.90 when both vectors exist. Constants are commented in code.
The conservative token threshold was selected against reordered dark-mode and
brown-switch keyboard paraphrases, and different switch/card/shopping assertions.
The overlap floor prevents a shared entity or small token subset from being enough.
Explicit opposite negation is excluded before approximate inference matching.
"You buy on Flipkart with the Axis card" and "You prefer Flipkart for electronics"
are different assertions and should not match: payment behavior does not establish
an electronics shopping preference. Semantic 0.90 retains the existing suppression
threshold; tests exercise scores above and below it. These are heuristic defaults,
not a calibrated guarantee for every embedding model or every paraphrase.

Feedback reuses a vector only for the exact wording with empty context; otherwise
it calls `llm.try_embed` on the feedback wording. A missing/failing endpoint leaves
the vector null and preserves lexical protection. The public read gate computes a
candidate embedding only after lexical matching misses and stored vectors exist.
Callers can pass a vector already computed for that wording. Dream pattern writes,
profile writes, and default profile reads use the public gate. Transaction-time
claim gates still check current suppression rows and fresh supporting evidence.
`get_profile(live_only=False)` intentionally returns historical raw snapshots.

Bridge changes required (outside this implementation's repository scope):

- Filter synthesized points before verification/return, and again after any stage
  that changes their text, with `await store.suppressed("preference", text,
  claim_ids=cited_source_ids)`. Omit suppressed points. Events used as premises
  remain available; filter the inferred lasting point, not the source event.
- Pass cited IDs on every filter call so outdated inferences can revive with new
  supporting evidence. Without IDs the safe result remains suppressed.
- Map inferred "no longer true" to `retire_inference(text, kind="preference",
  claim_ids=ids, reason=reason)`; keep inferred "never true" on `reject_inference`.
  Keep the returned event ID in the bridge undo list and update its feedback map.
- Continue using `retire_claim` for a directly represented lasting memory.
- When an inferred edit/scoped correction must withdraw the original assertion,
  reject the original wording as a separate reversible event alongside the new
  stated claim. This store change does not change bridge feedback mapping itself.
