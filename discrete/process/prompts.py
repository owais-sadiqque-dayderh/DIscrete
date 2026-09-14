DRAFT_EPISODE_PROMPT = """You draft structured memory from one user activity episode.

Episode observations (newest last):
{observations}

Return strict JSON:
{{
  "title": "<string>",
  "summary": "<string: 2-4 sentences>",
  "events": [
    {{"text": "<string ≤240 chars>", "quote_obs_id": <int>, "quote": "<verbatim substring>"}}
  ],
  "facts": [
    {{"text": "<string ≤240 chars>", "quote_obs_id": <int>, "quote": "<verbatim substring>"}}
  ],
  "preferences": [
    {{"text": "<string ≤240 chars>", "quote_obs_id": <int>, "quote": "<verbatim substring>"}}
  ],
  "entities": ["<proper noun>", "..."]
}}

Rules:
- One atomic claim per item. events = dated actions; facts = durable truths; preferences = beliefs/tastes.
- Every quote MUST be a verbatim substring of that observation's screen_text or caption.
- Do not merge unrelated activities.
"""

RELATE_PROMPT = """Classify relation between NEW claim and EXISTING claim.

NEW ({new_kind}): {new_text}
EXISTING ({old_kind}): {old_text}

Return JSON: {{"relation": "same"|"contradicts"|"refines"|"unrelated"}}
"""
