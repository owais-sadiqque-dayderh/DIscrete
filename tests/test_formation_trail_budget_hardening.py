from discrete import llm
from discrete.config import Endpoint
from discrete.process import trails


async def test_trail_verifier_gets_the_claim_verifier_budget_plus_extra_tokens(monkeypatch, fake):
    asked = []

    async def response(ep, messages, *, max_tokens, **kwargs):
        asked.append(llm._tuned(Endpoint("verifier", "", "k", "m", extra_tokens=1000), None, max_tokens)[1])
        return {"decisions": [{"index": i, "supported": True} for i in range(4)]}

    monkeypatch.setattr(trails, "complete_obj", response)
    link = {"relation": "step", "at": 100, "quote": "Acme", "explanation": "searched Acme"}
    await trails._verify({"text": "The user downloaded Acme", "occurred_from": 105}, [link] * 4)
    # The live failure was 60 + 30 * 4 + VERIFY_LM_EXTRA_TOKENS=1000 = 1180, all spent thinking.
    assert asked == [400 + 200 * 4 + 1000]
