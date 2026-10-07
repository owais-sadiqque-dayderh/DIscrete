import pytest

from discrete.llm import LLMError
from discrete.process import trails

LINK = {"relation": "step", "at": 100, "quote": "Acme", "explanation": "searched Acme"}
CLAIM = {"id": 1, "text": "The user downloaded Acme", "occurred_from": 105}


@pytest.mark.parametrize("decisions", [
    [{"index": True, "supported": True}],
    [{"index": 0, "supported": True}, {"index": 0, "supported": False}],
    [{"index": 0, "supported": "true"}], [{"index": 2, "supported": True}], [], "bad"])
async def test_malformed_trail_decisions_are_retried_not_applied(monkeypatch, fake, decisions):
    async def response(*args, **kwargs):
        return {"decisions": decisions}

    monkeypatch.setattr(trails, "complete_obj", response)
    with pytest.raises(LLMError):
        await trails._verify(CLAIM, [LINK])


async def test_complete_trail_decisions_select_supported_links(monkeypatch, fake):
    other = {**LINK, "quote": "Acme docs"}

    async def response(*args, **kwargs):
        return {"decisions": [{"index": 1, "supported": True}, {"index": 0, "supported": False}]}

    monkeypatch.setattr(trails, "complete_obj", response)
    assert await trails._verify(CLAIM, [LINK, other]) == [other]


async def test_malformed_trail_proposal_is_not_recorded_as_no_trail(store, monkeypatch, fake):
    async def response(*args, **kwargs):
        return {"links": "none"}

    async def moments(*args, **kwargs):
        return [{"id": 1}]

    monkeypatch.setattr(trails, "complete_obj", response)
    monkeypatch.setattr(trails, "candidates", moments)
    monkeypatch.setattr(trails, "_moment", lambda r, terms: "moment")
    with pytest.raises(LLMError):
        await trails.trail_for(store, CLAIM)
