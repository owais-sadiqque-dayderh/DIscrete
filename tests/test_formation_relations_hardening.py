from discrete.models import ClaimDraft
from discrete.process import pipeline
from discrete.process.pipeline import ProcessPipeline, _Durable

OLD = {"id": 1, "text": "The user uses Atlas", "context": "personal", "modality": "engaged",
       "occurred_from": 200, "occurred_to": 205, "valid_from": 200}


def _contradicting(at: float) -> _Durable:
    cd = ClaimDraft(kind="fact", text="The user uses Acme", modality="engaged", occurred_from=at, occurred_to=at + 5)
    return _Durable(cd, cd.text, candidates=[OLD], relations={1: "contradicts"})


def test_older_evidence_disputes_but_never_supersedes_a_newer_claim():
    w = ProcessPipeline(None)._operation(_contradicting(100))
    assert w.operation == "added" and w.target_id is None and w.contested_id == 1


def test_newer_evidence_still_supersedes():
    w = ProcessPipeline(None)._operation(_contradicting(300))
    assert w.operation == "invalidated" and w.target_id == 1


def test_evidence_older_than_the_latest_support_does_not_supersede():
    w = ProcessPipeline(None)._operation(_contradicting(203))  # after valid_from, before occurred_to
    assert w.operation == "added" and w.contested_id == 1


async def test_relation_prompt_carries_context_and_time(store, monkeypatch):
    prompts = []

    async def response(ep, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        return {"relations": [{"new": 0, "id": 1, "relation": "unrelated"}]}

    monkeypatch.setattr(pipeline, "complete_json", response)
    cd = ClaimDraft(kind="fact", text="The user uses Acme", context="work", occurred_from=100, occurred_to=105)
    await ProcessPipeline(store)._relate([_Durable(cd, cd.text, candidates=[OLD])])
    (prompt,) = prompts
    assert 'context: "work"' in prompt and 'context: "personal"' in prompt
    assert prompt.count("seen: 1970-01-01") == 2 or prompt.count("seen: 1969-12-31") == 2
