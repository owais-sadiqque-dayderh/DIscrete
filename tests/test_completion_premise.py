"""Completion checks stay deterministic; prompt construction never calls a model."""

import pytest

from discrete.bubble.engine import BubbleEngine, Prepared
from discrete.bubble.prompts import MODE_INSTRUCTIONS, PREMISE_INSTRUCTION, WRITER_SYSTEM
from discrete.models import RetrievalResult
from discrete.retrieval.grounding import GroundingDecision
from discrete.retrieval.query import build_plan, fallback_plan, is_premise_question
from discrete.retrieval.support import _PROMPT


@pytest.mark.parametrize("question", [
    "Which carrier did I actually book a ticket with?",
    "What did I really buy from the shop?",
    "Have I actually paid the invoice?",
    "Had I really sent the message?",
    "Did I switch the backend to Atlas or Boreal?",
    "Have I installed Atlas or Boreal?",
    "Which backend did I switch to, Atlas or Boreal?",
    "What did I order, the keyboard or the mouse?",
    "Did I actually order the keyboard?",
    "Did I submit the form?",
    "Have I subscribed to the service?",
    "Did I reply to the email?",
    "Have I really bought the keyboard?",
    "Did I purchase the monitor?",
])
def test_completion_premise_questions(question):
    assert is_premise_question(question)
    assert fallback_plan(question).premise_check


@pytest.mark.parametrize("question", [
    "When did I install the ChatGPT desktop app?",
    "When did I actually install the editor?",
    "What time did I really send the message?",
    "How long ago did I actually subscribe?",
    "Why did I actually buy the keyboard?",
    "Did I compare Atlas or Boreal?",
    "Which backend did I compare?",
    "Which editor did I install last week?",
    "What did I read about installing the editor?",
    "What did I open after I switched the backend?",
    "Did I open the checkout?",
    "What did I actually view?",
    "Tell me whether the book or magazine was interesting.",
    "Did I bookkeep the expenses?",
])
def test_ordinary_recall_is_not_a_completion_premise(question):
    assert not is_premise_question(question)
    assert not fallback_plan(question).premise_check


async def test_llm_planner_preserves_completion_flag(fake):
    assert (await build_plan("Did I switch to Atlas or Boreal?")).premise_check
    assert not (await build_plan("When did I install the editor?")).premise_check


@pytest.mark.parametrize("mode", ["grounded", "thin", "empty"])
def test_messages_append_premise_instruction_only_when_flagged(store, monkeypatch, mode):
    monkeypatch.setattr("discrete.bubble.engine.now", lambda: 1_789_000_000.0)
    question = "Did I switch to Atlas or Boreal?"
    plan = fallback_plan(question)
    result = RetrievalResult(question, plan)
    prepared = Prepared(question, question, "Notes", "text field", result, GroundingDecision(mode, 0, 0))
    engine = BubbleEngine(store)
    flagged = engine.messages(prepared)
    assert flagged[1]["content"].endswith(MODE_INSTRUCTIONS[mode] + "\n" + PREMISE_INSTRUCTION)

    plan.premise_check = False
    ordinary = engine.messages(prepared)
    assert "Premise check:" not in ordinary[1]["content"]
    assert ordinary[1]["content"].endswith(MODE_INSTRUCTIONS[mode])
    assert flagged[0] == ordinary[0]
    assert flagged[1]["content"] == ordinary[1]["content"] + "\n" + PREMISE_INSTRUCTION


def test_prompts_require_completion_evidence_and_preserve_offer_scope():
    assert "only when a record shows completion" in WRITER_SYSTEM
    assert 'button label ("Book with X", "Place order")' in WRITER_SYSTEM
    assert "nothing on file shows completion" in WRITER_SYSTEM
    assert "answer the premise first" in WRITER_SYSTEM
    assert "An offer to explain or compare supports only the offer" in WRITER_SYSTEM
    assert "A button label, checkout, form or comparison supports viewed/selected" in _PROMPT
    assert "Require completion evidence" in _PROMPT
    assert "does not support the explanation or comparison" in _PROMPT
    for evidence in ("confirmation", "receipt", "success message", "thank-you page", "changed config", "command output"):
        assert evidence in WRITER_SYSTEM
        assert evidence in _PROMPT
