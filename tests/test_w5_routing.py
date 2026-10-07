"""Routing regexes hold for paraphrases, not only the phrasings they were written from."""
import pytest

from discrete.retrieval.query import fallback_plan, is_premise_question, relative_event


@pytest.mark.parametrize('query,phrase,direction,last', [
    ('What did I do right after I ran deployctl publish?', 'ran deployctl publish', 'after', False),
    ('what did i open just after i ran deployctl publish', 'ran deployctl publish', 'after', False),
    ('What did I look at immediately after running deployctl publish?', 'running deployctl publish', 'after', False),
    ('Which page did I visit right before I ran deployctl publish?', 'ran deployctl publish', 'before', False),
    ('what was I doing before I opened the invoice?', 'opened the invoice', 'before', False),
    ('What did I do when I finished the quarterly report?', 'finished the quarterly report', 'after', True),
    ('What did I open after finishing the budget sheet?', 'finishing the budget sheet', 'after', True),
    ('Where did I go after I closed the budget sheet?', 'closed the budget sheet', 'after', False),
])
def test_relative_event_paraphrases(query, phrase, direction, last):
    event = relative_event(query)
    assert event is not None and (event.phrase, event.direction, event.last) == (phrase, direction, last)
    assert fallback_plan(query).intent == 'temporal'


@pytest.mark.parametrize('query', [
    'what did i do after lunch',                       # a time of day, not an event
    'what did i do after 3pm',                         # a clock is an interval
    'What did Ada reply after I sent the invoice?',    # someone else's content
    'Why did I open the page after the call?',         # causal questions have their own route
])
def test_non_events_stay_off_the_anchor_route(query):
    assert relative_event(query) is None


@pytest.mark.parametrize('query', [
    'Did I book the flight?', 'did i actually pay the invoice', 'Have I sent the report to Ada?',
    'Did I really order the chair?', 'Which did I buy, the blue or the red one?', 'Did I finish booking the hotel?',
    'Did I end up buying the chair?', 'Have I already paid the electricity bill?',
    'Did I go ahead and submit the form?', 'did I ever reply to Ada', 'Did I complete the purchase?',
    'Did I subscribe to the newsletter?',
])
def test_completion_check_paraphrases(query):
    assert is_premise_question(query)
    assert fallback_plan(query).premise_check


@pytest.mark.parametrize('query', [
    'When did I book the flight?', 'Why did I book the flight?', 'What did I read about booking flights?',
    'Did I read the booking page?',
])
def test_non_completion_questions_are_not_premise_checks(query):
    assert not is_premise_question(query)
