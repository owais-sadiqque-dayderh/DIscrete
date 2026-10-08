"""Semantic sets and navigation lists use eligible source evidence."""
import pytest

from discrete.models import Hit, ObservationDraft, QueryPlan, RetrievalResult
from discrete.retrieval.enumerate import shape_enumerate_items
from discrete.retrieval.hybrid import retrieve
from discrete.retrieval.query import fallback_plan


def test_reject_all_does_not_restore_listing():
    plan = QueryPlan('list articles I read in news', intent='enumerate')
    hit = Hit('observation', 1, 'News', '', is_listing=True, status='matched')
    assert shape_enumerate_items([hit], plan) == []


def test_proximity_is_not_membership_and_url_identity_is_preserved():
    plan = QueryPlan('list articles I opened from News', intent='enumerate')
    hits = [Hit('observation', 1, 'News', '', is_listing=True, status='matched'),
            Hit('observation', 2, 'Story', '', status='trail', navigated_from='News', url='https://example.test/a'),
            Hit('observation', 3, 'Story', '', status='trail', navigated_from='News', url='https://example.test/b'),
            Hit('observation', 4, 'Unrelated', '', status='trail')]
    assert [h.ref_id for h in shape_enumerate_items(hits, plan)] == [2, 3]


@pytest.mark.parametrize('query', [
    'summarise all the messages I viewed in campus planning group',
    'summarize every message I saw in campus planning group',
    'give me a summary of all the messages I viewed in campus planning group',
])
async def test_message_enumeration_searches_contents(store, fake, query):
    await store.add_observation(ObservationDraft(app='Chat', window_title='Campus planning group',
        screen_text='Campus planning group\nAda: The synthetic meeting starts at 14:00.',
        started_at=100, ended_at=110))
    res = await retrieve(store, query)
    assert res.hits and any('14:00' in h.body for h in res.hits)
    assert res.plan.intent == 'recall'


async def test_empty_page_candidates_fall_back_to_recall(store, fake, monkeypatch):
    import discrete.retrieval.hybrid as hybrid

    await store.add_observation(ObservationDraft(app='Browser', window_title='News',
        screen_text='News article about helium budget.', started_at=100, ended_at=110))

    async def empty(*args):
        pass

    monkeypatch.setattr(hybrid, 'enumerate_pages', empty)
    res = await retrieve(store, 'list all articles I read in News')
    assert res.hits and res.plan.intent == 'recall'


def test_capped_list_reports_known_total():
    from discrete.bubble.engine import format_enumerate_answer

    result = RetrievalResult('list pages', QueryPlan('list pages', intent='enumerate'),
        [Hit('observation', 1, 'Story', '', verification='verified')], enumeration_total=3,
        enumeration_truncated=True)
    assert 'showing 1 of 3' in format_enumerate_answer(result).lower()


def test_what_all_paraphrase_is_enumeration():
    assert fallback_plan('what all articles did I read in yc hacker news').intent == 'enumerate'


async def test_rejected_enumeration_falls_back_to_recall(store, fake):
    """The verifier rejecting every page must not leave an empty answer."""
    await store.add_observation(ObservationDraft(app='Browser', window_title='News', url='https://news.example/',
        screen_text='News front page: synthetic helium story.', started_at=100, ended_at=110))
    fake.support_indices = []
    res = await retrieve(store, 'what all articles did I read in news')
    assert res.plan.intent == 'recall' and not res.hits


async def test_source_named_by_part_of_the_subject(store, fake):
    """A page reached from 'Example News' answers a question about 'yc example news'."""
    await store.add_observation(ObservationDraft(app='Browser', window_title='Example News', url='https://news.example/',
        screen_text='Example News front page', is_listing=True, started_at=100, ended_at=110))
    await store.add_observation(ObservationDraft(app='Browser', window_title='Deep sea robots',
        url='https://blog.example/robots', screen_text='Underwater robots explore trenches.',
        navigated_from='Example News', started_at=5000, ended_at=5010))
    res = await retrieve(store, 'what all articles did I read in yc example news')
    assert res.plan.intent == 'enumerate'
    assert [h.title for h in res.hits] == ['Deep sea robots']


def test_nearby_pages_are_labelled_not_listed_as_members():
    from discrete.bubble.engine import format_enumerate_answer

    hits = [Hit('observation', 1, 'Opened story', '', status='trail', navigated_from='News', verification='verified'),
            Hit('observation', 2, 'Nearby page', '', status='trail', verification='verified')]
    text = format_enumerate_answer(RetrievalResult('q', QueryPlan('list pages', intent='enumerate'), hits))
    linked, nearby = text.split('Also open nearby')
    assert 'Opened story' in linked and 'Nearby page' in nearby


async def test_list_is_cut_with_a_visible_count(store, fake, monkeypatch):
    from discrete.bubble.engine import format_enumerate_answer
    from discrete.config import T

    monkeypatch.setattr(T, 'ENUMERATE_MAX', 2)
    for i in range(3):
        await store.add_observation(ObservationDraft(app='Browser', window_title=f'Lab page {i}',
            url=f'https://lab.example/{i}', screen_text='lab notes', started_at=100 + 10_000 * i,
            ended_at=110 + 10_000 * i))
    res = await retrieve(store, 'list every lab page')
    assert len(res.hits) == 2 and res.enumeration_total == 3
    assert format_enumerate_answer(res).endswith('(showing 2 of 3)')


@pytest.mark.parametrize('query,intent', [
    ('what all articles did I read in example news', 'enumerate'),
    ('list all the articles I opened in example news', 'enumerate'),
    ('show me every site I visited today', 'enumerate'),
    ('which pages did I open from example news, list all of them', 'enumerate'),
    ('summarise all the messages I viewed in the planning group', 'recall'),
    ('Summarize every message I saw in the planning group', 'recall'),
    ('give me a summary of all the messages I viewed in the planning group', 'recall'),
    ('list all the restaurants mentioned in the group chat', 'recall'),
    ('list every message I viewed in the planning group', 'recall'),
    ('summarize this paragraph', 'edit'),
    ('summarise the text above', 'edit'),
    ('rewrite the email I sent yesterday more politely', 'edit'),
])
def test_list_and_summary_paraphrases_route_by_what_is_asked(query, intent):
    assert fallback_plan(query).intent == intent
