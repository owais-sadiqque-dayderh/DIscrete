"""Deferred caption work must respect current policy before egress."""
from PIL import Image

from discrete.capture import privacy
from discrete.capture.enrich import Enricher, EnrichJob, backfill_page_fields
from discrete.models import ObservationDraft


async def test_blocked_caption_job_makes_no_call(store, fake, monkeypatch):
    monkeypatch.setattr(privacy, '_SITES', ('blocked.invalid',))
    job = EnrichJob([Image.new('RGB', (64, 64))], 'Safari', 'Document',
                    'https://blocked.invalid', 'Synthetic document', dict(keys=0, clicks=0, scrolls=0, clipboard=0))
    assert await Enricher(store).vlm(job) is None
    assert fake.calls == []


async def test_pending_blocked_source_is_finalized_without_call(store, fake, monkeypatch):
    oid, _ = await store.add_observation(ObservationDraft(app='Editor', window_title='Synthetic',
                                                        screen_text='Synthetic pending document', enrich_state='pending'))
    monkeypatch.setattr(privacy, '_APPS', frozenset({'editor'}))
    enricher = Enricher(store)
    await enricher.fill_pending()
    assert (await store.get_observation(oid))['enrich_state'] != 'pending'
    assert fake.calls == []


async def test_blocked_backfill_makes_no_call(store, fake, monkeypatch):
    await store.add_observation(ObservationDraft(app='Editor', window_title='Synthetic',
                                               screen_text='Synthetic backfill document', caption='Synthetic caption'))
    monkeypatch.setattr(privacy, '_APPS', frozenset({'editor'}))
    out = await backfill_page_fields(store)
    assert out['filled'] == 0
    assert fake.calls == []


async def test_forgotten_row_makes_no_call(store, fake):
    job = EnrichJob([], 'Editor', 'Document', '', 'Synthetic document', dict(keys=0, clicks=0, scrolls=0, clipboard=0))
    assert await Enricher(store).vlm(job, oid=999) is None
    assert fake.calls == []
