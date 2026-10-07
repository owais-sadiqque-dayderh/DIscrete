"""Visual similarity must not discard factual edits or failed retries."""
import asyncio
import time

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import screen as sc


async def test_same_pixels_changed_number_is_not_duplicate(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    text = ['Price is 100 dollars']
    monkeypatch.setattr(observer, '_screen_text', lambda *a: (text[0], 'ax'))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Synthetic', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    image = Image.new('RGB', (64, 64), 'white')
    await observer._capture(pending, ctx, None, image)
    first = observer.queue.get_nowait()
    await store.add_observation(first.draft)
    text[0] = 'Price is 101 dollars'
    await observer._capture(pending, ctx, None, image)
    second = observer.queue.get_nowait()
    assert not second.draft.frame_duplicate
    assert '101' in second.draft.screen_text


async def test_uningested_capture_cannot_poison_duplicate_gate(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: ('Synthetic document', 'ax'))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Synthetic', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    image = Image.new('RGB', (64, 64), 'white')
    await observer._capture(pending, ctx, None, image)
    observer.queue.get_nowait()  # simulate failed durable ingestion
    await observer._capture(pending, ctx, None, image)
    assert not observer.queue.get_nowait().draft.frame_duplicate


def test_ocr_numeric_edit_is_retained_even_with_matching_words():
    assert sc._novel_lines('Invoice total amount 200 USD', 'Invoice total amount 100 USD') == [
        'Invoice total amount 200 USD']


async def test_unchanged_frame_uses_only_a_durable_target(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: ('Synthetic document', 'ax'))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Synthetic', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    image = Image.new('RGB', (64, 64), 'white')
    await observer._capture(pending, ctx, None, image)
    first = observer.queue.get_nowait()
    await store.add_observation(first.draft)
    await observer._capture(pending, ctx, None, image)
    assert observer.queue.get_nowait().draft.frame_duplicate
    # An old source cannot indefinitely bypass the full-observation ceiling.
    await store.execute('UPDATE observations SET started_at=1')
    await observer._capture(pending, ctx, None, image)
    assert not observer.queue.get_nowait().draft.frame_duplicate

