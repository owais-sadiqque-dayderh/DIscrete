"""Offline privacy boundary regressions with synthetic frames."""
import asyncio
import time

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import screen as sc


async def test_unapproved_cached_before_frame_is_omitted(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: ('Safe document', 'ocr'))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Document', pid=7)
    before = Image.new('RGB', (64, 64), 'black')
    after = Image.new('RGB', (64, 64), 'white')
    pending = dict(kind='click', started=time.time(), before_ctx=ctx,
                   counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    await observer._capture(pending, ctx, before, after)
    cap = observer.queue.get_nowait()
    assert cap.job.frames == [after]


async def test_full_ocr_is_screened_before_excerpt(store, fake, monkeypatch):
    from discrete.capture import ocr
    monkeypatch.setattr(ax, 'read_screen_text', lambda *a, **kw: '')
    monkeypatch.setattr(ocr, 'ocr_image', lambda image: 'ordinary prose ' * 2000 + '\npassword: synthetic12345')
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Document', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    await observer._capture(pending, ctx, None, Image.new('RGB', (64, 64), 'white'))
    cap = observer.queue.get_nowait()
    assert cap.job.frames == []


async def test_failed_ocr_cannot_approve_pixels(store, fake, monkeypatch):
    from discrete.capture import ocr
    monkeypatch.setattr(ax, 'read_screen_text', lambda *a, **kw: 'Safe AX text')
    def fail(image):
        raise RuntimeError('synthetic extraction failure')
    monkeypatch.setattr(ocr, 'ocr_image', fail)
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Document', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    await observer._capture(pending, ctx, None, Image.new('RGB', (64, 64), 'white'))
    cap = observer.queue.get_nowait()
    assert cap.job.frames == []
    assert 'failed' in cap.draft.text_source


async def test_missing_frame_is_not_counted_as_redaction(store, fake, monkeypatch):
    monkeypatch.setattr(ax, 'read_screen_text', lambda *a, **kw: 'Safe AX text')
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    ctx = ax.WindowContext(app='Editor', window_title='Document', pid=7)
    pending = dict(kind='click', started=time.time(), counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    await observer._capture(pending, ctx, None, None)
    assert observer.stats['redacted'] == 0
    assert observer.queue.get_nowait().draft.screen_text == 'Safe AX text'
