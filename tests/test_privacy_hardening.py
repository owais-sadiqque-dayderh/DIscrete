"""Offline privacy boundary regressions with synthetic frames."""
import asyncio
import time

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import screen as sc


async def _capture_pair(store, monkeypatch, before):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: sc.ScreenText('Safe document', 'ocr', 'Safe document', True))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    ctx = ax.WindowContext(app='Editor', window_title='Document', pid=7)
    after = Image.new('RGB', (64, 64), 'white')
    pending = dict(kind='click', started=time.time(), before_ctx=ctx,
                   counts=dict(keys=0, clicks=1, scrolls=0, clipboard=0))
    await observer._capture(pending, ctx, before, after)
    return observer.queue.get_nowait().job, after


async def test_cached_before_frame_needs_its_own_approval(store, fake, monkeypatch):
    from discrete.capture import ocr
    from discrete.capture.enrich import screen_frames
    read = []
    def frame_text(image):
        read.append(image.getpixel((0, 0)))
        return 'password: synthetic12345' if image.getpixel((0, 0)) == (0, 0, 0) else 'Plain page'
    monkeypatch.setattr(ocr, 'ocr_image', frame_text)
    secret_before = Image.new('RGB', (64, 64), 'black')
    job, after = await _capture_pair(store, monkeypatch, secret_before)
    assert job.frames == [secret_before, after]  # queued, but not yet approved
    assert screen_frames(job).frames == [after]
    assert read == [(0, 0, 0)]  # the screened after frame is not read twice

    clean_before = Image.new('RGB', (64, 64), 'gray')
    job, after = await _capture_pair(store, monkeypatch, clean_before)
    assert screen_frames(job).frames == [clean_before, after]


async def test_unreadable_before_frame_is_withheld(store, fake, monkeypatch):
    from discrete.capture import enrich, ocr
    def fail(image):
        raise RuntimeError('synthetic extraction failure')
    sent = []
    async def complete_json(ep, messages, **kw):
        sent.append([p for p in messages[0]['content'] if p['type'] != 'text'])
        return {'caption': 'synthetic'}
    job, after = await _capture_pair(store, monkeypatch, Image.new('RGB', (64, 64), 'black'))
    monkeypatch.setattr(ocr, 'ocr_image', fail)
    monkeypatch.setattr(enrich, 'complete_json', complete_json)
    await enrich.Enricher(store).vlm(job)
    assert [len(images) for images in sent] == [1]  # only the after frame, approved at capture


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
