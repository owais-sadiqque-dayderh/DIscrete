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
