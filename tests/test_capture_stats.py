"""Frameless rotated captures: counted as such, and avoided when a frame exists."""
import asyncio
import logging
import time

from PIL import Image

from discrete.capture import accessibility as ax
from discrete.capture import screen as sc


def _rotate(observer, ctx):
    observer._pending = dict(kind='window_change', started=time.time(), before_ctx=ctx, before_img=None,
                             counts=dict(keys=0, clicks=0, scrolls=0, clipboard=0))
    observer._rotate()
    return asyncio.gather(*observer._tasks)


async def test_missing_rotated_frame_is_counted_without_claiming_drop(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: ('', 'ocr_failed'))
    await _rotate(observer, ax.WindowContext(app='Editor', window_title='Synthetic document', pid=7))
    assert not observer.queue.empty()  # the row is still stored, from metadata
    assert 'dropped_failed' not in observer.stats
    assert observer.stats['missing_frame'] == 1


async def test_rotation_finds_frame_cached_with_apple_events_url(store, fake, monkeypatch):
    observer = sc.ScreenObserver(queue=asyncio.Queue(), store=store)
    monkeypatch.setattr(observer, '_screen_text', lambda *a: ('Synthetic article text', 'ocr'))
    monkeypatch.setattr(observer, '_save', lambda *a: '')
    # The trigger's AX context has no URL; the frame loop added the tab URL.
    landed = ax.WindowContext(app='Safari', window_title='Synthetic article', pid=7)
    cached = ax.WindowContext(app='Safari', window_title='Synthetic article', pid=7,
                              url='https://news.example.invalid/a')
    frame = Image.new('RGB', (64, 64), 'white')
    observer._recent.append((frame, cached))
    await _rotate(observer, landed)
    cap = observer.queue.get_nowait()
    assert observer.stats['missing_frame'] == 0
    assert cap.job.frames == [frame] and cap.draft.url == 'https://news.example.invalid/a'


def test_one_flat_window_is_not_a_permission_warning(caplog):
    observer = sc.ScreenObserver(queue=asyncio.Queue())
    flat, content = Image.new('RGB', (64, 64), 'black'), Image.effect_noise((64, 64), 64).convert('RGB')
    with caplog.at_level(logging.WARNING, logger='discrete.capture.screen'):
        for _ in range(5):
            observer._check_blank(flat, ax.WindowContext(pid=1, window_id=1))
        observer._check_blank(content, ax.WindowContext(pid=2, window_id=2))
        for i in range(3, 9):
            observer._check_blank(flat, ax.WindowContext(pid=i, window_id=i))
    assert caplog.records == []
    denied_permission = sc.ScreenObserver(queue=asyncio.Queue())
    with caplog.at_level(logging.WARNING, logger='discrete.capture.screen'):
        for i in range(5):
            denied_permission._check_blank(flat, ax.WindowContext(pid=i, window_id=i))
    assert len(caplog.records) == 1 and 'Screen Recording' in caplog.records[0].message
    observer.grabber.close()
    denied_permission.grabber.close()
