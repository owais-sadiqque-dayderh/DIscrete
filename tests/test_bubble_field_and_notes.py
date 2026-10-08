"""Bubble in real apps: unreadable focus details are not a move, and notes stay off the field."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from discrete.bubble import service
from discrete.bubble.engine import EMPTY_WRITER_ANSWER, UNVERIFIED_NOTE
from discrete.capture.accessibility import FocusSnapshot, WindowContext


def snapshot(**kw):
    base = dict(pid=7, app='Arc', window_id=11, window_title='Chat', url='https://chat.example/c/1',
                element=object(), value='why did I download Helium?')
    return FocusSnapshot(**{**base, **kw})


def patch_focus(monkeypatch, ctx, focused):
    monkeypatch.setattr(service.ax, 'window_context', lambda: ctx)
    monkeypatch.setattr(service.ax, 'AXUIElementCreateApplication', lambda pid: object(), raising=False)
    monkeypatch.setattr(service.ax, '_ax_get', lambda *args: focused[0])
    monkeypatch.setattr(service, 'denied', lambda *args, **kwargs: None)


def test_unreadable_url_or_element_is_not_a_move(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot()
    ctx = WindowContext(pid=7, app='Arc', window_id=11, window_title='Chat', url='')
    focused = [None]
    patch_focus(monkeypatch, ctx, focused)
    assert bubble._focus_matches(snap)  # Chromium URL and focused element timed out
    focused[0] = object()
    assert not bubble._focus_matches(snap)  # a different, readable field
    focused[0] = snap.element
    ctx.app, ctx.pid = 'Slack', 8
    assert not bubble._focus_matches(snap)  # another app


def test_field_ax_never_exposed_still_types_in_same_window(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot(element=None, value='')
    ctx = WindowContext(pid=7, app='Arc', window_id=11, window_title='Chat', url=snap.url)
    patch_focus(monkeypatch, ctx, [None])
    assert bubble._focus_matches(snap)
    ctx.window_id = 12
    assert not bubble._focus_matches(snap)


async def run_with_writer(store, monkeypatch, chunks, field='why did I download Helium?'):
    overlay = Mock()
    bubble = service.BubbleService(store, overlay=overlay)
    snap = snapshot(value=field)
    monkeypatch.setattr(FocusSnapshot, 'restore', lambda self: None)
    monkeypatch.setattr(bubble, '_focus_matches', lambda s: True)
    monkeypatch.setattr(service, 'hold', lambda: None)
    monkeypatch.setattr(service, 'release', lambda: None)
    monkeypatch.setattr(service, 'note_agent_output', lambda s: None)
    monkeypatch.setattr(service, 'denied', lambda *args, **kwargs: None)

    async def watch(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(bubble, '_watch_focus', watch)
    prepared = SimpleNamespace(mode='grounded', result=SimpleNamespace(hits=[1], plan=SimpleNamespace(intent='recall')),
                               timings={'retrieve_s': 0.01}, query=field)

    async def prepare(*args, **kwargs):
        return prepared

    async def write(p, emit, **kwargs):
        out = ''
        for c in chunks:
            if not emit(c):
                break
            out += c
        return out
    monkeypatch.setattr(bubble.engine, 'prepare', prepare)
    monkeypatch.setattr(bubble.engine, 'write', write)
    typed = []
    monkeypatch.setattr(service, 'type_text', lambda s, **kwargs: typed.append(s) or True)
    await bubble._run(snap)
    return ''.join(typed), overlay, bubble


async def test_unverified_note_goes_to_pill_not_field(store, monkeypatch):
    typed, overlay, bubble = await run_with_writer(
        store, monkeypatch, [' ' + UNVERIFIED_NOTE + ' ', 'It looks like', ' you saw a post about it.'])
    assert typed == ' It looks like you saw a post about it.'
    assert 'not verified' in overlay.done.call_args.args[0]
    assert bubble.stats['completed'] == 1


async def test_empty_answer_is_a_pill_message_not_typed(store, monkeypatch):
    typed, overlay, bubble = await run_with_writer(store, monkeypatch, [' ' + EMPTY_WRITER_ANSWER])
    assert typed == ''
    overlay.message.assert_called_with('Nothing on file about that', tone='muted')
    assert bubble.stats['empty'] == 1 and bubble.stats['completed'] == 0
