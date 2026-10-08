"""Each emitted chunk belongs to the original window and AX field."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from discrete.bubble import service
from discrete.capture.accessibility import FocusSnapshot, WindowContext


def snapshot():
    return FocusSnapshot(pid=7, app='Editor', window_id=11, window_title='Draft',
                         url='https://example.test/a', element=object(), value='Question?')


def test_same_pid_focus_drift_fails_closed(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot()
    ctx = WindowContext(pid=7, app='Editor', window_id=11, window_title='Draft', url=snap.url)
    monkeypatch.setattr(service.ax, 'window_context', lambda: ctx)
    monkeypatch.setattr(service.ax, 'AXUIElementCreateApplication', lambda pid: object(), raising=False)
    element = [snap.element]
    monkeypatch.setattr(service.ax, '_ax_get', lambda *args: element[0])
    assert bubble._focus_matches(snap)
    element[0] = object()
    assert not bubble._focus_matches(snap)
    element[0] = snap.element
    ctx.window_id = 12
    assert not bubble._focus_matches(snap)
    ctx.window_id = 11
    ctx.url = 'https://example.test/b'
    assert not bubble._focus_matches(snap)


@pytest.mark.parametrize('failure', ['error', 'cancel'])
async def test_partial_output_registered_even_when_stream_fails(store, monkeypatch, failure):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot()
    monkeypatch.setattr(FocusSnapshot, 'restore', lambda self: None)
    monkeypatch.setattr(bubble, '_focus_matches', lambda s: True, raising=False)
    async def watch(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(bubble, '_watch_focus', watch)
    prepared = SimpleNamespace(mode='grounded', result=SimpleNamespace(hits=[1], plan=SimpleNamespace(intent='recall')),
                               timings={'retrieve_s': 0.01}, query='Question?')
    async def prepare(*args, **kwargs):
        return prepared
    async def write(p, emit, **kwargs):
        assert emit('First words ')
        if failure == 'cancel':
            raise asyncio.CancelledError
        raise RuntimeError('synthetic stream failure')
    monkeypatch.setattr(bubble.engine, 'prepare', prepare)
    monkeypatch.setattr(bubble.engine, 'write', write)
    monkeypatch.setattr(service, 'type_text', lambda *args, **kwargs: True)
    noted = []
    monkeypatch.setattr(service, 'note_agent_output', noted.append)
    monkeypatch.setattr(service, 'hold', lambda: None)
    monkeypatch.setattr(service, 'release', lambda: None)
    await bubble._run(snap)
    assert noted == ['First words ']


async def test_focus_drift_mid_answer_stops_typing(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot()
    monkeypatch.setattr(FocusSnapshot, 'restore', lambda self: None)
    monkeypatch.setattr(service, 'hold', lambda: None)
    monkeypatch.setattr(service, 'release', lambda: None)
    checks = iter([True, False])
    monkeypatch.setattr(bubble, '_focus_matches', lambda s: next(checks))

    async def watch(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr(bubble, '_watch_focus', watch)
    prepared = SimpleNamespace(mode='grounded', result=SimpleNamespace(hits=[1], plan=SimpleNamespace(intent='recall')),
                               timings={'retrieve_s': 0.01}, query='Question?')

    async def prepare(*args, **kwargs):
        return prepared

    async def write(p, emit, **kwargs):
        out = ''
        for chunk in ('One ', 'two '):
            if not emit(chunk):
                break
            out += chunk
        return out
    monkeypatch.setattr(bubble.engine, 'prepare', prepare)
    monkeypatch.setattr(bubble.engine, 'write', write)
    typed = []
    monkeypatch.setattr(service, 'type_text', lambda s, **kwargs: typed.append(s) or True)
    noted = []
    monkeypatch.setattr(service, 'note_agent_output', noted.append)
    await bubble._run(snap)
    assert typed == ['One '] and noted == ['One ']


def test_denied_destination_fails_closed(store, monkeypatch):
    bubble = service.BubbleService(store, overlay=Mock())
    snap = snapshot()
    ctx = WindowContext(pid=7, app='Editor', window_id=11, window_title='Draft', url=snap.url)
    monkeypatch.setattr(service.ax, 'window_context', lambda: ctx)
    monkeypatch.setattr(service.ax, 'AXUIElementCreateApplication', lambda pid: object(), raising=False)
    monkeypatch.setattr(service.ax, '_ax_get', lambda *args: snap.element)
    monkeypatch.setattr(service, 'denied', lambda *args, **kwargs: 'site')
    assert not bubble._focus_matches(snap)
