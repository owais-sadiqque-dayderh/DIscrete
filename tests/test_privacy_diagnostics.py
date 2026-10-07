"""Diagnostics never print or send the user's screen content."""
from argparse import Namespace

from PIL import Image, ImageDraw

from discrete.capture import accessibility as ax
from discrete.capture import frames
from discrete.cli import cmd_doctor

SECRET_TITLE = 'Synthetic private title 4242'


def _front_window(monkeypatch, app='Editor'):
    ctx = ax.WindowContext(app=app, window_title=SECRET_TITLE, pid=7, bounds=(0, 0, 200, 200), window_id=3)
    monkeypatch.setattr(ax, 'window_context', lambda: ctx)
    monkeypatch.setattr(ax, 'accessibility_granted', lambda: True)
    monkeypatch.setattr(ax, 'read_screen_text', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('text read')))
    image = Image.new('RGB', (200, 200), 'white')
    ImageDraw.Draw(image).text((10, 10), 'synthetic secret on screen', fill='black')
    monkeypatch.setattr(frames.FrameGrabber, 'grab', lambda self, *a, **kw: image)


async def test_doctor_checks_capture_without_showing_content(monkeypatch, capsys):
    _front_window(monkeypatch)
    assert await cmd_doctor(Namespace(models=False)) == 0
    out = capsys.readouterr().out
    assert 'screen capture: OK' in out
    assert SECRET_TITLE not in out and 'Editor' not in out and 'sample' not in out.lower()


async def test_doctor_does_not_grab_a_denied_window(monkeypatch, capsys):
    _front_window(monkeypatch, app='1Password')
    monkeypatch.setattr(frames.FrameGrabber, 'grab', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('grabbed')))
    await cmd_doctor(Namespace(models=False))
    assert 'not checked' in capsys.readouterr().out


async def test_model_check_sends_only_synthetic_content(monkeypatch):
    from discrete.evaluation import model_check
    sent = []
    async def complete_json(ep, messages, **kw):
        sent.append(messages)
        raise ValueError('offline')
    async def offline(*a, **kw):
        raise model_check.NotConfigured('offline')
    monkeypatch.setattr(model_check, 'complete_json', complete_json)
    for name in ('embed', 'rerank'):
        monkeypatch.setattr(model_check, name, offline)
    async def stream(*a, **kw):
        raise model_check.NotConfigured('offline')
        yield ''
    monkeypatch.setattr(model_check, 'stream', stream)
    _front_window(monkeypatch)
    await model_check.check_models()
    assert sent and SECRET_TITLE not in repr(sent) and 'Editor' not in repr(sent)
