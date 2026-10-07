"""Synthetic, offline regressions for retrieval and Bubble boundaries."""
from types import SimpleNamespace


async def test_bubble_unapproved_vision_never_captures_or_uploads(store, fake, monkeypatch):
    from discrete.bubble import service

    def forbidden(*args, **kwargs):
        raise AssertionError("unapproved image capture/upload")

    monkeypatch.setattr("discrete.capture.frames.FrameGrabber", forbidden)
    monkeypatch.setattr("discrete.llm.complete_json", forbidden)
    snap = SimpleNamespace(has_text=False, window_bounds=(0, 0, 100, 100), window_id=1)
    assert await service.BubbleService(store)._read_field(snap) == ("", "")
