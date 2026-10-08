"""Synthetic, offline regressions for retrieval and Bubble boundaries."""
from types import SimpleNamespace


async def test_bubble_vision_uploads_only_locally_approved_screens(store, monkeypatch):
    """Chromium fields expose no AX text, so Bubble reads them from the window image;
    the image leaves the device only after the on-device OCR finds no secret in it."""
    from PIL import Image

    from discrete.bubble import service
    from discrete.config import Endpoint

    class Grabber:
        def grab(self, *args, **kwargs):
            return Image.new("RGB", (200, 120), "white")

        def close(self):
            pass

    sent = []

    async def complete_json(ep, messages, **kwargs):
        sent.append(messages)
        return {"current_text": "why did I download Helium?", "field_description": "chat box"}

    monkeypatch.setattr(service, "FrameGrabber", Grabber)
    monkeypatch.setattr(service, "complete_json", complete_json)
    monkeypatch.setattr(service, "bubble_vision_endpoint", lambda: Endpoint("bubble_vision", None, "k", "m"))
    monkeypatch.setattr(service.T, "REDACT_SECRETS", True)
    snap = SimpleNamespace(has_text=False, window_bounds=(0, 0, 100, 100), window_id=1, url="")
    bubble = service.BubbleService(store, overlay=SimpleNamespace(status=lambda *a, **k: None))

    monkeypatch.setattr(service.ocr, "ocr_image", lambda img: "Message ChatGPT")
    assert await bubble._read_field(snap) == ("why did I download Helium?", "chat box")
    assert len(sent) == 1

    monkeypatch.setattr(service.ocr, "ocr_image", lambda img: "OPENAI_API_KEY=sk-proj-" + "a1B2c3D4e5F6g7H8" * 3)
    assert await bubble._read_field(snap) == ("", "")

    def unreadable(img):
        raise service.ocr.ExtractionFailed("vision unavailable")

    monkeypatch.setattr(service.ocr, "ocr_image", unreadable)
    assert await bubble._read_field(snap) == ("", "")
    assert len(sent) == 1  # neither the secret screen nor the unread one was uploaded
