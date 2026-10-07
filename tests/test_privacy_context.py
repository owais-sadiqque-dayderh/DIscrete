"""Offline window context privacy regressions."""


def test_invalid_bounds_never_capture_display(monkeypatch):
    from discrete.capture.frames import FrameGrabber
    grabber = FrameGrabber()
    grabber._sck = None
    calls = []
    class FakeScreen:
        monitors = [dict(left=0, top=0, width=100, height=100)]
        def grab(self, region):
            calls.append(region)
            raise RuntimeError('synthetic')
    monkeypatch.setattr(grabber, '_mss', lambda: FakeScreen())
    for bounds in (None, (0, 0, 0, 0), (200, 200, 50, 50), (0, 0, float('nan'), 50)):
        assert grabber.grab(bounds, max_width=100) is None
    assert calls == []


def test_browser_policy_handles_hosts_unknown_and_private(monkeypatch):
    from discrete.capture import privacy
    monkeypatch.setattr(privacy, '_SITES', ('example.invalid',))
    assert privacy.denied('Safari', url='https://WWW.EXAMPLE.INVALID./') == 'site'
    assert privacy.denied('Firefox') == 'unknown_url'
    for app in ('Arc', 'Google Chrome', 'Safari', 'Firefox'):
        assert privacy.denied(app, url='https://allowed.invalid', window_title='Private Browsing') == 'private'
