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
    for url in ('https://WWW.EXAMPLE.INVALID./', 'https://mail.Example.invalid.:443/x'):
        assert privacy.denied('Safari', url=url) == 'site'
    assert privacy.denied('Firefox') == 'unknown_url'
    assert privacy.denied('Safari', url='https://[bad') == 'unknown_url'
    assert privacy.denied('Safari', url='about:blank') == ''
    assert privacy.denied('Editor') == ''  # not a browser: no URL to know
    for app in ('Arc', 'Google Chrome', 'Safari', 'Firefox'):
        assert privacy.denied(app, url='https://allowed.invalid', window_title='Private Browsing') == 'private'


def test_unknown_url_only_matters_when_sites_are_denied(monkeypatch):
    from discrete.capture import privacy
    monkeypatch.setattr(privacy, '_SITES', ())
    assert privacy.denied('Firefox') == ''


def test_incognito_window_is_asked_once_and_denied(monkeypatch):
    from discrete.capture import browser, privacy
    asked = []
    def osascript(script):
        asked.append(script)
        return 'incognito'
    monkeypatch.setattr(browser, '_osascript', osascript)
    monkeypatch.setattr(browser, '_PRIVATE', browser.collections.OrderedDict())
    browser.note_private('Google Chrome', 41)
    browser.note_private('Google Chrome', 41)
    browser.note_private('Safari', 42)  # no mode to ask
    assert len(asked) == 1
    assert privacy.denied('Google Chrome', url='https://allowed.invalid', window_id=41) == 'private'
    assert privacy.denied('Google Chrome', url='https://allowed.invalid', window_id=43) == ''
