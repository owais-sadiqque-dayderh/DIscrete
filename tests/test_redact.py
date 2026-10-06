"""Secret redaction: credential formats are removed, ordinary prose survives."""

from __future__ import annotations

import pytest

from discrete.capture.redact import REDACTED, SecretMemory, redact, redact_url


@pytest.mark.parametrize(
    "text, secret",
    [
        ("export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456", "sk-proj-abcdefghijklmnopqrstuvwxyz123456"),
        ("key sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"),
        ("remote: ghp_abcdefghijklmnopqrstuvwxyz0123456789", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
        ("aws_access_key_id AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
        ("SLACK=xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
        ("Authorization: Bearer abcdefghijklmnop1234567890", "abcdefghijklmnop1234567890"),
        ("password: hunter2hunter", "hunter2hunter"),
        ('{"client_secret": "abcd1234efgh"}', "abcd1234efgh"),
        ("postgres://admin:s3cretpw@db.example.com/app", "admin:s3cretpw"),
        (
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "eyJhbGciOiJIUzI1NiJ9",
        ),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\n-----END OPENSSH PRIVATE KEY-----", "b3BlbnNzaC1rZXk"),
    ],
)
def test_credentials_are_removed(text, secret):
    result = redact(text)
    assert secret not in result.text
    assert REDACTED in result.text
    assert result.count >= 1


def test_the_key_name_survives_so_the_action_is_still_remembered():
    assert redact("OPENAI_API_KEY=abcdef123456").text == f"OPENAI_API_KEY={REDACTED}"


@pytest.mark.parametrize(
    "text",
    [
        "I reviewed the authentication design doc and the token bucket rate limiter.",
        "The password reset flow sends an email.",
        "Author: Jane Smith",
        "task-management-dashboard-component-v2",
        "",
    ],
)
def test_prose_about_secrets_is_left_alone(text):
    assert redact(text).text == text
    assert redact(text).count == 0


def test_urls_lose_credential_parameters_but_keep_the_page():
    result = redact_url("https://app.example.com/oauth/callback?code=abc123&state=xyz")
    assert result.text == f"https://app.example.com/oauth/callback?code={REDACTED}&state=xyz"
    assert result.count == 1
    plain = "https://docs.google.com/document/d/abc/edit?tab=t.0"
    assert redact_url(plain).text == plain and redact_url(plain).count == 0
    assert redact_url("https://u:p4ss@host.example/x").text == f"https://{REDACTED}@host.example/x"


def test_a_full_screen_of_text_redacts_in_linear_time():
    # Screen text reaches 12 000 chars per observation. A nested quantifier in the
    # key pattern once made long hyphenated words take minutes.
    import time

    text = ("task-management-dashboard-component-v2 " * 300 + "x" * 3000)[:12_000]
    started = time.perf_counter()
    redact(text)
    assert time.perf_counter() - started < 0.5


def test_a_value_ocr_split_at_a_space_is_redacted_whole():
    # Real OCR of a TextEdit window: "sk-proj-" and the rest of the key came out apart.
    result = redact("OPENAL_API_KEY=sk-proj- FAKEr2TestKey0987654321ZXWVABC")
    assert "FAKEr2TestKey" not in result.text
    assert result.text == f"OPENAL_API_KEY={REDACTED}"


def test_redaction_reports_what_it_removed():
    result = redact("db password: hunter2secret99 and ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert set(result.secrets) == {"hunter2secret99", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"}
    assert redact_url("https://h.example/cb?code=abc123def").secrets == ("abc123def",)


def test_secret_memory_catches_garbled_copies_and_later_fragments():
    now = [0.0]
    memory = SecretMemory(ttl_s=60, clock=lambda: now[0])
    screen = "OPENAI_API_KEY=sk-proj-FAKEr2TestKey0987654321zyxwvuABC\ndb password: hunter2secret99"
    memory.remember(redact(screen).secrets)

    # OCR's misread copy (O/0, a dropped hyphen) and a password's tail behind a dialog.
    scrubbed = memory.scrub("OPENAILAPLKE sk-proj FAKEr2TestKey0987654321ZXWVABCdb / 18 / secret99 / Save As:")
    assert "FAKEr2" not in scrubbed.text and "secret99" not in scrubbed.text
    assert "OPENAILAPLKE" in scrubbed.text and "Save As:" in scrubbed.text
    assert scrubbed.count == 2

    # Ordinary words that merely share a short piece with a secret survive.
    assert memory.scrub("the secret recipe uses 99 tomatoes").count == 0

    now[0] = 61.0
    assert memory.scrub("secret99").count == 0, "fragments outlived their ttl"


def test_secret_memory_never_holds_the_secret_itself():
    memory = SecretMemory()
    memory.remember(("hunter2secret99",))
    assert "hunter2" not in repr(memory) and all(isinstance(k, bytes) for k in memory._expiry)


def test_hugging_face_tokens_and_unnamed_key_material_are_redacted():
    # Both seen live on an Apple Notes sidebar, the second an OCR-garbled OpenRouter key.
    text = (
        "UI TARS hf_QETGUDGmIUwTzUz4YmdRKPDJTIsn/WCABQ Mona "
        "şk=Xe-60a9031be0kd3e681992e09679c493c78279d0ad279e519e28c874798612e6R V^X"
    )
    out = redact(text).text
    assert "QETGUDG" not in out and "60a9031be0" not in out
    assert out.startswith("UI TARS [REDACTED] Mona")


def test_ordinary_long_words_paths_and_numbers_survive():
    for text in (
        "results in runs/live-suite/final-20260930-202410",
        "www.instagram.com/stories/salmanaltf/4001843049772680208/?r=1",
        "Mona_Lisa_by_Leonardo_da_Vinci_from_C2RMF_retouched",
        "Language Models for Text Classification: From Bag-of-Words to Jev",
    ):
        assert redact(text).text == text


# ---------------------------------------------------------------- in DIscrete's capture


async def test_a_capture_with_a_secret_stores_no_secret_and_no_frame(store, fake, monkeypatch, tmp_path):
    import asyncio
    import time

    from PIL import Image

    from discrete.capture import accessibility as ax
    from discrete.capture import screen as sc

    q: asyncio.Queue = asyncio.Queue()
    obs = sc.ScreenObserver(queue=q)
    obs.dir = tmp_path / "frames"
    obs.dir.mkdir()
    monkeypatch.setattr(obs, "_screen_text",
                        lambda ctx, after, stale=False: ("export OPENROUTER_API_KEY=sk-or-v1-0123456789abcdef0123456789abcdef", "ax"))
    ctx = ax.WindowContext(app="Ghostty", bundle_id="com.mitchellh.ghostty", window_title="zsh — token=abcd1234efgh5678",
                           url="https://example.com/cb?code=abc123", pid=7)
    pending = {"kind": "keyboard", "started": time.time(), "counts": {"keys": 3, "clicks": 0, "scrolls": 0, "clipboard": 0}}

    await obs._capture(pending, ctx, Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64), "white"))

    cap = q.get_nowait()
    stored = " ".join([cap.draft.screen_text, cap.draft.window_title, cap.draft.url])
    assert "sk-or-v1" not in stored and "abcd1234efgh5678" not in stored and "abc123" not in stored
    assert REDACTED in cap.draft.screen_text
    assert cap.draft.frame_path == "" and (cap.job is None or cap.job.frames == [])
    assert obs.stats["redacted"] == 1
    assert list(obs.dir.iterdir()) == []
