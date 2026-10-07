"""No real Keychain, data, capture, models or replay: all artifacts live in tmp_path."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from discrete import encryption as enc
from discrete import encryption_migration as migration
from discrete.capture.retention import purge_frames
from discrete.models import ClaimDraft, ObservationDraft
from discrete.store import Store
from discrete.store.db import connect

pytest.importorskip("sqlcipher3")
pytest.importorskip("cryptography")


class FakeKeychain:
    def __init__(self):
        self.keys: dict[str, bytes] = {}

    def get(self, key_id: str) -> bytes:
        if key_id not in self.keys:
            raise enc.EncryptionError("Keychain key is missing or locked")
        return self.keys[key_id]

    def put(self, key_id: str, key: bytes) -> None:
        assert key_id not in self.keys
        self.keys[key_id] = key


@pytest.fixture
def keys(monkeypatch):
    backend = FakeKeychain()
    monkeypatch.setattr(enc, "_keychain", backend)
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "false")
    return backend


def make_plain(tmp_path: Path) -> Path:
    path = tmp_path / "store.db"
    conn = connect(str(path))
    conn.execute("INSERT INTO meta VALUES ('secret', 'sensitive_unicorn_marker')")
    conn.close()
    return path


def make_frame(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    path = root / "test.jpg"
    Image.new("RGB", (32, 24), "red").save(path, "JPEG")
    return path


async def test_encrypted_store_roundtrip_search_and_wal(tmp_path, keys, monkeypatch, fake):
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "true")
    path = tmp_path / "store.db"
    store = await Store.open(str(path))
    cid = await store.create_claim(ClaimDraft(kind="fact", text="sensitive_unicorn_marker prefers violet notebooks"))
    oid, _ = await store.add_observation(ObservationDraft(app="Code", window_title="sensitive_unicorn_marker.txt",
                                                         screen_text="violet notebooks", session_id="test"))
    pages = await store.list_pages()
    assert len(pages) == 1 and pages[0]["title"] == "sensitive_unicorn_marker.txt"
    await store.save_embeddings([("claim", cid, [1.0, 0.0], "hash")])
    assert cid in await store.fts_ids("claim", "violet")
    assert oid in await store.fts_ids("observation", "violet")
    assert (await store.dense_search("claim", [1.0, 0.0], limit=1))[0][0] == cid
    assert await store.fts_ids("claim", '"') == []
    assert (await store.q1("PRAGMA temp_store"))[0] == 2
    for suffix in ("", "-wal"):
        payload = await asyncio.to_thread(Path(str(path) + suffix).read_bytes)
        assert b"sensitive_unicorn_marker" not in payload
        assert b"violet" not in payload
    await store.close()
    # Ciphertext detection doesn't depend on the flag remaining on.
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "false")
    store = await Store.open(str(path))
    assert cid in await store.fts_ids("claim", "violet")
    assert (await store.list_pages())[0]["canonical_url"] == "doc://code/sensitive_unicorn_marker.txt"
    assert (await store.dense_search("claim", [1.0, 0.0], limit=1))[0][0] == cid
    await store.close()
    assert not path.read_bytes().startswith(enc.SQLITE_HEADER)
    with pytest.raises(sqlite3.DatabaseError):
        conn = sqlite3.connect(path)
        try:
            conn.execute("SELECT * FROM claims")
        finally:
            conn.close()


def test_plaintext_uses_standard_sqlite_and_never_keychain(tmp_path, keys):
    path = make_plain(tmp_path)
    assert path.read_bytes().startswith(enc.SQLITE_HEADER)
    assert keys.keys == {}
    conn = connect(str(path), read_only=True)
    assert conn.execute("SELECT value FROM meta WHERE key='secret'").fetchone()[0] == "sensitive_unicorn_marker"
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM meta")
    conn.close()


def test_plaintext_flag_does_not_implicitly_migrate(tmp_path, keys, monkeypatch):
    path = make_plain(tmp_path)
    before = path.read_bytes()
    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "true")
    with pytest.raises(enc.EncryptionError, match="discrete encrypt"):
        connect(str(path))
    assert path.read_bytes() == before
    assert keys.keys == {}


async def test_migration_both_ways_preserves_search_schema_frames_and_age(tmp_path, keys, fake):
    path = make_plain(tmp_path)
    root = tmp_path / "frames"
    frame = make_frame(root)
    original = frame.read_bytes()
    old = time.time() - 7 * 86400
    os.utime(frame, (old, old))
    store = await Store.open(str(path))
    cid = await store.create_claim(ClaimDraft(kind="fact", text="sensitive_unicorn_marker violet notebooks"))
    await store.save_embeddings([("claim", cid, [1.0, 0.0], "hash")])
    await store.add_observation(ObservationDraft(screen_text="violet", frame_path=str(frame), session_id="test"))
    version = await store.scalar("PRAGMA user_version")
    await store.close()
    result = migration.migrate_encryption(path, root, operation="encrypt")
    assert result.frames == 1 and result.backup is None
    assert enc.read_frame(frame) == original
    assert frame.read_bytes().startswith(enc.FRAME_MAGIC)
    assert frame.stat().st_mtime == old
    store = await Store.open(str(path))
    assert cid in await store.fts_ids("claim", "violet")
    assert (await store.dense_search("claim", [1.0, 0.0], limit=1))[0][0] == cid
    assert await store.scalar("PRAGMA user_version") == version
    await store.close()
    migration.migrate_encryption(path, root, operation="decrypt")
    assert not enc.state_path(path).exists()
    assert path.read_bytes().startswith(enc.SQLITE_HEADER)
    assert frame.read_bytes() == original and frame.stat().st_mtime == old
    assert not list(tmp_path.rglob("*.discrete-stage-*"))
    assert not list(tmp_path.glob("*.backup-*"))


def test_rotation_backup_and_old_key_cannot_open_new_database(tmp_path, keys):
    path = make_plain(tmp_path)
    root = tmp_path / "frames"
    frame = make_frame(root)
    original = frame.read_bytes()
    report = migration.migrate_encryption(path, root, operation="encrypt", keep_backup=True)
    assert report.backup and Path(report.backup).is_dir()
    assert (Path(report.backup) / "1").read_bytes().startswith(enc.SQLITE_HEADER)
    old = enc.read_state(path).key_id
    migration.migrate_encryption(path, root, operation="rotate")
    new = enc.read_state(path).key_id
    assert new != old
    assert enc.read_frame(frame) == original
    with pytest.raises(enc.EncryptionError, match="wrong Keychain key"):
        enc.raw_database(path, old, read_only=True)
    conn = connect(str(path))
    conn.close()


def test_missing_wrong_keys_no_changes_and_no_regeneration(tmp_path, keys):
    path = make_plain(tmp_path)
    migration.migrate_encryption(path, tmp_path / "frames", operation="encrypt")
    key_id = enc.read_state(path).key_id
    original = path.read_bytes()
    keys.keys.pop(key_id)
    with pytest.raises(enc.EncryptionError, match="missing"):
        connect(str(path))
    assert not keys.keys and path.read_bytes() == original
    keys.keys[key_id] = b"x" * 32
    with pytest.raises(enc.EncryptionError, match="wrong Keychain key"):
        connect(str(path))
    assert path.read_bytes() == original


def test_frame_write_read_tamper_retention_and_plaintext_compatibility(tmp_path, keys, monkeypatch):
    path = make_plain(tmp_path)
    root = tmp_path / "frames"
    frame = make_frame(root)
    monkeypatch.setenv("DISCRETE_DB", str(path))
    monkeypatch.setenv("DISCRETE_DATA_DIR", str(tmp_path))
    migration.migrate_encryption(path, root, operation="encrypt")
    enc.write_frame(frame, Image.new("RGB", (32, 24), "blue"), quality=65, encryption_state=enc.read_state(path))
    with enc.open_frame(frame) as image:
        assert image.size == (32, 24) and image.getpixel((0, 0))[2] > 240
    payload = frame.read_bytes()
    key_id = enc.read_state(path).key_id
    key = keys.keys[key_id]
    keys.keys[key_id] = b"z" * 32
    with pytest.raises(enc.EncryptionError, match="wrong Keychain key"):
        enc.read_frame(frame)
    keys.keys[key_id] = key
    frame.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(enc.EncryptionError, match="damaged frame"):
        enc.read_frame(frame)
    # Deleting expired frames needs no Keychain access or plaintext materialization.
    keys.keys.clear()
    old = time.time() - 7 * 86400
    os.utime(frame, (old, old))
    assert purge_frames(ttl_days=3) == 1 and not frame.exists()
    plain = make_frame(root)
    assert enc.read_frame(plain) == plain.read_bytes()


def test_migration_refuses_active_connections(tmp_path, keys):
    path = make_plain(tmp_path)
    conn = connect(str(path))
    with pytest.raises(enc.EncryptionError, match="in use"):
        migration.migrate_encryption(path, tmp_path / "frames", operation="encrypt")
    conn.close()
    migration.migrate_encryption(path, tmp_path / "frames", operation="encrypt")


def test_failure_before_commit_keeps_originals(tmp_path, keys, monkeypatch):
    path = make_plain(tmp_path)
    root = tmp_path / "frames"
    frame = make_frame(root)
    original = frame.read_bytes()

    def fail(*args):
        raise OSError("injected disk failure")

    monkeypatch.setattr(migration, "_export_database", fail)
    with pytest.raises(OSError, match="disk failure"):
        migration.migrate_encryption(path, root, operation="encrypt")
    assert path.read_bytes().startswith(enc.SQLITE_HEADER) and frame.read_bytes() == original
    assert not enc.journal_path(path).exists() and not enc.state_path(path).exists()
    assert not list(tmp_path.rglob("*.discrete-stage-*"))


@pytest.mark.parametrize("phase", ["staging", "committing"])
@pytest.mark.parametrize("operation", ["encrypt", "decrypt", "rotate"])
def test_process_crash_and_recovery(tmp_path, keys, monkeypatch, phase, operation):
    path = make_plain(tmp_path)
    root = tmp_path / "frames"
    frame = make_frame(root)
    original = frame.read_bytes()
    monkeypatch.setattr(enc.secrets, "token_bytes", lambda n: b"k" * n)
    if operation != "encrypt":
        migration.migrate_encryption(path, root, operation="encrypt")
    # The child fake Keychain can be reconstructed in memory, with no key file.
    script = '''
import os, sys
from pathlib import Path
from discrete import encryption as e, encryption_migration as m
class Keys:
    def get(self, key_id): return b'k' * 32
    def put(self, key_id, key): pass
e._keychain = Keys()
e.secrets.token_bytes = lambda n: b'k' * n
path, root, phase, operation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
if phase == 'staging':
    m._export_database = lambda *args: os._exit(78)
else:
    replace = os.replace
    def crash(src, dst):
        replace(src, dst)
        if Path(dst) == root / 'test.jpg': os._exit(78)
    os.replace = crash
m.migrate_encryption(path, root, operation=operation)
'''
    result = subprocess.run([sys.executable, "-c", script, str(path), str(root), phase, operation],
                            env={**os.environ, "DISCRETE_DATA_DIR": str(tmp_path), "DISCRETE_ENCRYPT_AT_REST": "false"},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 78, result.stderr
    with pytest.raises(enc.EncryptionError, match="Interrupted"):
        connect(str(path))
    pending = migration._load(path)
    if pending.key_id:
        keys.keys[pending.key_id] = b"k" * 32
    report = migration.recover_migration(path)
    assert report.recovery and not enc.journal_path(path).exists()
    assert enc.read_frame(frame) == original
    conn = connect(str(path))
    assert conn.execute("SELECT value FROM meta WHERE key='secret'").fetchone()[0] == "sensitive_unicorn_marker"
    conn.close()
    plaintext = (operation == "encrypt") if phase == "staging" else (operation == "decrypt")
    assert path.read_bytes().startswith(enc.SQLITE_HEADER) == plaintext


async def test_pending_caption_reads_encrypted_frame_in_memory(tmp_path, keys, monkeypatch, fake):
    from discrete.capture.enrich import Enricher

    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "true")
    path = tmp_path / "store.db"
    monkeypatch.setenv("DISCRETE_DB", str(path))
    store = await Store.open(str(path))
    root = tmp_path / "frames"
    root.mkdir()
    frame = root / "pending.jpg"
    enc.write_frame(frame, Image.new("RGB", (32, 24), "red"), quality=65, encryption_state=store.frame_encryption)
    oid, _ = await store.add_observation(ObservationDraft(app="Code", screen_text="reading notes", session_id="test",
                                                        frame_path=str(frame), enrich_state="pending"))
    enricher = Enricher(store)
    try:
        assert await enricher.fill_pending() == 1
        await asyncio.gather(*enricher._tasks)
        assert (await store.get_observation(oid))["enrich_state"] == "done"
        assert "caption" in fake.calls
        assert frame.read_bytes().startswith(enc.FRAME_MAGIC)
        assert list(root.iterdir()) == [frame]
    finally:
        await enricher.cancel_all()
        await store.close()


async def test_export_requires_explicit_plaintext_opt_in(tmp_path, keys, monkeypatch, fake):
    from discrete.cli import _refresh_export
    from discrete.export import export_markdown
    from discrete.process.dream import run_dream

    monkeypatch.setenv("DISCRETE_ENCRYPT_AT_REST", "true")
    store = await Store.open(str(tmp_path / "store.db"))
    try:
        with pytest.raises(enc.EncryptionError, match="allow-plaintext"):
            await export_markdown(store, tmp_path / "memory", git=False)
        assert not (tmp_path / "memory").exists()
        dream = await run_dream(store)
        assert "skipped" in dream["export"]
        await _refresh_export(store)
        assert not (tmp_path / "data" / "memory").exists()
        await export_markdown(store, tmp_path / "memory", git=False, allow_plaintext=True)
        assert (tmp_path / "memory" / "MEMORY.md").exists()
    finally:
        await store.close()


def test_cli_conversion_rotation_and_clear_key_error(tmp_path, keys, capsys, fake, monkeypatch):
    from discrete.cli import main
    from discrete.perf import PERF

    path = make_plain(tmp_path)
    monkeypatch.setenv("DISCRETE_DB", str(path))
    monkeypatch.setattr(PERF, "source", PERF.source)
    frame = make_frame(tmp_path / "frames")
    for command in ("encrypt", "rotate-key", "decrypt", "encrypt"):
        with pytest.raises(SystemExit) as exit_code:
            main(["--db", str(path), command])
        assert exit_code.value.code == 0
        assert frame.read_bytes().startswith(enc.FRAME_MAGIC) == (command != "decrypt")
    keys.keys.clear()
    with pytest.raises(SystemExit) as exit_code:
        main(["--db", str(path), "stats"])
    assert exit_code.value.code == 2
    assert "encryption:" in capsys.readouterr().err


def test_missing_sidecar_never_opens_as_plaintext(tmp_path, keys):
    path = make_plain(tmp_path)
    migration.migrate_encryption(path, tmp_path / "frames", operation="encrypt")
    enc.state_path(path).unlink()
    before = path.read_bytes()
    with pytest.raises(enc.EncryptionError, match="no encryption metadata"):
        connect(str(path))
    assert path.read_bytes() == before


def test_native_keychain_api_contract_without_accessing_real_keychain(monkeypatch):
    from types import SimpleNamespace

    constants = ["kSecClass", "kSecClassGenericPassword", "kSecAttrService", "kSecAttrAccount", "kSecReturnData",
                 "kSecMatchLimit", "kSecMatchLimitOne", "kSecUseAuthenticationUI", "kSecUseAuthenticationUIFail",
                 "kSecAttrAccessible", "kSecAttrAccessibleWhenUnlockedThisDeviceOnly", "kSecAttrSynchronizable", "kSecValueData"]
    security = SimpleNamespace(**{name: name for name in constants}, errSecSuccess=0)
    queries = []

    def add(query, result):
        queries.append(query)
        return 0, None

    security.SecItemAdd = add
    security.SecItemCopyMatching = lambda query, result: (0, b"k" * 32)
    monkeypatch.setattr(enc.MacKeychain, "_security", staticmethod(lambda: security))
    backend = enc.MacKeychain()
    backend.put("1" * 32, b"k" * 32)
    assert bytes(queries[0][security.kSecValueData]) == b"k" * 32
    assert queries[0][security.kSecAttrSynchronizable] is False
    security.SecItemCopyMatching = lambda query, result: (-25308, None)
    with pytest.raises(enc.EncryptionError, match="missing, locked or access was denied"):
        backend.get("1" * 32)
    security.SecItemAdd = lambda query, result: (-25299, None)
    with pytest.raises(enc.EncryptionError, match="Cannot create"):
        backend.put("1" * 32, b"k" * 32)
