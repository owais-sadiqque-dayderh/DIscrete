"""Optional SQLCipher and authenticated frame encryption, with Keychain-only keys.

The sidecar and frame headers contain key identifiers, never keys. Keep this module
importable without encryption extras so plaintext evaluation stores stay unchanged.
"""

from __future__ import annotations

import fcntl
import io
import json
import os
import secrets
import sqlite3
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from .config import db_path, encrypt_at_rest

SQLITE_HEADER = b"SQLite format 3\x00"
FRAME_MAGIC = b"DISCRETE-FRAME\x01"
SERVICE = "org.discrete.encryption.v1"


class EncryptionError(RuntimeError):
    """Actionable encryption/key/migration failure; never fall back to plaintext."""


class Keychain(Protocol):
    def get(self, key_id: str) -> bytes: ...
    def put(self, key_id: str, key: bytes) -> None: ...


class MacKeychain:
    """Native Security binding avoids exposing keys in `security -w` process argv."""

    @staticmethod
    def _security() -> Any:
        if sys.platform != "darwin":
            raise EncryptionError("Encryption keys require the macOS Keychain.")
        try:
            import Security
        except ImportError:
            raise EncryptionError("Install encryption extras: uv pip install -e '.[encryption]'") from None
        return Security

    def get(self, key_id: str) -> bytes:
        s = self._security()
        status, data = s.SecItemCopyMatching({
            s.kSecClass: s.kSecClassGenericPassword,
            s.kSecAttrService: SERVICE,
            s.kSecAttrAccount: key_id,
            s.kSecReturnData: True,
            s.kSecMatchLimit: s.kSecMatchLimitOne,
            # An unattended daemon must fail clearly rather than hang on a prompt.
            s.kSecUseAuthenticationUI: s.kSecUseAuthenticationUIFail,
        }, None)
        if status != s.errSecSuccess:
            raise EncryptionError(f"Keychain key {key_id} is missing, locked or access was denied (status {status}). "
                                  "Unlock/restore the original Keychain; a replacement key cannot recover this store.")
        return bytes(data)

    def put(self, key_id: str, key: bytes) -> None:
        s = self._security()
        from Foundation import NSData

        status, _ = s.SecItemAdd({
            s.kSecClass: s.kSecClassGenericPassword,
            s.kSecAttrService: SERVICE,
            s.kSecAttrAccount: key_id,
            s.kSecAttrAccessible: s.kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            s.kSecAttrSynchronizable: False,
            s.kSecValueData: NSData.dataWithBytes_length_(key, len(key)),
        }, None)
        if status != s.errSecSuccess:
            raise EncryptionError(f"Cannot create encryption key in macOS Keychain (status {status}).")
        if self.get(key_id) != key:
            raise EncryptionError("Keychain key verification failed; no data was encrypted.")


_keychain: Keychain = MacKeychain()


def key_bytes(key_id: str) -> bytes:
    try:
        uuid.UUID(hex=key_id)
    except ValueError:
        raise EncryptionError("Invalid encryption key identifier.") from None
    key = _keychain.get(key_id)
    if not isinstance(key, bytes) or len(key) != 32:
        raise EncryptionError(f"Keychain key {key_id} is not a 256-bit key.")
    return key


def new_key() -> str:
    key_id = uuid.uuid4().hex
    key = secrets.token_bytes(32)
    _keychain.put(key_id, key)
    if key_bytes(key_id) != key:
        raise EncryptionError("Keychain key verification failed.")
    return key_id


def derive_key(key_id: str, purpose: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError:
        raise EncryptionError("Install encryption extras: uv pip install -e '.[encryption]'") from None
    # Separate subkeys prevent cross-protocol use of the master data key.
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"discrete.v1/" + purpose).derive(key_bytes(key_id))


@dataclass(frozen=True)
class EncryptionState:
    key_id: str
    version: int = 1


def state_path(path: Path) -> Path:
    return Path(str(path) + ".encryption.json")


def journal_path(path: Path) -> Path:
    return Path(str(path) + ".encryption-migration.json")


def read_state(path: Path) -> EncryptionState | None:
    marker = state_path(path)
    if not marker.exists():
        return None
    try:
        state = EncryptionState(**json.loads(marker.read_text()))
        if state.version != 1 or len(state.key_id) != 32:
            raise ValueError
        uuid.UUID(hex=state.key_id)
        return state
    except (ValueError, TypeError, OSError):
        raise EncryptionError(f"Invalid encryption metadata: {marker}") from None


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """Durable same-directory replacement with owner-only permissions."""
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with open(tmp, "xb") as stream:
            os.chmod(tmp, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def store_lock(path: Path, *, exclusive: bool = False) -> Iterator[BinaryIO]:
    """Held for each connection's lifetime; migration refuses active DIscrete users."""
    stream = open(str(path) + ".encryption.lock", "a+b")
    try:
        fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise EncryptionError("Store is in use or encryption migration is running. Stop DIscrete and retry.") from None
    try:
        yield stream
    finally:
        stream.close()


def cipher_driver() -> Any:
    try:
        from sqlcipher3 import dbapi2
    except ImportError:
        raise EncryptionError("SQLCipher is unavailable. Install encryption extras with "
                              "uv pip install --only-binary=sqlcipher3 -e '.[encryption]'; "
                              "plaintext fallback is disabled.") from None
    return dbapi2


def raw_database(path: Path, key_id: str | None, *, read_only: bool = False,
                 force_cipher: bool = False, factory: Any = None) -> Any:
    driver = cipher_driver() if key_id or force_cipher else sqlite3
    target = path.as_uri() + "?mode=ro" if read_only else str(path)
    kwargs = {"factory": factory} if factory else {}
    conn = driver.connect(target, uri=read_only, isolation_level=None, check_same_thread=False, timeout=30, **kwargs)
    try:
        if key_id:
            key = derive_key(key_id, b"database")
            conn.execute(f"PRAGMA key = \"x'{key.hex()}'\"")
            if not conn.execute("PRAGMA cipher_version").fetchone():
                raise EncryptionError("The installed SQLite driver does not support SQLCipher.")
            conn.execute("PRAGMA cipher_memory_security=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        conn.row_factory = driver.Row
        return conn
    except BaseException as exc:
        conn.close()
        if isinstance(exc, driver.DatabaseError) and key_id:
            raise EncryptionError("Cannot unlock encrypted store: wrong Keychain key or damaged database. "
                                  "No data was changed.") from None
        raise


def _assert_ready(path: Path) -> None:
    if journal_path(path).exists():
        raise EncryptionError("Interrupted encryption migration. Stop DIscrete, then run "
                              "`discrete encrypt --recover` (also works for decrypt/rotation) to finish recovery.")


def open_database(path: str, *, read_only: bool = False) -> Any:
    if path == ":memory:":
        conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn
    target = Path(path).expanduser().resolve()
    if read_only and not target.is_file():
        raise sqlite3.OperationalError("unable to open database file")
    lease = store_lock(target)
    lease.__enter__()
    try:
        _assert_ready(target)
        state = read_state(target)
        exists = target.exists() and target.stat().st_size > 0
        if encrypt_at_rest() and state is None:
            if exists:
                raise EncryptionError("Existing plaintext store: run `discrete encrypt` before enabling "
                                      "DISCRETE_ENCRYPT_AT_REST.")
            # Serialize first-use metadata creation; a crash after it is written reuses
            # the same Keychain item, never creates a replacement for a missing key.
            with store_lock(state_path(target), exclusive=True):
                state = read_state(target)
                if state is None:
                    state = EncryptionState(new_key())
                    atomic_write(state_path(target), json.dumps(asdict(state)).encode())
        if exists:
            with target.open("rb") as stream:
                plaintext = stream.read(16) == SQLITE_HEADER
            if state and plaintext:
                raise EncryptionError("Encryption metadata refers to a plaintext store; restore matching files.")
            if not state and not plaintext:
                raise EncryptionError("Encrypted or invalid SQLite store has no encryption metadata. "
                                      "Restore its .encryption.json sidecar and original Keychain.")
        driver: Any = cipher_driver() if state else sqlite3

        class LeasedConnection(driver.Connection):
            _lease: Any = None
            _discrete_encrypted = bool(state)
            _discrete_operational_error = driver.OperationalError

            def close(self) -> None:
                try:
                    super().close()
                finally:
                    if self._lease is not None:
                        self._lease.__exit__(None, None, None)
                        self._lease = None

            def __del__(self) -> None:
                if self._lease is not None:
                    self.close()

        conn = raw_database(target, state.key_id if state else None, read_only=read_only, factory=LeasedConnection)
        conn._lease = lease
        if state:
            os.chmod(target, 0o600)
        return conn
    except BaseException:
        lease.__exit__(None, None, None)
        raise


def encrypt_frame(data: bytes, key_id: str) -> bytes:
    key = derive_key(key_id, b"frame")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    header = FRAME_MAGIC + bytes.fromhex(key_id) + nonce
    return header + AESGCM(key).encrypt(nonce, data, header)


def decrypt_frame(data: bytes) -> bytes:
    if not data.startswith(FRAME_MAGIC):
        return data
    offset = len(FRAME_MAGIC)
    if len(data) < offset + 16 + 12 + 16:
        raise EncryptionError("Encrypted frame is truncated.")
    key_id = data[offset:offset + 16].hex()
    nonce = data[offset + 16:offset + 28]
    header = data[:offset + 28]
    key = derive_key(key_id, b"frame")
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        return AESGCM(key).decrypt(nonce, data[len(header):], header)
    except InvalidTag:
        raise EncryptionError("Cannot unlock encrypted frame: wrong Keychain key or damaged frame.") from None


def read_frame(path: str | Path) -> bytes:
    return decrypt_frame(Path(path).read_bytes())


def open_frame(path: str | Path) -> Any:
    """Pillow decodes plaintext bytes in memory; no decrypted file is created."""
    from PIL import Image

    return Image.open(io.BytesIO(read_frame(path)))


def write_frame(path: Path, image: Any, *, quality: int) -> None:
    target = db_path().expanduser().resolve()
    with store_lock(target):
        _assert_ready(target)
        state = read_state(target)
        if encrypt_at_rest() and state is None:
            raise EncryptionError("Open/create the encrypted store before saving frames, or run `discrete encrypt`.")
        data: BinaryIO = io.BytesIO()
        image.save(data, "JPEG", quality=quality)
        payload = data.getvalue()
        if state:
            payload = encrypt_frame(payload, state.key_id)
        atomic_write(path, payload)
