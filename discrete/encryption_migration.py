"""Verified offline conversion with durable, restartable atomic file replacements.

A set of files cannot be renamed atomically in place. The journal blocks all normal
store access until every replacement is complete. Before the commit decision a
crash rolls back staging; afterwards recovery rolls forward, including key rotation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .encryption import (
    EncryptionError,
    EncryptionState,
    atomic_write,
    decrypt_frame,
    derive_key,
    encrypt_frame,
    journal_path,
    key_bytes,
    new_key,
    raw_database,
    read_state,
    state_path,
    store_lock,
    sync_dir,
)


@dataclass
class Replacement:
    path: str
    staged: str
    digest: str = ""
    delete: bool = False


@dataclass
class Migration:
    operation: str
    key_id: str | None
    phase: str = "staging"
    files: list[Replacement] = field(default_factory=list)
    backup: str | None = None
    version: int = 1


@dataclass(frozen=True)
class MigrationResult:
    operation: str
    frames: int = 0
    backup: str | None = None
    recovery: str = ""


def _save(path: Path, migration: Migration) -> None:
    atomic_write(journal_path(path), json.dumps(asdict(migration)).encode())


def _load(path: Path) -> Migration:
    try:
        raw = json.loads(journal_path(path).read_text())
        raw["files"] = [Replacement(**item) for item in raw["files"]]
        migration = Migration(**raw)
        if migration.version != 1 or migration.phase not in {"staging", "committing"}:
            raise ValueError
        return migration
    except (ValueError, TypeError, KeyError, OSError):
        raise EncryptionError("Invalid encryption migration journal; keep all files for manual recovery.") from None


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _remove_stage(migration: Migration) -> None:
    for item in migration.files:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(item.staged + suffix).unlink(missing_ok=True)
        # A kill during atomic_write can precede its rename/finally cleanup.
        staged = Path(item.staged)
        for unfinished in staged.parent.glob(staged.name + ".tmp-*"):
            unfinished.unlink()
        sync_dir(staged.parent)


def _finish(path: Path, migration: Migration) -> None:
    # Validate every remaining replacement before swapping any more originals.
    for item in migration.files:
        if item.delete:
            continue
        staged, target = Path(item.staged), Path(item.path)
        candidate = staged if staged.exists() else target
        if not candidate.is_file() or _digest(candidate) != item.digest:
            raise EncryptionError(f"Migration verification failed for {target}; journal retained for recovery.")
    for item in migration.files:
        target, staged = Path(item.path), Path(item.staged)
        if item.delete:
            target.unlink(missing_ok=True)
        elif staged.exists():
            os.replace(staged, target)
        sync_dir(target.parent)
    _remove_stage(migration)
    journal_path(path).unlink()
    sync_dir(path.parent)


def recover_migration(path: Path) -> MigrationResult:
    path = path.expanduser().resolve()
    with store_lock(path, exclusive=True):
        if not journal_path(path).exists():
            raise EncryptionError("No interrupted encryption migration to recover.")
        migration = _load(path)
        if migration.phase == "staging":
            _remove_stage(migration)
            if migration.backup and Path(migration.backup).exists():
                shutil.rmtree(migration.backup)
            journal_path(path).unlink()
            sync_dir(path.parent)
            return MigrationResult(migration.operation, recovery="staging discarded; originals unchanged; rerun the command")
        # Recovery never manufactures keys. Check the committed key still exists.
        if migration.key_id:
            key_bytes(migration.key_id)
        db_item = next(item for item in migration.files if Path(item.path) == path)
        candidate = Path(db_item.staged) if Path(db_item.staged).exists() else path
        check = raw_database(candidate, migration.key_id, read_only=True, force_cipher=True)
        try:
            _verify_database(check, encrypted=migration.key_id is not None)
        finally:
            check.close()
        _finish(path, migration)
        return MigrationResult(migration.operation, backup=migration.backup, recovery="atomic replacements completed")


def _verify_database(conn: Any, *, encrypted: bool) -> None:
    if [row[0] for row in conn.execute("PRAGMA integrity_check")] != ["ok"]:
        raise EncryptionError("SQLite integrity check failed; original files were retained.")
    if encrypted and conn.execute("PRAGMA cipher_integrity_check").fetchall():
        raise EncryptionError("SQLCipher authentication check failed; original files were retained.")


def _table_fingerprints(conn: Any) -> dict[str, str]:
    """Compare all stored values, including FTS shadow tables and embedding bytes."""
    out = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        quoted = '"' + name.replace('"', '""') + '"'
        # Table scan order is not guaranteed across export. Sort row hashes rather
        # than holding decrypted text/vector values in a second large collection.
        hashes = sorted(hashlib.sha256(repr(tuple(row)).encode()).digest() for row in conn.execute(f"SELECT * FROM {quoted}"))
        out[name] = hashlib.sha256(b"".join(hashes)).hexdigest()
    return out


def _export_database(source: Any, staged: Path, key_id: str | None) -> None:
    key = f"x'{derive_key(key_id, b'database').hex()}'" if key_id else ""
    # A decrypt stage contains sensitive plaintext: restrict access BEFORE export,
    # rather than leaving SQLite's default permissions in place while it writes.
    fd = os.open(staged, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    source.execute("ATTACH DATABASE ? AS converted KEY ?", (str(staged), key))
    try:
        source.execute("SELECT sqlcipher_export('converted')")
        version = source.execute("PRAGMA user_version").fetchone()[0]
        source.execute(f"PRAGMA converted.user_version={int(version)}")
    finally:
        source.execute("DETACH DATABASE converted")
    os.chmod(staged, 0o600)
    check = raw_database(staged, key_id, read_only=True, force_cipher=True)
    try:
        _verify_database(check, encrypted=key_id is not None)
        if _table_fingerprints(source) != _table_fingerprints(check):
            raise EncryptionError("Database row verification failed; original files were retained.")
        original_schema = source.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name").fetchall()
        converted_schema = check.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name").fetchall()
        if [tuple(row) for row in original_schema] != [tuple(row) for row in converted_schema]:
            raise EncryptionError("Database schema verification failed; original files were retained.")
    finally:
        check.close()
    with staged.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_dir(staged.parent)


def _frames(source: Any, root: Path) -> list[Path]:
    paths = set(root.rglob("*.jpg")) if root.exists() else set()
    columns = {row[1] for row in source.execute("PRAGMA table_info(observations)")}
    if "frame_path" in columns:
        paths |= {Path(row[0]).expanduser() for row in source.execute("SELECT DISTINCT frame_path FROM observations "
                                                                   "WHERE frame_path != ''") if Path(row[0]).exists()}
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise EncryptionError(f"Frame leaves the selected frame directory: {path}. Select the matching --frames-dir.")
    return sorted(path.resolve() for path in paths)


def migrate_encryption(path: Path, frames_dir: Path, *, operation: str, keep_backup: bool = False) -> MigrationResult:
    if operation not in {"encrypt", "decrypt", "rotate"}:
        raise ValueError(operation)
    if path.is_symlink():
        raise EncryptionError("Migrate the original database path, not a symlink.")
    path, frames_dir = path.expanduser().resolve(), frames_dir.expanduser().resolve()
    with store_lock(path, exclusive=True):
        if journal_path(path).exists():
            raise EncryptionError("Interrupted migration; run this command with --recover first.")
        if not path.is_file():
            raise EncryptionError("Store does not exist; start DIscrete with DISCRETE_ENCRYPT_AT_REST=true to create one.")
        state = read_state(path)
        if operation == "encrypt" and state:
            raise EncryptionError("Store is already encrypted; use `discrete rotate-key` to rotate its key.")
        if operation != "encrypt" and not state:
            raise EncryptionError("Store is plaintext; there is no encryption key to decrypt/rotate.")
        source = raw_database(path, state.key_id if state else None, force_cipher=True)
        migration = None
        try:
            _verify_database(source, encrypted=state is not None)
            checkpoint = source.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0] or source.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise EncryptionError("SQLite is busy. Close all store users before migration.")
            frames = _frames(source, frames_dir)
            key_id = None if operation == "decrypt" else new_key()
            token = uuid.uuid4().hex
            targets = [*frames, path, state_path(path)]
            items = [Replacement(str(target), str(target) + ".discrete-stage-" + token) for target in targets]
            items[-1].delete = key_id is None
            backup = None
            if keep_backup:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                backup = str(path.with_name(path.name + ".backup-" + stamp))
            migration = Migration(operation, key_id, files=items, backup=backup)
            _save(path, migration)  # Persist the complete staging plan before writing any sensitive temporary file.
            for item in items[:-2]:
                target = Path(item.path)
                original = target.read_bytes()
                plain = decrypt_frame(original)
                converted = encrypt_frame(plain, key_id) if key_id else plain
                if decrypt_frame(converted) != plain:
                    raise EncryptionError("Frame round-trip verification failed.")
                staged = Path(item.staged)
                atomic_write(staged, converted)
                stat = target.stat()
                os.utime(staged, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # Preserve retention age.
                with staged.open("rb") as stream:
                    os.fsync(stream.fileno())
                item.digest = _digest(staged)
            db_item = items[-2]
            _export_database(source, Path(db_item.staged), key_id)
            db_item.digest = _digest(Path(db_item.staged))
            if key_id:
                marker = items[-1]
                atomic_write(Path(marker.staged), json.dumps(asdict(EncryptionState(key_id))).encode())
                marker.digest = _digest(Path(marker.staged))
            source.close()
            if backup:
                backup_root = Path(backup)
                backup_root.mkdir(mode=0o700)
                mapping = {}
                for index, target in enumerate(targets):
                    if not target.exists():
                        continue
                    dest = backup_root / str(index)
                    shutil.copy2(target, dest)
                    os.chmod(dest, 0o600)
                    with dest.open("rb") as stream:
                        os.fsync(stream.fileno())
                    mapping[str(target)] = dest.name
                atomic_write(backup_root / "restore.json", json.dumps(mapping, indent=2).encode())
                sync_dir(backup_root.parent)
            migration.phase = "committing"
            _save(path, migration)  # Commit decision: recovery must now roll forward, never restore stale originals.
            _finish(path, migration)
            return MigrationResult(operation, len(frames), backup)
        except BaseException:
            # On a real process crash this cleanup doesn't run; --recover follows
            # the same staged/committing decision from the durable journal.
            if migration is not None and migration.phase == "staging":
                _remove_stage(migration)
                if migration.backup and Path(migration.backup).exists():
                    shutil.rmtree(migration.backup)
                journal_path(path).unlink(missing_ok=True)
                sync_dir(path.parent)
            raise
        finally:
            source.close()
