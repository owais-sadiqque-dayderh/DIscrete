# Encryption at rest

Implemented on `feat/memory-evolution`, opt-in (`DISCRETE_ENCRYPT_AT_REST=false`).
No schema migration is needed. This closes the database/screenshot plaintext
storage issue in `audit_inputs.md` recommendation 3. It does not rotate the
previously leaked provider credential or purge observation 239: those require the
owner's provider account and real data, which this implementation never accessed.

## Options and choice (2026-10-07)

| Option | macOS arm64 / Python 3.13 | Search, migration and performance | Decision |
|---|---|---|---|
| `sqlcipher3` with SQLCipher Community | Current 0.6.3 has a CPython 3.13 macOS arm64 wheel. Installed and tested here; bundles SQLCipher 4.12.0 Community. The wrapper is maintained by Charles Leifer and has a permissive zlib-style license; SQLCipher Community is BSD-3-Clause. | Transparent encrypted SQLite pages preserve FTS5, triggers and ordinary embedding BLOBs. Existing NumPy dense search works unchanged. `sqlcipher_export` handles plaintext/encrypted conversion; explicitly preserve `user_version`. Native encryption adds I/O/CPU overhead, especially for small transactions. | Selected. Optional dependency floor 0.6.2, upper bound <0.7; wheel availability means no fallback is necessary on this host. |
| `sqlcipher3-binary` | Current 0.6.0 release lists Linux x86-64 wheels, including CPython 3.13; no macOS arm64 wheel. Same upstream wrapper lineage; bundled versions can lag. | Same full-database approach, but wrong platform distribution here. | Use `sqlcipher3` rather than this older binary distribution. |
| SQLite SEE | SQLite maintainers supply source to commercial licensees, who build their own binaries. | Full database encryption retains SQLite features, including FTS when compiled in. Native Python binding/build/distribution and license management are additional work. | Viable enterprise alternative; not selected for this branch. |
| Application encryption of sensitive columns | `cryptography` has native macOS wheels. | Encrypted text cannot participate in ordinary FTS5. Leaving token indexes plaintext leaks sensitive terms. Searchable keyed-token indexes change ranking/query semantics. Encrypting embeddings requires decrypting them for NumPy search and still leaves schema/metadata/index leakage. Requires invasive schema and query changes. | Inferior fallback. No silent downgrade; fail closed if SQLCipher is unavailable. A controlled SQLCipher source build or licensed SEE is preferable before redesigning search. |
| Per-file AES-256-GCM via `cryptography` | Installed 50.0.2 here, with an arm64 wheel. | Authenticated encryption of compressed screenshots; decrypt in memory for Pillow/OCR/captioning. Random 96-bit nonce per write; authenticates header, key ID, nonce and ciphertext. | Selected for frames. |

Primary sources: [sqlcipher3 package and wheels](https://pypi.org/project/sqlcipher3/),
[wrapper maintenance](https://github.com/coleifer/sqlcipher3),
[wrapper license](https://github.com/coleifer/sqlcipher3/blob/master/LICENSE),
[SQLCipher license](https://github.com/sqlcipher/sqlcipher/blob/master/LICENSE.md),
[sqlcipher3-binary distributions](https://pypi.org/project/sqlcipher3-binary/),
[SEE distribution/licensing](https://sqlite.org/see/doc/trunk/www/index.wiki),
[SQLCipher export and integrity APIs](https://www.zetetic.net/sqlcipher/sqlcipher-api/),
[AES-GCM API and nonce requirements](https://cryptography.io/en/latest/hazmat/primitives/aead/).

## Keys and file format

A random 32-byte master data key is generated with the OS CSPRNG and stored as a
generic password under Keychain service `org.discrete.encryption.v1`, account a
random UUID. PyObjC Security calls `SecItemAdd` / `SecItemCopyMatching` directly;
there is no secret in command arguments, environment variables or key files.
Creation verifies the stored key before encrypting anything. Access failures
(missing item, locked Keychain, denied ACL or unavailable binding) fail clearly.
Items request `WhenUnlockedThisDeviceOnly` accessibility and disable sync. The
Keychain access query disallows interactive prompts, avoiding an unattended daemon
hanging on a dialog. macOS login-Keychain ACL behavior and device-lock enforcement
must be validated with the final signed application; this branch does not claim
that locking the screen revokes already-open database keys.

HKDF-SHA256 derives independent database and frame subkeys with different domain
strings. SQLCipher receives a raw 256-bit key, avoiding password-based KDF cost.
SQLCipher's normal page authentication remains enabled. Encrypted connections use
in-memory SQLite temporary storage and SQLCipher memory cleanup. Decrypted values
and keys still exist in process memory; Python cannot promise complete zeroization.

`<database>.encryption.json` contains version 1 and the master key UUID. Copy it
with the database. SQLCipher encrypts the database header and pages; WAL contains
encrypted page images. The `-shm` file contains coordination/index metadata.
Frame files keep their original paths (including `.jpg`) so stored references,
retention and purge behavior remain valid. Their payload is:

`DISCRETE-FRAME + version byte | key UUID (16 bytes) | nonce (12 bytes) | AES-GCM ciphertext + tag`.

The envelope adds 59 bytes. Readers detect it without consulting the opt-in flag,
fetch the named Keychain item, authenticate/decrypt and decode via `BytesIO`.
Legacy JPEG/PNG reads remain supported. Frames have owner-only permissions and
are written by durable same-directory replacement. A damaged/wrong-key envelope
does not fall back to plaintext decoding.

Frame writes use the encryption state/key of the owning open store. The runtime
passes that state explicitly through `ScreenObserver`; a custom `Store.open(path)`
keeps its frames encrypted even when the global database path differs or the
opt-in flag is subsequently disabled. `write_frame(..., encryption_state=...)`
requires an explicit state (explicit `None` means a verified plaintext owner).
An absent/unknown owner or unavailable key fails before writing any frame;
the writer never consults the global `db_path()` to choose encryption. The open
store's lease prevents conversion or key rotation while it owns capture writes.

Native Keychain API/accessibility references: [PyObjC Security binding](https://pyobjc.readthedocs.io/en/latest/apinotes/Security.html),
[Apple item accessibility](https://developer.apple.com/documentation/security/ksecattraccessiblewhenunlockedthisdeviceonly),
[Apple query API](https://developer.apple.com/documentation/security/secitemcopymatching(_:_:)).

## Migration, recovery and rotation

Quit DIscrete/Bubble and close other SQLite clients first. `Store.open`, read-only
check connections and frame writes hold shared filesystem locks; conversion takes
an exclusive lock and refuses active DIscrete users. External programs opening
SQLite directly do not honor that lock and must be closed by the owner.

`encrypt`, `decrypt`, and `rotate-key` share one offline conversion path:

1. Unlock and integrity-check the original, checkpoint WAL, select DELETE journal
   mode, enumerate saved/referenced frames inside the selected frame root.
2. Create/verify a fresh Keychain key for encryption/rotation. Persist a durable
   staging journal containing all paths and key IDs, never key bytes.
3. Transform frames individually into same-directory temporary files. Verify
   authenticated round trips and retain original mtimes. Export the complete
   database with SQLCipher, preserving schema/user_version. Reopen it and compare
   every table's row hashes (including FTS shadow tables and vectors), schema and
   integrity/authentication checks. Originals remain in place during staging.
4. Optionally make the requested timestamped original backup, with an owner-only
   directory and `restore.json` mapping each original path to a numbered file.
5. Persist the commit decision and SHA-256 digests, then atomically replace frames,
   database and metadata in sequence. Fsync files/directories and remove the
   journal only after all replacements succeed.

In-place replacement of multiple independent files cannot be one filesystem
atomic operation. The journal prevents DIscrete clients from observing a usable
mixed state. `discrete encrypt --recover` handles every operation: before commit
it removes staged files/partial staging temporaries and leaves originals unchanged;
after commit it verifies staged/already-replaced files and completes the swap.
An interruption in the middle of either direction or rotation is tested with real
child-process `os._exit`, not just a Python exception. Disk/verification failures
before commit clean up immediately; failures after commit retain recovery state.
Manual deletion of the journal/sidecar is unsafe.

Rotation uses a fresh master key, new frame nonces and a complete verified export,
rather than in-place SQLCipher rekey (which cannot cover frames atomically).
Previous Keychain items are retained for copied databases/frames and explicit
backups. Old keys do not open the new database. Losing the original Keychain is
not recoverable from the sidecar or database; replacement keys cannot recover it.

## Synthetic measurements

`tools/measure_encryption_overhead.py` uses only temporary directories and a fake
in-memory Keychain. No replay, paid calls, real data or benchmark harness runs.
Python 3.13.15 arm64; `sqlcipher3` 0.6.3 / SQLCipher 4.12.0 Community;
`cryptography` 50.0.2. Medians of five alternating plaintext/encrypted runs, each
with 3,000 claim inserts, FTS queries, 128-dimensional synthetic vectors, and a
1600×900 noise JPEG (quality 65). Inserts use actual `Store.create_claim`, including
its FTS/history/outbox writes; normal WAL and synchronous=NORMAL settings.

| Operation | Plaintext | Encrypted | Observed difference |
|---|---:|---:|---:|
| Claim insert, per transaction | 267.4 µs | 630.3 µs | +135.7% (+362.9 µs) |
| FTS query, top 20 | 154.8 µs | 198.5 µs | +28.2% (+43.7 µs) |
| Dense query, cached vectors | 149.2 µs | 140.1 µs | −6.1% (noise; arithmetic/cache unchanged) |
| First dense query/load | 6.441 ms | 6.095 ms | −5.4% (noise) |
| Frame bytes read/decrypt | 39.7 µs | 187.4 µs | +371.4% (+147.6 µs) |
| Frame read + Pillow decode | 7.878 ms | 6.696 ms | −15.0% (decode/cache/CPU variance; no claimed speedup) |

These are local wall-clock measurements, **not** a <5% CPU guarantee or native
Keychain latency measurement. Real Keychain lookups, disk sizes, cold caches and
application workloads may cost more. The large relative insert overhead is real
for this synthetic small-transaction workload; absolute insert latency stayed
under 1 ms. No security settings were weakened to improve timings.

## Access-control boundary and remaining work

Encrypted stores auto-detect regardless of the flag. A flag-on plaintext writer
fails and requires explicit conversion; flag-off plaintext uses Python's standard
SQLite driver and never touches Keychain. CLI/daemon/Bubble/replay target stores
all go through the common connector. Enrichment restart, caption backfill and replay
image readers use the in-memory frame reader; retention/purge need no frame key.
Automatic Markdown exports are skipped for encrypted stores; an explicit export
requires `--allow-plaintext`. Bubble no longer logs query text for encrypted stores.

## Changed files and validation

- New implementation: `discrete/encryption.py`, `discrete/encryption_migration.py`.
- Connection integration: `discrete/store/db.py` (only `connect`),
  `discrete/store/base.py`, `discrete/store/index.py` (SQLCipher exception class).
- Frames: `discrete/capture/screen.py`, `enrich.py`, `retention.py`.
- CLI/export/runtime readers: `discrete/cli.py`, `discrete/export.py`,
  `discrete/process/dream.py`, `discrete/bubble/service.py`,
  `discrete/evaluation/invariants.py`, `tools/replay_session.py`,
  `tools/compare_models.py`.
- Read-only audit/export consumers: `bench/audit/{ax_vs_ocr,inputs_stats,
  claims_coverage,caption_novelty,show_obs}.py`, `bench/export_discrete_session.py`.
- Test/probe: `tests/test_encryption.py`, `tools/measure_encryption_overhead.py`.
- Setup/documentation: `pyproject.toml`, `discrete/config.py`, `.env.example`,
  `README.md`, `docs/encryption.md`.

Validation: `.venv/bin/python -m pytest -q` passed **523 tests** (including **20
encryption tests**);
`.venv/bin/ruff check discrete tests bench tools` passed. Encryption tests cover
plaintext compatibility, FTS/dense searches before/after migration/reopen, DB/WAL
absence of sensitive markers, frames, retention without keys, wrong/missing keys,
missing sidecars, active-user locking, both conversion directions, rotation,
backups, injected failures, real process crashes in each direction/rotation,
pending-caption restart, explicit export and CLI error behavior. The native
Keychain API contract uses a stub; the real login Keychain was never accessed.
Focused Pyright reports zero errors with one expected untyped PyObjC `NSData`
warning. No commits, paid calls, replays, benchmark harness runs, private fixtures
or real DIscrete data were used. The timing probe is entirely synthetic.

## Follow-ups

This is encryption at rest, not a claim of complete Recall security:

- Add a signed helper/app and tested user-presence/Touch ID policy for exports,
  debug views and key release, with a clear unattended-daemon policy. A Python
  interpreter/login-Keychain ACL is broader than a dedicated signed executable.
  CLI LocalAuthentication alone would not protect keys from direct same-user
  access; no superficial authentication gate was added.
- Validate actual Keychain creation/ACL/locked-session behavior in an owner-approved
  macOS integration test. This work deliberately used a fake Keychain only.
- Design protected device-to-device key recovery and explicit old/orphan-key
  pruning; device-local keys and retained backup keys have operational tradeoffs.
- Existing Markdown/Git history, `.env`, raw validation manifests, old logs and
  pending-case YAML are not migrated. Decrypt/export/debug stdout intentionally
  expose plaintext. FileVault, restrictive permissions, secret redaction and safe
  backup handling remain necessary. APFS snapshots/SSD remnants of originals are
  not securely erased by atomic replacement.
- Legacy bench scripts that open/copy SQLite directly (`run_g46.py`, `redraft.py`)
  remain plaintext-only. `run_g46.py` was explicitly assigned to another concurrent
  agent and was not edited here. Run them against a deliberate temporary decrypted
  copy; the main Store/replay path supports encrypted data transparently.
- Native SQLCipher or Keychain absence is an actionable error, never a downgrade
  to a partially encrypted database with plaintext FTS indexes.
