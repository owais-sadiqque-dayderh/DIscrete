# Release checks

`discrete check` remains a read-only legacy audit: missing measurements are shown
as `n/a`, and are not failures in that mode. Use **`discrete check --release`** for
release decisions. It requires every check to pass, full live-source embedding
coverage, an empty embedding outbox, and valid samples for all three latency
stages. Release SLOs use only `src=live`; replay, process, and evaluation samples
cannot establish daemon latency. A single sample establishes measurement
availability, not statistical confidence or soak-test coverage.

`discrete eval` returns nonzero for stale case snapshots, incompatible baseline
snapshots, empty case sets, regressions, or incorrect cases. A stale run cannot
save a new baseline. Snapshot ids now hash store contents, schema and resolved
configuration; old count-based snapshot ids must be regenerated against reviewed
inputs. Measurements and query diagnostics are excluded from input identity.
The snapshot is checked again after the suite to detect concurrent input changes.

The benchmark records schema/content, answering revision/dirty state, resolved
nonsecret model settings, tunables, and case/judge-prompt hashes. Arbitrary provider
bodies and endpoint URLs are hashed; credentials are omitted. An encrypted input
is copied as an encrypted database, with its key-id sidecar, using the same
Keychain identity. No automatic plaintext benchmark export is performed.

These checks establish deterministic invariants and measured availability. The
judge still sees returned evidence and expected strings: it does not independently
establish source truth, complete list membership, calibration, or generalization
across unseen sessions. A release requires independently annotated source truth,
complete sets, session-disjoint holdouts, installed-artifact checks, and native/
long-running tests in addition to these gates. No such quality or deployment
validation was performed by this offline hardening task.

Cost totals include only reported prices. Unknown prices and dropped call samples
make the total a lower bound; a missing price never establishes a free call.
Telemetry writes use atomic, idempotent batches retained in memory on failure.
This avoids loss on a failed flush and double counting after cancellation at
commit, but an abrupt process/host crash can still lose unflushed measurements.
