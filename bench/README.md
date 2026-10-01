# Bench snapshots

One JSON file per optimization step, written by `discrete bench --save NAME`
(see [`docs/C7.md`](../docs/C7.md)). Compare any two with
`discrete bench --compare A B`, or one against the current store with
`discrete bench --compare A`.

Naming: `C7-baseline.json` for step 0, then `C8-<change>.json` and so on, each
taken over a comparable real work session.
