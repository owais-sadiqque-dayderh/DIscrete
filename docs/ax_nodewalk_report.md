# AX node-walk cleanup: live Arc verification

Measured 2026-10-07 on this Mac, branch `feat/memory-evolution`. This follows
[the marker experiment](ax_text_markers_report.md) and implements
[audit recommendation #4](audit_inputs.md#4-ranked-recommendations), in the node
walk used by browsers. No commit was made.

## Result

The median of eight page latency medians fell **209.0 → 34.0 ms**.
Every page's median improved. The final 56 after reads hit **zero** walk deadlines,
versus **28/56** before; the slowest after read was **189.6 ms** under the existing
250 ms soft deadline. No read was empty. After text was identical across all seven
reads of each page.

Hacker News fell **3,204 → 2,553 chars**, **72 → 40 lines**, and **18.1% → 0.0%**
duplicate lines, while covering the same **98.4%** of OCR content-token types.
Its median latency fell **250.6 → 125.3 ms**. Its screenshot OCR has **2,410 chars**.
The earlier marker report's 4,748-char/19.1%-duplicate read is historical evidence,
not this run's baseline: both feed content and deadline-truncated output vary.

The population table now emits country, rank, population, percentage, date,
source and notes on one row. Its lines fell **86 → 45**, single-word share
**55.8% → 37.8%**, and latency **250.5 → 85.2 ms**, with unchanged OCR token support
and coverage. Example:

```text
1 | India | 1,429,404,000 | 17.3% | 1 Jul 2026 | Official projection [ 4 ] | [ b ]
```

## Implementation

- Keep container aria/title suppression, and delay container value/title reads
  until descendants fail to provide text. This eliminates unnecessary native
  attribute round trips on normal web containers.
- For `AXTable`/`AXGrid` with rows, skip the alternative `AXColumn` projections
  within that table. Live HN exposed the same headline cells through rows and
  columns. Preserve a table exposing only columns.
- Deduplicate within **eight neighbouring emissions**, requiring equal text
  after whitespace normalization plus the same AX element or matching frames
  within two points. Do not deduplicate distinct cells. A radio/checkbox aria
  title and adjacent static label can also collapse, but require the same
  parent/group (at most one wrapper), line alignment and a gap of at most 32
  points. Preserve repeated words/comments at separate positions and repetitions
  without reliable geometry; remove the old unconditional adjacent-line filter.
- Read `AXPosition` + `AXSize`, with `AXFrame` fallback. Intersect window bounds
  with nested web/scroll viewports; continue to reject Chromium's unreadable
  one-pixel strips. When no viewport is supplied, derive it from the window.
- Offscreen or unlocated text is retained only as a fallback tail when the
  formatted visible result is **under 80 chars**. The tail is limited to **256
  chars**, collected from at most **eight nonempty node emissions**. Once the
  fallback is full, prune offscreen branches rather than spend the node/deadline
  budget on their descendants. No second walk or new setting is introduced.
- Assemble semantic rows from `AXCell`/grid/header roles, keeping empty cells
  and joining descendant text inside each cell. Live Wikipedia exposes ranks
  as direct `AXGroup` children of rows, which are handled as cells too; wrappers
  around semantic cells do not add phantom columns. AX row/column spans and
  columns omitted by the accessibility adapter are not expanded.

## Paired live measurements

Each value is **before / after**. Percentages are per page, not pooled. Text
metrics describe the read at median latency, matching the earlier report.

| Page/state | Chars | Lines | Single-word % | Duplicate % | OCR-supported % | OCR content coverage % | Median ms | Deadline reads (of 7) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAS article, scrolled body | 1253 / 1253 | 20 / 20 | 30.0 / 30.0 | 0.0 / 0.0 | 94.7 / 94.7 | 96.0 / 96.0 | 36.6 / 14.6 | 0 / 0 |
| GitHub Requests README | 1160 / 1160 | 40 / 40 | 22.5 / 22.5 | 0.0 / 0.0 | 89.9 / 89.9 | 81.5 / 81.5 | 167.8 / 37.6 | 0 / 0 |
| Hacker News feed | 3204 / 2553 | 72 / 40 | 26.4 / 2.5 | 18.1 / 0.0 | 99.5 / 99.6 | 98.4 / 98.4 | 250.6 / 125.3 | 7 / 0 |
| Wikipedia Accessibility | 2398 / 2398 | 67 / 67 | 37.3 / 37.3 | 1.5 / 1.5 | 85.4 / 85.4 | 95.2 / 95.2 | 250.4 / 56.9 | 7 / 0 |
| Python tutorial, Fibonacci code | 37 / 1554 | 2 / 51 | 50.0 / 15.7 | 0.0 / 17.6 | 100.0 / 100.0 | 2.7 / 96.4 | 250.3 / 29.7 | 7 / 0 |
| Google, collapsed overview | 1536 / 1536 | 37 / 37 | 18.9 / 18.9 | 5.4 / 5.4 | 69.9 / 69.9 | 91.0 / 91.0 | 96.1 / 30.4 | 0 / 0 |
| Wikipedia population table | 1851 / 1971 | 86 / 45 | 55.8 / 37.8 | 3.5 / 4.4 | 98.1 / 98.1 | 95.5 / 95.5 | 250.5 / 85.2 | 7 / 0 |
| Guardian, browser context menu overlay | 1044 / 1044 | 30 / 30 | 30.0 / 30.0 | 0.0 / 0.0 | 84.2 / 84.2 | 81.4 / 81.4 | 90.2 / 21.9 | 0 / 0 |

The Python baseline frequently expired before reaching the viewport code.
Its seven character counts ranged **37–1,235**; the median-latency read contained
only the browser's two chrome lines. After returns **1,554 chars** consistently,
covering **96.4%** of OCR content-token types. Its **17.6%** duplicate-line metric
includes real repeated code output (`1` twice), continuation prompts and list
bullets. These remain deliberately: globally reducing duplicate-line share
would damage content. The table's 3.5% → 4.4% share likewise does not imply new
repeated text: row grouping reduces the denominator, while genuine repeated
sidebar headings remain.

Hacker News baseline counts ranged **2,472–3,204** because the walk reached its
deadline during the column projection. Other final baseline lengths were fixed.

### Method and state differences

- Same eight public URLs as the marker report: article, README, feed, Wikipedia
  prose, docs/code, search, table and an overlay case. Arc PID 46177; each page
  used its own temporary Little Arc window. `cua-driver launch_app` opened them
  in the background. Only the CAS test window was scrolled. Existing tabs were
  not navigated or selected.
- The Guardian's old **support overlay did not reappear** in the fresh window.
  The new overlay case is its **browser context menu** over the same public
  article, opened by the background AX show-menu action. It is not a reproduction
  of the prior support-overlay result; no subscription, login or sharing action
  was completed. The screenshot and source text for that state remain in scratch.
- Baseline source was copied before edits to
  `/tmp/discrete-ax-nodewalk-20261007/before.py`; the measurement script loads it
  as a separate module alongside the final implementation. Marker mode was false
  for both. No configuration file was changed.
- Called production `read_screen_text` directly with **4,000 nodes, 250 ms,
  16,000 chars**, window bounds **(20, 50, 1368, 727)**. The experiment intercepts
  only `AXFocusedWindow` to return the exact temporary AX window, verified by
  title and bounds against WindowServer, so reading a background window does not
  require changing the user's selected window/tab. Every node, frame, value,
  title and parameterized read remains a real AX call. AX trust was verified.
- Discarded one warmup per path, then interleaved **seven reads per path**,
  reversing order on alternate rounds. Final timing measurements ran without
  pytest alongside them. Timings include the production read, exclude screenshot,
  OCR, load/navigation and diagnostic tree walks. Health counters were drained
  only in memory.
- `cua-driver get_window_state` captured the verified window. Its current image
  ceiling returned **1567 × 833** screenshots, unlike the native **2736 × 1454**
  screenshots in the earlier report. Both paths use the same screenshot and
  Apple Vision `ocr_image`; before/after comparisons are paired, but OCR numbers
  must not be treated as directly comparable to that earlier native-pixel run.
  Screenshots were visually inspected for body text, README, Fibonacci code,
  table rows, collapsed search overview and overlay state.
- Chars are Python string length; lines are nonempty stripped lines;
  single-word share uses `textkit.tokens`; duplicate share uses `textkit.norm_key`.
  OCR-supported share counts AX content-token occurrences found in the OCR token
  set; coverage counts OCR content-token types found in AX. These are visibility
  proxies, not positioned character recall. Repeated/offscreen words already
  visible elsewhere can inflate support, and OCR omissions reduce it.
- The existing soft deadline and per-call AX timeout remain. This run establishes
  improved typical latency, not a hard whole-function bound against a hung app.
  Viewport culling does not detect text covered by overlays, collapsed previews
  or text partially painted inside otherwise intersecting nodes. Google's
  **69.9%** support is unchanged; geometry alone cannot fix that case.

### Local evidence

Scratch directory: `/tmp/discrete-ax-nodewalk-20261007/`.
For each of `cas_news`, `github`, `hackernews`, `wikipedia`, `python_docs`, `search`,
`table`, `news`: `.json` stores paired timings/ranges/metrics, `.png` the screenshot,
`.ocr.txt` OCR, `.before.txt` / `.after.txt` representative text, and `.window.json`
the window snapshot. `measure.py` reproduces paired reads while those temporary
windows exist; `before.py` is the baseline. Public diagnostic trees and profiles
are also scratch-only. No private benchmark or recorded session was used.

## Validation and cleanup

```sh
PYTHON_DOTENV_DISABLED=1 \
DISCRETE_DATA_DIR=/tmp/discrete-ax-nodewalk-20261007/test-data \
.venv/bin/python -m pytest -q
.venv/bin/ruff check discrete tests bench tools
```

**524 passed in 14.50 s; ruff: all checks passed.** Focused capture/node-walk suite:
**26 passed in 0.53 s**. Synthetic trees cover local duplicate projections, link
aria/title suppression, repeated short comments and words, bounded neighbourhoods,
radio label siblings, frame-only geometry, nested viewports, clipped/offscreen
text, tiny-visible fallback limits, hidden-subtree node-budget preservation,
AXTable/AXGrid rows, repeated/empty cells, plain group ranks, column-only tables,
cell wrappers, visible ranges, marker/fallback routing and deadline/node limits.

Changed files:

- `discrete/capture/accessibility.py`
- `tests/test_capture_text.py` (align existing expectations with tiny-visible fallback)
- `tests/test_ax_nodewalk.py` (new synthetic regression cases)
- `docs/ax_nodewalk_report.md` (this report)

`discrete/textkit.py` and other agents' implementation files were not edited.
`PYTHON_DOTENV_DISABLED=1` was set before project imports. No paid/model API call,
`discrete run`, `jarvis`, `.env`, real data directory, commit or browser-profile
change was involved.

Closed all **eight** temporary Little Arc windows using fresh native close-button
AX handles and background delivery. Verified each disappeared from WindowServer;
final native window IDs and the original normal-window/tab/selected-tab ID
snapshot matched exactly. The Guardian's floating “Close” control did not close
the window; cleanup used the native red close button instead. Arc remained running.
The temporary context menu was gone before the successful native closure.
