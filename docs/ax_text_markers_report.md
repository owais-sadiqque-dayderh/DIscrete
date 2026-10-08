# Chromium AX text-marker verification

Measured on 2026-10-07 on this Mac, branch `feat/memory-evolution` (HEAD at completion: `fa01372`).

## Decision

**Keep `DISCRETE_AX_TEXT_MARKERS=false`. Do not enable an app allowlist from these results.**

On eight public pages in Arc, **0/40 measured marker attempts returned text**. All 40 markers-on reads fell back to the existing node walk. The median of the eight page latency medians was **209.4 ms off / 209.0 ms on**. The apparent small timing differences are ordinary variability in the same fallback walk, not a marker speedup. No production code or configuration was changed.

This verifies the previously interrupted experiment in Worklog Phase 5 (capture text round 1) and Phase 6 (round 2): the current fast path does not work on the installed Arc. It does not establish whether another browser/version supports a different marker approach.

## Environment and method

- macOS **26.6.2 (25G83)**; Arc **1.167.1**; Python **3.13.15** from `.venv/bin/python`; PyObjC **12.2.2**.
- `ApplicationServices.AXIsProcessTrusted()` returned **True** for the actual Python process before browser work. No permission changes were needed.
- Read `discrete/capture/accessibility.py`, `discrete/capture/screen.py`, Worklog Phase 5/6, and `docs/audit_inputs.md` §1a/§1c before testing.
- Opened pages with `open -g -a Arc URL`. Arc used temporary Little Arc windows. The original normal-window IDs and tab/selected-tab IDs were recorded before opening anything. Hacker News supplied the requested public long-feed case; X was not opened and no login flow was exercised.
- Called `read_screen_text` directly in a small standalone script, with `T.AX_TEXT_MARKERS` toggled only in that process. Limits matched capture defaults: **4,000 nodes, 250 ms, 16,000 characters**. Passed the focused AX window's global bounds, **(20, 50, 1368, 727)** points. No daemon or hotkey was started.
- Discarded one warmup read per path, then performed **five reads per path**, interleaving off/on and reversing the order on alternate rounds. Timing includes `read_screen_text`, excluding browser load, screenshot capture, OCR, and the separate diagnostic calls. Instrumented the real parameterized AX call to retain error codes, `_web_visible_text` to distinguish marker hits from fallback, and the process-local PERF buffer for deadline events. No PERF data was flushed to a store.
- Captured the matching window with `screencapture -x -o -l WINDOW_ID`, after confirming the WindowServer title agreed with the focused AX window. Screenshots were **2736 × 1454 native pixels**. OCR used the project's **`discrete.capture.ocr.ocr_image`** (Apple Vision, accurate, language correction). Inspected the screenshots to confirm README, code, table, article-body and overlay states. The CAS article was scrolled in its test window to show body paragraphs. One initial GitHub screenshot referenced the earlier Wikipedia window; that comparison was discarded and GitHub was fully remeasured with its correct window ID.
- Set **`PYTHON_DOTENV_DISABLED=1`** before importing project modules. All experiment files are under `/tmp/discrete-ax-markers-20261007/`; no `.env`, `bench/private/`, real DIscrete data directory, caption or LLM endpoint was accessed. No model/API calls were made by the measurement script.

### Metric definitions

Chars count the returned Python string. Lines count nonempty stripped lines. Single-word share is the fraction of those lines with exactly one `textkit.tokens` token. Duplicate-line share is `(lines - unique textkit.norm_key lines) / lines`, matching the audit's definition.

**OCR-supported share** estimates viewport-visible text as the fraction of AX **content-token occurrences** whose token appears anywhere in the screenshot OCR's content-token set, using `textkit.content_tokens` exactly as the input audit does. This is a visibility proxy, not a literal percentage of correctly positioned visible characters: OCR omissions lower it, while repeated/offscreen words that also appear onscreen inflate it. Screenshot inspection supplements it. It cannot certify absence of occluded text, nor preservation of table columns or sentence layout.

Latency is the median of five reads. Text metrics below describe the **read at median latency**, separately for each path; returned lengths can vary when the walk hits its deadline. Each paired value is **off / on**. Deadline counts are library `health.capture_ax_deadline` events, not empty reads or OS messaging timeouts.

## Per-page results

| Page / state | Chars off/on | Lines off/on | Single-word % off/on | Duplicate % off/on | OCR-supported % off/on | Median ms off/on | Deadline reads off/on |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAS news article, scrolled body | 1473 / 1473 | 21 / 21 | 28.6 / 28.6 | 0.0 / 0.0 | 94.2 / 94.2 | 65.5 / 66.6 | 0/5 / 0/5 |
| GitHub Requests README | 1160 / 1160 | 40 / 40 | 22.5 / 22.5 | 0.0 / 0.0 | 89.0 / 89.0 | 168.5 / 167.8 | 0/5 / 0/5 |
| Hacker News front page | 4748 / 4748 | 94 / 94 | 20.2 / 20.2 | 19.1 / 19.1 | 99.6 / 99.6 | 250.5 / 250.5 | 5/5 / 5/5 |
| Wikipedia Accessibility article | 2398 / 2398 | 67 / 67 | 37.3 / 37.3 | 1.5 / 1.5 | 85.4 / 85.4 | 250.3 / 250.2 | 5/5 / 5/5 |
| Python tutorial, Fibonacci code block | 1497 / 1459 | 45 / 43 | 15.6 / 16.3 | 17.8 / 18.6 | 100.0 / 100.0 | 250.3 / 250.3 | 5/5 / 5/5 |
| Google search results, collapsed overview | 1459 / 1459 | 38 / 38 | 18.4 / 18.4 | 5.3 / 5.3 | 70.5 / 70.5 | 74.6 / 72.7 | 0/5 / 0/5 |
| Wikipedia population table, visible rows | 1851 / 1851 | 86 / 86 | 55.8 / 55.8 | 3.5 / 3.5 | 100.0 / 100.0 | 250.4 / 250.4 | 5/5 / 5/5 |
| Guardian news page, support overlay | 1826 / 1826 | 45 / 45 | 24.4 / 24.4 | 0.0 / 0.0 | 56.0 / 56.0 | 87.2 / 85.0 | 0/5 / 0/5 |

All pages: **0/5 empty reads per path**, **0/5 successful marker reads**, **5/5 marker fallbacks when on**. Both paths hit the walk deadline **20/40** times. No measured parameterized AX call returned the OS messaging-timeout code `-25204`; failed marker calls returned immediately. Full-read timing exceeded 250 ms slightly on the same 20 reads per path (typically 250.2–250.5 ms), consistent with the soft walk deadline rather than a hard whole-function timeout.

Character ranges across the five reads were constant except:

- Hacker News: **4198–4748 off / 3489–4748 on**.
- Python docs: **1497–1552 off / 1327–1552 on**.

Across the eight representative page samples, both paths had median **23.5% single-word lines**, **2.5% duplicate lines**, and **91.6% OCR-supported content tokens**. These are medians across pages, not a pooled corpus score.

### Pages and local evidence

| Case | Public URL | JSON / native PNG / returned text prefix in the scratch directory |
|---|---|---|
| CAS news | [Article](https://english.cas.cn/newsroom/news--archives/2024/cas-in-media/202406/t20240625_1129633.shtml) | `cas_news` |
| README | [psf/requests](https://github.com/psf/requests#readme) | `github` |
| Feed | [Hacker News](https://news.ycombinator.com/) | `hackernews` |
| Article | [Accessibility](https://en.wikipedia.org/wiki/Accessibility) | `wikipedia` |
| Docs/code | [First steps towards programming](https://docs.python.org/3/tutorial/introduction.html#first-steps-towards-programming) | `python_docs` |
| Search | [Search query](https://www.google.com/search?q=python+accessibility+API) | `search` |
| Table | [Population table](https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population#Sovereign_states_and_dependencies_by_population) | `table` |
| Overlay | [Guardian article](https://www.theguardian.com/world/article/2024/jun/25/chinas-change-6-lunar-probe-returns-world-first-samples-from-far-side-of-the-moon) | `news` |

For each prefix: `.json` records all five timings, error/deadline counts, character ranges and metrics; `.png` is the native screenshot; `.ocr.txt`, `.False.txt` and `.True.txt` are OCR/off/on text. The standalone measurement script is `/tmp/discrete-ax-markers-20261007/read_page.py`. These scratch artifacts are local and temporary, not tracked repository files.

## Failure diagnosis and remaining capture issues

Arc's `AXWebArea` advertises both `AXStartTextMarkerForBounds` and `AXEndTextMarkerForBounds`, but advertising is not proof of working support. The current `_web_visible_text` passes an AppKit `NSValue` rectangle. On **every measured page/read**, both calls returned **`-25201` (`kAXErrorIllegalArgument`)**: **80 errors for 40 attempts**. The helper swallows these errors, returns empty text, and the caller falls back normally.

A diagnostic replaced only the argument with `AXValueCreate(kAXValueCGRectType, (origin, size))`, without changing the library. Both attributes then returned **`-25212` (`kAXErrorNoValue`) on all eight pages** (16 calls). On Wikipedia, testing the web-area bounds and several smaller interior rectangles also produced no marker. Thus a small argument-type correction alone does **not** establish a working or better fast path. No such fix was shipped, and no app-specific enablement is supported by this experiment. Chromium's [AX implementation source](https://chromium.googlesource.com/chromium/src/+/91b8a39168581465b04ca89ef0ba2a9e17e7c35e/ui/accessibility/platform/browser_accessibility_cocoa.mm) was consulted for the advertised attributes; the behavior conclusion here is based on the actual Mac calls.

The current node walk still needs separate work:

- A table naturally has short cell lines; **55.8%** single-word lines on this table is not entirely prose fragmentation. Neither setting preserves a structured table.
- Hacker News returns **4748 AX chars versus 2440 OCR chars**, with **19.1% exact duplicate lines**. The high OCR-supported share does not penalize repeats or summaries composed of already-visible words.
- Python docs has repeated code output and returns different tails at the deadline. The marker branch consumes lookup/call work from the same deadline; a smaller returned tail is not improved culling.
- The Guardian overlay case has only **56.0% OCR-supported tokens**. Screenshot comparison shows article text exposed behind the support overlay. The Google collapsed overview similarly has **70.5%** support. Viewport intersection alone does not establish that text is unobscured or fully painted.

Changing line grouping or culling would not repair the unsupported marker path. Those changes would need their own focused correctness tests and live comparison; they were not added to this verification task.

## Tests and cleanup

```sh
PYTHON_DOTENV_DISABLED=1 \
DISCRETE_DATA_DIR=/tmp/discrete-ax-markers-20261007/test-data \
.venv/bin/python -m pytest tests/test_capture_text.py -q
```

**11 passed in 0.43 s.** This covers marker/fallback routing, bounded marker validation, node text grouping, viewport filtering, visible ranges, aria/icon suppression and native OCR handoff. The mocked marker-success test passes but cannot establish live browser support, which is why the live result controls the decision.

Closed all **nine** temporary Little Arc windows, including the extra CAS navigation window, using native background-targeted close actions. Verified each window disappeared from the window list. Final native IDs matched the original set; original normal windows, tabs and selected-tab IDs matched the pre-test snapshot exactly. Arc remained running. Although opens used `-g` and cleanup requested background delivery, Arc was observed frontmost after cleanup; background input did not guarantee foreground preservation.

Repository change: **this report only**. No capture code, tests, `.env`, live data, paid APIs, daemon/hotkey, or other agents' files were changed by this task. No commit was made.
