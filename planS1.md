# Stage 1 — Data Collection: Full Specification

**Status: design record, revised in place with a visible revision log.**
This file is self-sufficient: a reader with no access to the project's
history should be able to implement Stage 1 from it alone. Design changes are
made directly in this file, and every change is recorded in the Revision log
below (what changed, why, and the evidence). Implementation bugs — cases where
the design was right but the code was wrong — are recorded in `insights.md`,
not here. Sections changed by a revision carry a `[Rev N]` marker.

---

## Revision log

### Rev 1 — 2026-09-14 (architecture audit against the real WorkAA store)

Evidence base: one real 70-minute session captured by the WorkAA prototype
(89 observations), plus a survey of comparable systems (screenpipe, Microsoft
Recall, GUM, napsack, tada) and current on-device model tooling. Full numbers
are in `audit.md`.

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 1.1 | §6.1, §8 | **Text source hierarchy inverted.** The primary on-screen text is now a direct Accessibility (AX) text walk of the focused window; Apple Vision OCR is the fallback. Raw text is stored verbatim as a new field `screen_text`. The VLM no longer transcribes text at all; the `dense_text` field is removed. | The VLM-curated `dense_text` averaged **150 characters** per capture; 15 captures stored the single word "discrete". The whole 70-minute corpus was ~1255 distinct tokens, too thin for retrieval. AX text is exact (no recognition errors) and takes ~50 ms. screenpipe, the most mature open-source screen-memory system, uses this exact order: AX tree first, OCR only when AX is empty. |
| 1.2 | §6.1 | OCR and the VLM image are **cropped to the focused window's frame**, not the whole monitor. | This removes menu bars, docks, and neighbouring windows deterministically, instead of asking a model to ignore them. It also defines which monitor to capture (see 1.8). |
| 1.3 | §6.2, §8 | **`app` is AX-only.** The model can no longer write `app`. The web property goes in a new `site` field, normalized to a registrable domain. | The store held 25 distinct `app` values where the true count was ~8, including model typos: `logosystem.co` / `logosesystem.co` / `logosytem.co` / `Logosytem`, `Grainient` / `Graident`, `logofoilo.com`. The old field definition ("application **or website name**") invited the model to overwrite ground truth. |
| 1.4 | §2 | **Triggers are native OS events, not ActivityWatch polling.** App activation, focused-window change and title change come from macOS notifications; idle comes from the OS idle counter. Clipboard-copy and typing-pause triggers are added. ActivityWatch is no longer a dependency. | Polling every 30 s added up to 30 s of latency and a second server process. The OS already emits these events instantly. screenpipe triggers on the same set, including clipboard. |
| 1.5 | §6.3 | **Browser URL comes from a ground-truth source now; this is no longer deferred.** Order: the browser's own scripting interface (or a URL-reporting browser extension) → AX `AXURL`/`AXDocument` → the model's reading. | The deferred "known limitation" is where the `app`/`site` typo clusters above came from: Arc returned an empty AX URL, so the model guessed. |
| 1.6 | §5 | **A capture is aborted and re-armed if the foreground app differs between the before-frame and the after-frame.** | 12 captures stored `app = Adobe Illustrator 2026` alongside a browser tab title. Captions fused two apps ("working with an ASCII art generator *and then viewing a collection of logos*"). Two `app` values were literally compound (`LogoFolio and ASCII Magic`). |
| 1.7 | new §9 | **Privacy tier added: exclusion list, secret scrubbing, PII detection, and an explicit local-versus-cloud model policy.** | The store contained the user's email address and a live Supabase project reference, both promoted to entities, plus a full OAuth consent screen. None of the three plans had any sensitive-content policy. Microsoft Recall ships sensitive-content filtering on by default. GUM has an audit step. screenpipe does PII redaction. |
| 1.8 | §5, §12 | **Multi-monitor question closed.** Capture targets the display that contains the AX-focused window, not the display under the mouse. | The mouse-display rule loses the passive second-screen case, which is exactly the blind spot §2.3 exists to close. |
| 1.9 | §8 | **New fields: `interaction` and a capture interval.** `interaction` records input counts during the capture window. `started_at` / `ended_at` record the capture interval. | This separates "the user *did* X" from "the user *viewed* X". GUM's own prompt warns that "the input is what the user is viewing. It may not be what the user is actively doing". The WorkAA store turned logos the user scrolled past into entities. The interval gives Stage 3 real event time for temporal questions. |
| 1.10 | §6.4, §11 | **Heartbeat recapture is content-driven, with a floor and a ceiling.** The fixed every-Nth-heartbeat rule is replaced. | Screen text is now computed on every heartbeat anyway, so the text-change signal the content-driven option needs is free. The fixed count was chosen only for simplicity. |
| 1.11 | §6.5, new §10 | **All screen text is treated as untrusted data.** Prompts must delimit it as data, never as instructions. | Memory poisoning (a web page's text injected into long-term memory through summarization) is a documented attack class against memory-augmented agents. A screen recorder ingests arbitrary web text by design. |
| 1.12 | §12 | Audio is recommended as a Stage 1b component using Apple's on-device `SpeechAnalyzer`, **pending the user's scope decision**. | "What did I do this morning" (Stage 3) is incomplete without meetings. On English speech, `SpeechAnalyzer` benchmarks ahead of Whisper Small in accuracy and ~3× faster, fully on-device. |
| 1.13 | §13 | A **build-versus-buy gate for the capture tier** is recorded: evaluate screenpipe as the capture substrate before writing custom capture code. | screenpipe already implements AX-first text, OCR fallback, event triggers, frame deduplication, PII redaction, audio, and SQLite FTS5 locally. |

### Rev 2 — 2026-09-14 (user decisions before implementation)

Two scope decisions by the user, applied before any code was written:

| # | Section | Change | Why |
|---|---|---|---|
| 2.1 | §6.2, §11, §12, §13 | **No locally hosted models.** The VLM, and every other model call, goes through an OpenAI-compatible API (OpenRouter or similar). The Rev 1 "local VLM by default / cloud by opt-in" policy (`VLM_ALLOW_CLOUD`) is removed. Rev 1 row 1.12's on-device transcription is replaced by an API speech-to-text model, if audio is adopted. | User decision: running local models is not affordable for this product. Apple Vision OCR is **retained**. It is an operating-system API with no model to download or host, and it costs nothing per call. |
| 2.2 | §1, §6, §8, §9, §11 | **The privacy tier is removed from Stage 1.** This covers the exclusion lists, secret scrubbing, PII detection, the secure-field check, the `sensitive` VLM field, and the related tunables. It supersedes Rev 1 row 1.7. | User decision: privacy is out of focus for now. Retained, because it is correctness rather than privacy: §9, which keeps the system from modelling itself, and §10, which treats screen text as untrusted input (memory-poisoning defence). |

### Rev 3 — 2026-09-15 (first live run of the v1 build)

Evidence: the first `discrete run` session (37 frames → 17 observations, stored in
`~/.cache/discrete`), a code review of the v1 build, and a re-run after fixes.
Implementation bugs from the same review are in `insights.md` §7.11–7.26.

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 3.1 | §2.2, §2.3 | **App, window and title changes are detected by a poller** (`POLL_S` = 0.5 s) over the Quartz window list, not by `NSWorkspace` or AX notifications. | Those notifications are delivered on a Cocoa run loop. The asyncio daemon runs none, and in v1 they never fired. One poll costs about 1 ms. |
| 3.2 | §6.1 | **Window identity comes from the Quartz window list**: app, pid and frame, plus the title when AX is not granted. AX adds the title, document/URL and the text walk only when the process is trusted. | Identity must not depend on one permission. In v1 a failed AX import blanked `app`, `window_title`, `url` and `site` on 17/17 rows. `NSWorkspace.frontmostApplication` also goes stale without a run loop. |
| 3.3 | §6.2 | **VLM image budget:** frames sent to the VLM are downscaled to `VLM_MAX_WIDTH` (default **1024**) with `VLM_IMAGE_DETAIL` = `low`. The harness may raise it if caption specificity suffers. | The first run used ~305K tokens in ~15 minutes with two full-screen images per call. On-screen text reaches the model through AX/OCR, so the images only need to carry layout and change. |
| 3.4 | §9 | **Self-observation is also detected by window title** (`SELF_WINDOW_PATTERNS`, default `bin/discrete`, `discrete run`). Agent-origin captures are stored for audit with no VLM call, and are excluded from Stage 2 and 3. | Observations 1, 2 and 5 of the first run were the daemon's own terminal log, summarized as user activity. The terminal is an ordinary app, so `SELF_APPS` (app names) cannot catch it. |
| 3.5 | §5 | **`doctor` verifies Screen Recording by grabbing a frame**, not only through `CGPreflightScreenCaptureAccess`. | The preflight reported "missing" for a terminal-launched Python whose captures worked. |

### Rev 4 — 2026-09-15 (navigation is a capture boundary)

Evidence: a live run reading a burst of Hacker News articles produced **one**
observation for the whole burst instead of one per article.

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 4.1 | §4, §5 | **A navigation to a different window/app finalizes the pair in flight and opens a fresh one**, rather than resetting the shared debounce timer. Previously every trigger — including each `window_change` — cancelled and rescheduled the single `_pending`/timer, so continuous browsing coalesced into a single before/after that spanned the first and last page. The debounce still governs settling *within* one page. | Reading N articles faster than `MOUSE_DEBOUNCE_S` (2 s) yielded 1 observation, not N. The interval was never the cause; the single-slot coalescing model was. Generalises the existing focus-drift guard (§5) from "abort at flush" to "segment at the boundary". |
| 4.2 | §6.1 | **A rotated (boundary) capture reads its text from the frame it froze, not from the live window.** After a tab/window switch the live AX tree and browser URL already describe the *new* page, so a stale capture uses the last cached frame as its `after`, OCRs that image for `screen_text`, and keeps the URL/title captured when the page was in focus. The VLM still sees the frozen frame, so the caption and entities stay correct per article. | Same-process tab navigation cannot be read retroactively — the AX tree only exposes the current tab — so the previous article's text must come from its own captured pixels. |

### Rev 5 — 2026-09-15 (browser page identity)

| # | Section | Change | Why (evidence) |
|---|---|---|---|
| 5.1 | §2, §6.3 | **Browser tab identity comes from the tab's `AXWebArea`** (`AXURL`, `AXTitle`). It is cached, costs about 3 ms per poll, and is re-found when the window title changes or every 2 s. The change key is (pid, title, URL), so in-tab navigation triggers a capture. AppleScript is only a fallback. | On the live store most Arc rows were titled "Space 1" with no URL, because Arc titles windows by Space. AppleScript cost 0.25 s in Arc and 3.7 s in Chrome per call (`insights.md` §7.33). |
| 5.2 | §5 | **A rotated capture uses a frame of its own window**, the newest match in a 6-frame ring, or is dropped. A navigation trigger keeps no "before" frame from the page just left. | Rows were captioned from the next page's frame ("Blueprint" captioned as a Google Doc) (`insights.md` §7.37). |
| 5.3 | §6.2 | **The caption describes the last screenshot** and names the page, article, document or item by its title. | Before/after captions narrated transitions ("switched from Hacker News to…"), which leaked the previous page into every row. |

---

## 1. Purpose and scope

The Data Collection stage watches the user's screen and activity continuously
and turns raw moments into structured, enriched **Observation** records — the
raw material every later stage builds on. It decides *when something worth
recording happened* and *records it faithfully and completely*. It does not segment, form claims, extract
long-term entities, or answer questions — those are later stages (§14).

### Pipeline, end to end  `[Rev 1]`

```
Native OS event triggers (app switch / window focus / title change / click / scroll /
typing pause / clipboard copy / content-driven heartbeat)
        │
        ▼
   Arm + debounce  (wait for the user to settle)
        │
        ▼
  Frame capture of the focused window's display (before + after)
        │   └─ foreground app changed between frames?  → abort, re-arm
        ▼
  Frame dedup tier-0 (perceptual hash vs. last kept frame)  → duplicate: bump, stop
        │
        ▼
  Ground-truth reads: AX metadata + URL source + AX text walk (OCR fallback, window-cropped)
        │
        ▼
  Semantic enrichment: VLM call via API (caption / entities / useful), fed
  screen_text as delimited data
        │
        ▼
  Structured Observation draft (§8) → Stage 2
```

---

## 2. Trigger sources  `[Rev 1]`

All triggers feed one shared capture pipeline. Any one of them can arm a
pending capture.

**2.1 Input events.** A system-level listener watches mouse clicks, scrolls
and key presses. Keyboard arming can be switched off (`KEYBOARD_CAPTURE`,
default **on**). Key *contents* are never recorded — only counts (§8
`interaction`).

**2.2 App activation.** `[Rev 3]` A poller reads the frontmost normal-layer
window from the Quartz window list every `POLL_S` (default **0.5 s**). A change
of owning process arms an `app_switch` capture. (Rev 1 specified the
`NSWorkspace` activation notification. It needs a Cocoa run loop, which the
daemon does not run, so it never fired.)

**2.3 Focused-window and title change.** `[Rev 3]` The same poller arms a
`window_change` capture when the front window's title changes within the same
process. This catches passive context switches with no input at all (a new
video autoplaying, a new tab, a document switched by shortcut) within half a
second. It replaces the Rev 1 AX observers, for the same run-loop reason.

**2.4 Typing pause.** A burst of key presses followed by
`KEY_IDLE_DEBOUNCE_S` of silence arms a capture. This is the moment the text
the user just wrote is complete on screen.

**2.5 Clipboard copy.** A change in the pasteboard change-count arms a
capture. Copying is a strong signal of salience ("the user took this"). The
clipboard *contents* are not stored separately; the capture records the
screen, and the clipboard event is recorded as a flag in `interaction`.

**2.6 Content-driven heartbeat.** See §6.4. This covers long passive stretches
(video, reading, calls) in which none of the triggers above fire.

**2.7 Idle / away.** The OS idle counter
(`CGEventSourceSecondsSinceLastEventType`) is sampled. Crossing
`IDLE_THRESHOLD_S` (default **300 s**) marks the user away; any input marks
them back. Idle transitions affect session boundaries only (§3) and suspend
heartbeats while away. Why an OS idle signal rather than "no trigger for N
seconds": a passive session (watching, reading) has no triggers but the user is
present, so trigger silence cannot distinguish "present and passive" from
"gone".

*Why not ActivityWatch (the prior design):* it was polled every 30 s, so every
context switch was detected up to 30 s late. It also required a separately
running server. The OS already publishes every signal needed, as events.
ActivityWatch remains an acceptable optional **URL source** through its
browser extension (§6.3), not as the trigger mechanism.

---

## 3. Session tracking

All triggers share one session identifier (a UUID). It rotates on either:
- no trigger of *any* kind for `SESSION_GAP_S` (default **300 s**), or
- an idle transition (§2.7) in either direction.

Every Observation records its session. Sessions are a coarse *capture*
boundary only. The finer, meaning-based unit ("episode") is computed in Stage 2
and is what summaries and claims are built on. `[Rev 1]`

---

## 4. Debounce — waiting for the user to settle

On the first trigger since the last completed capture, a pending capture
begins and a **before** frame is taken immediately from the warm frame cache
(§5). A timer then starts:
- `MOUSE_DEBOUNCE_S` (default **2.0 s**) after a click, scroll, app-activation,
  window-change or clipboard trigger.
- `KEY_IDLE_DEBOUNCE_S` (default **6.0 s**) after a keyboard trigger.

Every further trigger before the timer elapses resets it; a keyboard trigger
arriving mid-wait upgrades the pending capture to "keyboard". The capture fires
only once the delay elapses with no further trigger, so it always reflects a
moment the user settled on.

---

## 5. Frame capture mechanics  `[Rev 1]`

- A lightweight background loop keeps a fresh in-memory frame (~every 0.2 s)
  of the **display that contains the AX-focused window** (the frontmost app's
  focused window frame, from AX `AXPosition`/`AXSize`). Where there is no
  focused window, it falls back to the display under the mouse. No disk write,
  no model call — a warm cache only.
- Every real capture is a **before/after pair**: the warm frame from when the
  trigger armed, plus a fresh frame when the debounce elapses. The pair lets
  enrichment describe a *change* rather than guess from one still.
- **Focus-drift abort** `[Rev 1]`: the AX foreground application is read at the
  before-frame and again at the after-frame. If they differ, the capture is
  aborted without any model call and immediately re-armed, so the new app gets
  its own clean capture. A pair spanning two apps cannot be described
  faithfully as one moment. In the prior design such pairs produced fused
  captions and compound `app` values.
- Saved frames are downscaled to a maximum width of **1600 px**, JPEG quality
  **65** (see §11).
- **Frame dedup tier-0** `[Rev 1]`: a perceptual hash (pHash, 64-bit) of the
  after-frame, cropped to the focused window, is compared with the last *kept*
  frame for the same window. At Hamming distance ≤ `FRAME_PHASH_MAX_DIST`
  (default **6**) with the same app, window title and URL, the capture is a
  duplicate. Stage 2 bumps the existing observation's `dup_count`, and no AX
  text walk, OCR or VLM call runs. This is the cheapest and most reliable
  duplicate signal available: the same paused or unchanged screen is
  pixel-near-identical. A downstream text comparison has already been shown to
  miss it (§6.1). (Library: `imagehash`.)
- **Blank-frame detection:** the first time a frame's pixel standard deviation
  falls below 2.0, a loud warning is logged once. macOS silently returns black
  frames when Screen Recording permission is missing.
- **Permissions are checked at startup.** Screen Recording and Accessibility
  are both queried through the OS permission APIs, with a loud warning if
  either is missing. Automation (Apple Events) permission for each supported
  browser is requested the first time it is needed (§6.3).

---

## 6. Enrichment — turning a capture into a structured record

### 6.1 Ground-truth reads, gathered before any model call  `[Rev 1]`

1. **AX metadata.** A synchronous AX read of the frontmost app's localized name
   (`app`), the focused window's title (`AXTitle`), and its document path or URL
   (`AXDocument` / `AXURL`). This is ground truth from the OS or the app, never
   inferred from pixels. Its only failure mode is absence.

2. **Screen text — AX text walk first.** A bounded walk of the focused window's
   AX subtree collects the `AXValue` / `AXTitle` / `AXDescription` of text-bearing
   elements (static text, text areas, text fields, web-area text, table cells),
   in reading order. Roles that are window chrome are skipped: menu bar,
   toolbar, tab strip, scroll bars, and buttons without text. Limits:
   `AX_WALK_MAX_NODES` (default **4000**), `AX_WALK_TIMEOUT_MS` (default
   **250 ms**). For Chromium- and Electron-based apps the walk first sets
   `AXManualAccessibility` / `AXEnhancedUserInterface` on the app element,
   because these apps expose a complete tree only when an assistive client asks
   for it.
   - Why AX first: the text is exact (correct characters, casing and Unicode,
     no recognition confidence), takes on the order of tens of milliseconds,
     and already excludes pixels that are not text. screenpipe uses the same
     order.

3. **Screen text — OCR fallback.** If the AX walk yields fewer than
   `AX_TEXT_MIN_CHARS` (default **80**) characters, or the app is on a
   known-canvas list (e.g. Figma, Illustrator's canvas, games, remote
   desktops), Apple Vision OCR runs on the after-frame **cropped to the focused
   window's frame**. On macOS 26+ this uses `RecognizeDocumentsRequest`, which
   returns paragraphs, lists and tables in reading order; older systems use
   `VNRecognizeTextRequest` at `.accurate`. OCR runs locally and costs nothing
   per call.
   - `text_source` records which path produced the text: `ax`, `ocr` or `ax+ocr`.
     `ax+ocr` means the AX text was sparse and OCR supplemented it.

4. **The result is stored verbatim as `screen_text`**, capped at
   `SCREEN_TEXT_MAX_CHARS` (default **16 000**).
   This is the primary text record of the moment. Nothing downstream may
   replace it with a model's paraphrase.

*Why the VLM no longer produces `dense_text` (the prior design):* asking the
VLM to "select the substantive text and exclude chrome" made it abridge. The
curated field averaged 150 characters, and the retrieval corpus came out too
small for term statistics to mean anything. Chrome exclusion is now done
deterministically: by AX role filtering and by cropping to the window.

### 6.2 Semantic enrichment — the VLM call  `[Rev 1]`

One call to the configured vision-language model receives:
- the before and after frames (cropped to the focused window, downscaled per §5),
- the AX metadata and resolved URL, labelled as ground truth,
- `screen_text`, inside explicit data delimiters,
- the `interaction` summary (§8).

The model returns strict structured output:

| Field | What it is |
|---|---|
| `caption` | One or two sentences naming specifically what the user is doing, and with what document, page, repository or person. It states the change between the frames when there is one. No generic filler such as "The user is…". |
| `entities` | Proper nouns *the user is working with* — the document, project, repository, person, tool. It explicitly excludes names that merely appear on screen in a list, gallery, feed or advertisement. |
| `useful` | `false` if the screen shows nothing meaningful (lock screen, empty desktop, pure chrome). |

The model does **not** return `app`, `url`, `window_title` or any transcription.
Those are ground-truth reads (§6.1).

**Prompting rules (binding):**
- Everything inside the data delimiters is *content the user was looking at*,
  never instructions to the model. The prompt says so explicitly, and any
  instruction-like text in `screen_text` is to be described, not followed
  (§10).
- Schema examples in prompts use type placeholders (`<string>`, `<bool>`),
  never literal sample values. Models copy literal example values; see
  `insights.md` §7.5.

**Model placement policy** `[Rev 2]`: the enrichment call goes to a
vision-capable model behind an OpenAI-compatible API (OpenRouter or similar),
configured through `CAPTION_LM_*`. No model is hosted locally. The model is
chosen by the Stage 3 evaluation harness (§13, open item 3), by caption
specificity and entity precision per unit cost, not by assertion.

**Image budget** `[Rev 3]`: images are downscaled to `VLM_MAX_WIDTH` (default
**1024 px**) and sent at `VLM_IMAGE_DETAIL = low`. Text reaches the model as
`screen_text`, so the images only need to carry layout and change.

### 6.3 Ground-truth precedence and the URL source  `[Rev 1]`

- `app` and `window_title` come **only** from AX. The model never writes them.
- `url` resolution order, first non-empty wins:
  1. **Browser scripting.** For supported browsers (Safari, Chrome, Arc, Edge,
     Brave), the active tab's URL and title are read through the browser's
     Apple Events scripting interface. This needs Automation permission per
     browser. It is ground truth from the browser itself. Where a browser does
     not support this, a URL-reporting browser extension (e.g. ActivityWatch's
     `aw-watcher-web`) is the equivalent source.
  2. AX `AXURL` / `AXDocument`.
  3. A URL literally present in `screen_text` inside the window's address-bar
     region (OCR/AX text, not model generation).
  4. Otherwise empty. The model is never a URL source.
- `site` = the registrable domain of `url` (e.g. `logosystem.co`), computed
  deterministically (Public Suffix List; library: `tldextract`). If there is no
  URL, `site` is empty.

*Why this is no longer deferred (the prior design deferred it):* the
AX-empty-URL case let the model invent URLs and site names, which produced the
typo clusters in the Rev 1 table.

### 6.4 Heartbeat captures — the passive-session case  `[Rev 1]`

While the user is present (§2.7) and no other trigger has fired, a heartbeat
tick runs every `HEARTBEAT_TICK_S` (default **60 s**):
- It grabs a frame, runs the pHash check (§5) and the AX text walk / OCR
  (§6.1). All of this is local and free.
- It computes `text_change` = 1 − trigram-Jaccard(`screen_text` now,
  `screen_text` at the last *full* capture).
- **Full capture** (the complete §6.1–§6.3 pipeline with VLM) is forced when
  `text_change ≥ HEARTBEAT_TEXT_CHANGE` (default **0.5**) and at least
  `HEARTBEAT_MIN_GAP_S` (default **120 s**) has passed since the last full
  capture (the floor). It is also forced whenever `HEARTBEAT_MAX_GAP_S`
  (default **600 s**) has passed with no full capture (the ceiling).
- Otherwise, if the pHash differs, a **light row** is stored: the ground-truth
  metadata plus `screen_text`, with `caption` set to the deterministic string
  `"{app}: {window_title}"` and no VLM call. If the pHash matches, the tick is
  a duplicate (a `dup_count` bump).

The effect is that a slide deck being clicked through or subtitles changing
get real captures roughly every 2 minutes, while a paused video costs nothing
beyond one full capture every 10 minutes.

*Why this replaced the fixed every-3rd-heartbeat rule:* that rule was chosen
for predictability over adaptivity. The signal the adaptive rule needs is now
computed on every tick at no extra cost, and the floor and ceiling keep cost
bounded and predictable anyway.

### 6.5 Failure handling — nothing broken becomes stored data

- If the VLM call does not return parseable structured output after
  `CAPTURE_MAX_RETRIES` (default **3**) attempts, with exponential backoff from
  `CAPTURE_RETRY_BASE_S` (default **4.0 s**), the capture is **degraded to a
  light row** `[Rev 1]`: ground-truth metadata plus `screen_text`, with the
  deterministic caption. The ground-truth reads are valid regardless of the
  model, so a model failure must not discard them. Raw error or exception text
  is never stored.
- `useful: false` → dropped.
- A capture with `screen_text` under 40 characters **and** a caption under 5
  characters is dropped.
- Every drop and every degrade is logged with its reason and counted. "How many
  captures were discarded or degraded, and why" is always answerable from the
  running system's logs. `[Rev 1]` The counters are also exposed on the store
  card (Stage 3 §9) so a systematic failure is visible without reading logs.

---

## 7. Accessibility — the second, unrelated use elsewhere (boundary note)

The Accessibility API is also used, separately, by the product's
text-completion feature to read the literal contents of the field the user is
focused on. That feature is out of scope here. The one interaction that
matters is §9: the completion feature's synthetic keystrokes must not be
captured as user activity.

---

## 8. What gets produced — the Observation draft  `[Rev 1]`

| Field | Description |
|---|---|
| `source` | `screen` (full capture), `heartbeat` (light row), or `degraded` (VLM failed; ground truth only). |
| `origin` | `user` or `agent` (§9.3). |
| `event_kind` | Which trigger armed the capture: `click`, `scroll`, `keyboard`, `app_switch`, `window_change`, `clipboard`, `heartbeat`. |
| `session_id` | Shared session identifier (§3). |
| `started_at` / `ended_at` | The capture interval: when the pending capture armed and when it fired. This is **event time**. |
| `app` | Foreground application name. **AX only.** |
| `bundle_id` | Foreground application bundle identifier (AX / NSRunningApplication). Stable across localization and renames. |
| `window_title` | AX window title. |
| `url` | Resolved per §6.3; empty if there is no ground-truth source. |
| `site` | Registrable domain of `url` (§6.3). |
| `screen_text` | Verbatim on-screen text of the focused window (§6.1). |
| `text_source` | `ax`, `ocr` or `ax+ocr`. |
| `caption` | VLM caption (§6.2) or the deterministic `"{app}: {window_title}"`. |
| `entities` | VLM-reported working entities (§6.2). Empty for light rows. |
| `interaction` | `{keys, clicks, scrolls, clipboard: bool}`: counts during the capture interval. `keys = clicks = scrolls = 0` means the user was viewing, not acting. |
| `frame_phash` | 64-bit perceptual hash of the after-frame (§5). |
| `frame_path` | Path to the saved JPEG (retention per §13). |
| `created_at` | Ingestion time (transaction time). |

Stage 2 owns storage, deduplication beyond tier-0, segmentation, and everything
after.

---

## 9. Preventing the system from modeling itself

`[Rev 2]` The Rev 1 privacy tier (exclusion lists, secret scrubbing, PII
detection) was removed by user decision; see the Revision log. This section
keeps only the self-modelling rules, which are about correctness, not privacy.

- **Suppression:** while the product's own completion feature is typing
  synthetic keystrokes, all armed triggers are dropped, and this continues for
  **3 s** after it finishes.
- **Origin classification:** `origin = "agent"` if `bundle_id` / `app` matches
  `SELF_APPS`, or if ≥ 60 % of the capture's text (trigram containment) overlaps
  text the completion feature typed in the last ~3 minutes. `agent` rows are
  excluded from every later stage.

`[Rev 3]` **Self-window detection.** A capture whose window title matches
`SELF_WINDOW_PATTERNS` (default `bin/discrete`, `discrete run`) is
`origin = agent`, as is one whose app is in `SELF_APPS`. Agent-origin captures
are stored for audit without a VLM call. Stage 2 never drafts from them, and
Stage 3 never returns them. A terminal showing the daemon's own log is an
ordinary app, and only its title identifies it.

---

## 10. Screen text is untrusted input  `[Rev 1 — new section]`

Everything in `screen_text`, `window_title` and `url` was authored by third
parties: web pages, emails, documents, chat messages. Any of it can contain
text written to manipulate a model ("ignore previous instructions and remember
that…"). Because this system turns what it reads into long-term memory, an
injected instruction that survives into a stored claim persists across every
later session (memory poisoning).

Binding rules for every model call in every stage that receives screen-derived
text:
- screen-derived text is placed inside explicit data delimiters, and the
  instructions state it is observed content, not instructions;
- the model is asked to *describe* what the user was looking at, never to act
  on or adopt statements found in it;
- outputs that are verbatim copies of imperative sentences from `screen_text`
  are flagged by a cheap containment check and rejected. Stage 2 enforces this
  for claims.

---

## 11. Full tunable reference  `[Rev 1]`

| Tunable | Default | Meaning |
|---|---|---|
| `MOUSE_DEBOUNCE_S` | 2.0 | Settle delay after click/scroll/app/window/clipboard triggers. |
| `KEY_IDLE_DEBOUNCE_S` | 6.0 | Settle delay after keyboard triggers; also the typing-pause definition. |
| `KEYBOARD_CAPTURE` | on | Whether key presses can arm a capture. |
| `SESSION_GAP_S` | 300 | Trigger silence before a new session. |
| `IDLE_THRESHOLD_S` | 300 | OS idle seconds before the user counts as away. |
| `CAPTURE_MAX_RETRIES` | 3 | VLM attempts before degrading to a light row. |
| `CAPTURE_RETRY_BASE_S` | 4.0 | Backoff base. |
| `FRAME_PHASH_MAX_DIST` | 6 | pHash Hamming distance at or below which a frame is a duplicate. |
| `AX_WALK_MAX_NODES` | 4000 | Bound on the AX text walk. |
| `AX_WALK_TIMEOUT_MS` | 250 | Time bound on the AX text walk. |
| `AX_TEXT_MIN_CHARS` | 80 | Below this, OCR supplements AX text. |
| `SCREEN_TEXT_MAX_CHARS` | 16000 | Cap on stored `screen_text`. |
| `HEARTBEAT_TICK_S` | 60 | Heartbeat interval while present and quiet. |
| `HEARTBEAT_TEXT_CHANGE` | 0.5 | Text change that justifies a full capture. |
| `HEARTBEAT_MIN_GAP_S` | 120 | Floor between heartbeat-forced full captures. |
| `HEARTBEAT_MAX_GAP_S` | 600 | Ceiling: force a full capture at least this often. |
| Screenshot max width / JPEG quality | 1600 px / 65 | See §12. |
| Blank-frame stddev threshold | 2.0 | Warn that Screen Recording permission may be missing. |

---

## 12. Resolved decisions

1. **Image compression stays at 1600 px / JPEG q65.** Exact text now comes from
   AX/OCR, so the image's job is layout and scene understanding, which tolerates
   this compression. Full quality would double per-capture cost for a benefit
   the text path already delivers.
2. `[Rev 1]` **Text hierarchy: AX text walk → window-cropped OCR → never the
   VLM.** The VLM describes; it does not transcribe.
3. `[Rev 1]` **Heartbeat recapture is content-driven with a floor and a
   ceiling** (§6.4). This supersedes the earlier fixed-count rule.
4. `[Rev 1]` **Triggers are native OS events**; ActivityWatch is not a
   dependency.
5. `[Rev 1]` **`app` is AX-only; `site` is derived from a ground-truth URL; the
   model is never a URL source.**
6. `[Rev 1]` **Multi-monitor:** capture the display containing the AX-focused
   window.
7. `[Rev 2]` **All model calls go through OpenAI-compatible APIs (OpenRouter or
   similar); no locally hosted models.** There is no privacy tier in this stage
   (user decision; see the Revision log).

---

## 13. Open questions

1. **Audio (recommended: adopt as Stage 1b; needs the user's scope
   decision).** Meetings and calls are a large share of "what did I do today",
   and nothing here captures them. `[Rev 2]` Recommended shape: transcription
   through an API speech-to-text model, off by default. It would be
   auto-enabled only while a meeting app holds the microphone, with transcripts
   stored as `source="audio"` observations. If the user declines, the Stage 3
   product claim must explicitly exclude meetings.
2. **Build versus buy for the capture tier.** screenpipe (open source, local,
   SQLite/FTS5, REST API) already implements the text hierarchy, event
   triggers, frame dedup, PII redaction and audio specified here. Before
   writing custom capture code, a one-day spike should check whether Discrete
   can consume screenpipe's store as its Stage 1 and add only what is missing:
   the VLM caption, the `interaction` summary, the focus-drift abort, and the
   origin rules. Decision criteria: data completeness against
   §8, licence, and operational weight.
3. **Which API VLM.** `[Rev 2]` It is chosen by running two or three
   vision-capable models, available through OpenRouter, against a fixed sample
   of real captures. The comparison scores caption specificity and entity
   precision against cost per capture, measured by the Stage 3 harness.
4. **Browser coverage of the URL source.** Scripting support must be verified
   for each browser the user actually runs (Arc first). Browsers without it
   fall back to a URL-reporting extension.
5. `[Rev 4]` **Other agents' output on screen.** A coding agent's transcript
   (Conductor, Cursor, a Claude chat) is not the user's own work, but §9 only
   recognises DIscrete's own output. The live run turned the agent's
   narration into claims like "the user is improving OCR accuracy"
   (`insights.md` §7.28).
   - **Proposed:** a third origin, `agent_view`, for known agent-chat
     windows, detected by app plus pane structure. These observations are
     kept and searchable, but excluded from claim drafting, except for the
     user's own typed prompts, which are high-value intent.
   - **Until then:** users can list such windows in `SELF_APPS` or
     `SELF_WINDOW_PATTERNS`.

---

## 14. Explicitly out of scope for Stage 1

- Deduplication beyond frame tier-0, episode segmentation, claim formation,
  entity resolution, retrieval — Stage 2 and Stage 3.
- The text-completion feature itself (§7).
- Non-screen sources (email, calendar, filesystem). They are legitimate future
  connectors, but they are not specified here.

---

## 15. Overall assessment

Stage 1 now records each moment with three independent, mostly deterministic
readings: the OS's own metadata, the exact text the app exposes (or OCR where
it exposes none), and a model's description of what the user is doing. Only
the description is generated, and it can no longer overwrite the other two.
Rev 1's changes each trace to a measured failure in the
prototype's real data: thin corpus, typo-polluted app names, fused
cross-app captures, and a dedup key that could not match.
They do not trace to a hypothetical. The two genuinely open items are scope
decisions (audio; build versus buy), not design gaps.
