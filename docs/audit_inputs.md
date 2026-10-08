# DIscrete: audit of the input side (capture → observation → episode → claim)

Date: 2026-10-07. Branch `feat/memory-evolution`. No existing file was modified and no paid model calls were made.

**Evidence base**

- **Live:** `~/.cache/exp_discrete/discrete.db`, the evening session of 2026-10-06, 22:06–22:57 (0.85 h). It has 251 observations (228 `origin=user`), 9 episodes and 42 claims. The store is at schema v6, so it was written by the baseline code, before `redact.py` and before the action caps were added.
- **g46:** `bench/private/g46/manifest.jsonl`, 136 observations from 2026-09-30/10-01, about 1.1 active hours. These came from an older capture configuration: a 12 000-character cap, and AX *or* OCR rather than both merged. I did not open `base.db`, because `replay_session.py` was still running.
- **Frames:** 15 checked by eye against their stored text and caption: 14 from the live session and 1 from g46. I also re-ran Apple Vision OCR locally (`capture/ocr.py`) on all 225 live frames.
- **Scripts** (new files only, all read-only): `bench/audit/inputs_stats.py`, `claims_coverage.py`, `ax_vs_ocr.py`, `caption_novelty.py`, `show_obs.py`. Every number below can be reproduced with them.

How the labels work: **[M]** means the finding was measured here. **[O]** means opinion or an outside source, not measured on this data.

---

## 0. The short answer to the five questions

| Question | Answer |
|---|---|
| Is AX good, or too long and noisy? | **AX is the right primary source, and it is not too long.** The median `ax+ocr` row is 1.7k characters and only 1 of 227 rows hit the 16k cap. The real problems are noise and what gets hidden downstream. Noise: 43% of lines are single words (Chromium splits text into runs), 15% of lines are duplicated within a row (aria-label plus visible text), and a median 26% of AX tokens are **not visible on screen** (scrollback, truncated previews, covered pages). Downstream, the drafter sees only 51% of each episode's novel text, and none of the first 52 rows of episode 5. [M] |
| Is OCR good, or should we use a better VLM? | **Apple Vision OCR is good and cheap** (median 352 ms per frame, locally). It recovers what AX lacks: canvases, videos, and the "Helium" menu bar inside Theo's video in obs 79. It fails on small text in media because it runs on the frame **after downscaling to 1600 px**. Don't replace it with a VLM. Run it on native pixels, and keep the VLM for describing what the user is doing. [M] + [O] |
| Are preprocessing and schema correct? | The core is sound: quote-verified claims, a bi-temporal claim lineage, typed links. There are four structural gaps: (1) the drafter window silently drops the start of long episodes, (2) episode boundaries follow (app, site) rather than the task, (3) dedup collapse destroys visit times and dwell, (4) evidence and text carry no provenance (caption vs AX vs OCR, visible vs hidden). [M] |
| Is it enough for a picture of preferences? | **Not yet.** 51 minutes produced 35 "viewed/did" events, 3 facts and 4 preferences. All 4 preferences came from one Notes **sidebar preview** that was not even visible on screen. Zero preferences came from behaviour, although the session contained clear revealed choices: installing Helium, choosing DuckDuckGo out of 6 engines, making Helium the default browser, importing from Arc. [M] |
| What would make it production grade? | Close the silent-loss paths: the drafter window, the 4000-character caption cut, the 1200-character drafter cut, and head-truncation. Add provenance and status codes to every stage. Make sure DIscrete never ingests its own output. Encrypt the store at rest. Gate every change on replaying the golden sessions. Details in §4. [M] + [O] |

---

## 1. Findings, with numbers

### 1a. Screen text size and the chrome vs content split [M]

Live session (`inputs_stats.py`), user rows with text, n=227:

| text_source | n | chars median / p90 / max | lines median / p90 |
|---|---|---|---|
| ax+ocr | 114 | 1 668 / 3 915 / 16 000 | 68 / 140 |
| ocr | 107 | 846 / 1 583 / 3 015 | 42 / 64 |
| ax | 6 | 984 / 11 191 / 11 191 | 38 / 130 |

By app, median characters: Arc 1 118 (n=154), Helium 916, Finder 358, Terminal 1 429 (p90 11 191), T3 Code 2 135.

In g46, AX-only rows were far longer: median 4 092 characters, p90 11 999. That configuration had no viewport filter on some pages (whole Substack articles, YouTube feeds, WhatsApp).

**Chrome heuristic.** A line counts as chrome when it recurs in more than X% of the captures of the same **(app, site)** group, for groups with at least 5 captures. This covers 208 of 227 rows. Grouping by app alone undercounts browser chrome: Arc's sites differ, so each site's navigation is diluted.

| X | chrome share of characters | examples |
|---|---|---|
| 20% | 26% (29% over the 208 covered rows) | Arc/x.com 14%, Arc/github.com 31%, Terminal 46%, T3 Code 73% |
| 30% | **16%** (17% over covered rows) | Arc/x.com 10%, openrouter.ai 8%, Terminal 10%, T3 Code 66% |
| 50% | 8% (9%) | |

Chrome lines on x.com at X=30%: "Home timeline", "Search and explore", "Verified account", "More menu items", "Direct messages", "What's happening", trending items.

Caveat: recurrence also catches *persistent content*. T3 Code's 66% is the same chat transcript visible across captures, not chrome. Recurrence measures redundancy in general, not just chrome. **Recommended default: X=30% per (app, site) over the last ~50 captures, applied at draft and index time only, never at storage.**

Other redundancy, all measured on the same rows:

- **43% of characters** were already present in one of the previous 10 captures of the same app (41% in g46).
- **Single-word lines:** 43.6% of `ax+ocr` lines and 43.0% of `ocr` lines. For AX this is Chromium splitting text into per-word `AXStaticText` runs. Obs 57 (ChatGPT) stores "So ⏎ imagine ⏎ an ⏎ agent ⏎ has …": 201 lines for 1 670 characters.
- **Duplicate lines within one row:** 15.2% for ax+ocr and 14.8% for ax, against 0.1% for OCR. These come from the AX walk emitting aria labels next to visible text. On X, the article's aria-label repeats the whole tweet. In Helium setup (obs 97), each search engine appears twice, and the aria text "A privacy-first search engine which doesn't profile or track you" is not on screen at all. Spotify's library sidebar subtree appears twice in obs 4.

### 1b. Truncation [M]

| Stage | Where | Live session | g46 |
|---|---|---|---|
| `SCREEN_TEXT_MAX_CHARS` (16k; 12k in g46) | `accessibility.py:273`, `screen.py:355` | 1/227 rows (obs 205, Terminal) | 7/135 rows (5%) |
| Caption prompt `tk.truncate(…, 4000)` | `enrich.py:205` | 12/227 rows; 40% of their content characters cut | 31/135; 57% cut |
| Drafter `DRAFT_TEXT_CHARS_PER_OBS=1200` | `pipeline.py:191` | 98/227 rows; 55% of their content characters cut | 61/135; 76% cut |
| Drafter `DRAFT_MAX_OBS=40` | `pipeline.py:184` (`rows[-40:]`) | episode 5: **52 of 92 rows never shown** | n/a |

Across all episodes, the drafter was shown 117k of 229k characters of novel screen text, after cross-row line dedup: **51%**.

What gets cut is mostly **content, not chrome**. Every truncation keeps the *head*:

- **Terminal (obs 205):** AX returns the whole scrollback as one `AXTextArea` value, starting at "Last login…". At 16 000 characters the **newest output, which is what was on screen, is cut**. Only the OCR lines placed in front of it survive.
- **g46 WhatsApp:** the chat-list sidebar comes first in the tree, so the open conversation is cut.
- **g46 Substack:** two captures of the same article are cut at the identical byte. Anything the user scrolled to past 12k characters was lost.

The global cap is not the main problem. The drafter window and the 1200-character excerpt are, and they lose far more.

### 1c. AX vs OCR [M]

- **When AX is empty.** 107 of 228 user rows (47%) are OCR-only. 57 of these are rotated `window_change` captures, which use OCR by design (insights §7.32). The other **50 (22% of user rows)** came from ordinary click, scroll and key triggers where AX returned nothing: Arc 33, Terminal 7, Finder 2, DB Browser 1, and others. The code records no reason. Possible causes are a viewport-mismatch early return (`accessibility.py:261-262`, for example Arc's command bar as a separate panel), the 250 ms walk deadline, or `AXFocusedWindow` being None. These cannot be told apart offline. On these rows OCR did carry the content: in 15 spot checks I found no OCR-only row where the caption plus OCR missed the page's main text.
- **AX text not visible on screen.** I re-ran OCR on each saved frame and measured the share of stored content tokens absent from that OCR. The noise floor is a median of **0.02** for `ocr` rows. `ax+ocr` rows have a median of **0.26** and a mean of 0.29 (n=112). By app: Arc 0.29, T3 Code 0.25, Notes 0.38, TextEdit 0.73. Worst cases:
  - obs 20: AX read the X page **underneath** a black loading overlay (0.97).
  - obs 205: Terminal scrollback (0.80).
  - obs 208, Notes: the sidebar preview of the note "Travel prefs" shows only "I always p…" on screen, but AX exposes the full hidden text "I always prefer aisle seats, vegetarian meals, …". **All 4 preference claims (39–42) were drafted from that unseen text.** They are probably true, since the user wrote the note, but nobody *saw* them in this session.
- **What OCR adds.** The OCR-novel lines are placed *first* (`screen.py:340-355`), and that design is validated. In obs 79 (22:20:28), OCR read **"Helium ⏎ File ⏎ Edit ⏎ View ⏎ Tabs …"** from the menu bar inside Theo's video, 20 s before the user searched Google for "helium". That line is the causal evidence for "why did I install Helium", and it survived because it went first.
- **OCR limits.** In obs 78 the same video was smaller, and the menu bar is unreadable at 1600 px. Upscaled re-OCR of the crop (2–4×) returned garbage: the detail is lost when the frame is captured at `FRAME_MAX_WIDTH=1600` and saved as JPEG at quality 65. `frames.py:116` caps width at 1600, so OCR runs on downscaled pixels for any window wider than 800 pt. Tables lose their structure: in obs 146 (EmbeddingGemma 2 benchmark), OCR reads column by column and numbers come apart from their labels. The caption restored the meaning there.
- **Text visible in the frame but absent from both text and caption** (15 frames): the Helium menu bar in the obs 78 video (both); "Helium" in the obs 79 and obs 101 videos (absent from the caption only); the table structure in obs 146 (text only); thumbnail text on the YouTube feed in g46 (titles were present via AX). Text inside video and images is the systematic gap.

### 1d. Captions [M]

- **Information added.** I counted caption content tokens that appear in neither the screen text, the title, nor the URL, after removing boilerplate such as "user", "viewing" and "scrolling". The median is **38%**. 163 of 230 captions have at least 30% novel tokens. Only 1 of 230 merely restates the title. 115 of 230 name a user action (typed, selected, scrolled, playing…). Part of the "novel" share is paraphrase ("feed", "embedded"), so this is an upper bound. The spot checks, however, show real added value:
  - state: "video paused at 0:04", "response still loading", "dialog open";
  - action: "typing a misspelled query 'artiffical inde'", "selected the Helium disk image";
  - meaning: table semantics in obs 146.
- **Accuracy in spot checks.** Of 14 live captions, **12 were accurate**. Obs 101 had the wrong page identity: a video *inside* Theo's X post was captioned as "viewing the GitHub pull request …", with `page_title` set to the PR's title. In obs 220, `navigated_from='claims'` is a speculative detail. I saw **no invented content**. Three captions (obs 78, 79, 101) omitted the decisive in-video brand "Helium". In 79 and 101 it was legible and OCR read it. Captions report "no interaction" from the counts they were given; counts on collapsed rows change afterwards (obs 87: dup=5, clicks=4). This is cosmetic.
- **Privacy note.** Obs 66 captioned a passcode entry ("typed a 4-digit passcode"). The digits were not stored.
- **Cost and latency** (perf tables): 253 calls cost **$0.0715**, or $0.00028 per call and about $0.084 per hour at 295 rows per hour. Average vision latency is **8.96 s**, the maximum 73 s, and queue wait 4.5 s. Calls average **436 completion tokens for roughly 60 tokens of answer**: the reasoning model's thinking dominates.

### 1e. Claims: where evidence comes from, and what was missed [M]

`claims_coverage.py`:

- **Field holding the quote.** 34 of 42 quotes were found in the **caption only**, 7 in screen_text, and 1 in screen_text plus title. **81% of the "verbatim evidence" quotes a model-written caption, not ground-truth screen text.** Cited rows by text_source: ocr 16, ax+ocr 23, ax 3.
- **Coverage.** 37 of 231 episode-assigned user observations (**16%**) are cited by any claim. Of 155 engaged rows (keys, clicks or copies), 20 are cited. Every claim cites exactly one observation.
- **Claim mix.** 35 events, 3 facts, 4 preferences. The 4 preferences all come from obs 208, the Notes preview not visible on screen (see 1c).
- **Self-ingestion.** Claims 32–34 came from obs 205 and 207, which are the Terminal output of `jarvis ask "Which browser did user use?"`, i.e. DIscrete's own answer. Claim 34 ("The user was setting up Helium as a browser…") is built from DIscrete's own recall output. `classify_origin` (`suppress.py:42`) only matches `SELF_WINDOW_PATTERNS=["bin/discrete","discrete run"]` and in-process text. The CLI's title is "python3 ◂ jarvis ask …", and after exit it is just "-zsh".
- **The Helium case, corrected.** The download had no claim **because the drafter never saw it, not because of the event cap.**
  - Episode 5 had 92 observations. `_call_draft` sends only `rows[-40:]`, which are rows 53–92 (22:25:46 onward).
  - The whole sequence fell in the 52 dropped rows: the trigger (obs 79, "Helium File Edit…" OCR'd from the video), the Google search (82–83), the site (84–86), "Thank you for downloading Helium!" (87, clean AX text), Finder showing `helium_0.18.3.1_arm64-macos.dmg … Today at 10:21 PM` (89), the DMG drag-to-Applications window (90), and the setup wizard (95–110).
  - The `metrics` table shows `claims_rejected_over_cap = 0` and only 2 `not_entailed`. The baseline cap was 5 events, and episode 5 drafted exactly 5, all from the last 40 rows (132, 146, 152, 158, 165).
  - **The current code still has `DRAFT_MAX_OBS=40`**, so the action-first prompt fix (d6ecb5b) would not have saved this case. `trails.py` needs an action claim to start from, so the "why" chain never begins.
- **Other activity with no claim.** All in episode 5:
  - chose DuckDuckGo as search engine (97);
  - imported from Arc and Chrome, then hit "Import profile still in use" (104–106);
  - selected "Yes, make Helium my default browser" (109);
  - customised the Helium toolbar via Settings (122–127);
  - entered the X DM passcode and read a DM thread with a friend (66–68);
  - browsed Agently / Omar Ghandour's posts (70–76).

  Episode 9 (23 rows) was still open in the snapshot.

### 1f. Episodes [M]

There are 9 episodes in 51 minutes, lasting 1.6, 1.9, 2.0, 2.9, 4.1, 5.1, 6.9 and 12.6 minutes (plus the open one). **None comes near the 30-minute cap.** The binding limit is the 40-observation drafter window: at 295 rows per hour, that is about 8 minutes of activity.

- **Over-split.** A single research thread on the "Agent Memory Repo" (cognition.com → GitHub → ChatGPT) became episodes 2, 3 and 4. Each sustained site switch of 90 s or more forces a boundary.
- **Under-split.** Episode 5 runs 12.6 minutes with 92 rows, 4 apps and 6 sites. It covers five or more topics: X feed, X DMs, Helium install and setup, the OpenRouter embeddings list, Terminal logs, Artificial Analysis. Each excursion either returned to (Arc, x.com) and was "absorbed" (`episodes.py:77-78`), or reset the pending switch by moving to a *different* new pair (`episodes.py:86`). Within Helium alone, the site changed between '' and github.com, so the 90 s timer never matured.
- **Overlapping intervals.** Episode 7 (Terminal, 22:33:53–22:35:27) lies inside episode 6's interval. Obs 203 in episode 6 has dup_count=6, and later sightings of the same OpenRouter page extended its `ended_at` to 22:41:36 (see 1g).

### 1g. Dedup, heartbeats, storage [M]

- **Collapsed captures.** 120 captures merged into existing rows: 371 captures became 251 rows, **32% collapsed**. 39 of these were perceptual-hash frame duplicates. Only 1 heartbeat was stored, plus 2 light ones. 19 rows were discarded as not useful.
- **Residual near-duplicates.** Among consecutive same-(app, site) rows, trigram Jaccard was at least 0.8 for 1/227, at least 0.7 for 9, and at least 0.5 for 22. **Dedup works well.**
- **The cost of collapsing.** `_bump` (`observations.py:44`) sums counts and extends `ended_at`, but keeps **no record of each sighting**. Obs 203's interval spans 8 minutes during which the user was in Terminal and Notes. Dwell time and episode intervals are wrong as a result.
- **Storage per active hour:**
  - 295 rows;
  - screen_text 0.42 MB, captions 50 KB;
  - frames 24.7 MB (88 KB average; 240 files);
  - the whole DB, including FTS and embeddings, about 5 MB (4.4 MB for 0.85 h).
  - With `FRAME_TTL_DAYS=3` and 8 h days, frames come to about 600 MB. With `SCREEN_TEXT_TTL_DAYS=90`, text comes to about 300 MB plus FTS.
  - For comparison, Screenpipe reports 5–10 GB per month.
- **Capture latency.** Local work (`capture.text`, AX plus OCR) averages 459 ms with a maximum of 1.12 s, of which OCR is about 350 ms. The lag from capture to segmentation averages 15.5 s.

### 1h. Security finding [M]

**Obs 239 (TextEdit, a `.env` file) in the live snapshot DB holds 4 strings shaped like OpenRouter API keys** in `screen_text`, and its frame on disk shows them too. The current `capture/redact.py` removes all 4 when run offline on that text. The snapshot simply predates the redactor.

Action: **rotate that key**, and delete the snapshot row and the frame `~/.cache/discrete/frames/…` for obs 239. The key is not reproduced in this report.

Related risks [M]:

- AX exposes hidden previews, such as the Notes sidebar preview "2FA codes:" in obs 208.
- A passcode screen was captured (obs 66). Only the caption mentions it.

### 1i. What is already good [M]

- Event-driven capture, navigation as a capture boundary, and page identity from `AXWebArea`: 206 of 228 user rows have `page_title`, and browser rows have URLs.
- AX first, with OCR-novel lines prepended. This is the right combination, and obs 79 shows its value.
- Text is not too long at capture time: the median row is under 2k characters, and 1 row in 227 hits the cap.
- Dedup collapses about a third of captures, with few near-duplicates left.
- Captions are accurate in 12 of 14 spot checks, with no invented content.
- Claims are well formed and quote-checked, and the verifier is independent. Apart from the self-ingestion case, I saw no false events.
- The whole pipeline cost **about $0.15** for 51 minutes; captions were half of that.

---

## 2. Research summary: how comparable products capture activity

| Product or work | Capture | Representation | Notes |
|---|---|---|---|
| **Screenpipe** (open source, Rust) | Event-driven: app switch, focus, click/scroll, typing pause, clipboard, plus an idle capture about every 5 s. **Accessibility tree first, OCR only when AX is empty** (Apple Vision on macOS) | SQLite: `frames` (AX text, or OCR in `full_text`), **`elements`** (UI elements from the AX tree), `ui_events`, audio tables, FTS5 | Almost the same design as DIscrete. Its separate `elements` table keeps structure, where DIscrete flattens. Reports 5–10 GB per month and 5–10% CPU. [docs](https://docs.screenpipe.com/architecture), [overview](https://mintlify.com/screenpipe/screenpipe/architecture) |
| **Microsoft Recall** | Snapshot about every 5 s when the screen changes. Local models: screen-region detector, **OCR**, image encoder | "Windows Semantic Index", a local vector store | Sensitive-content filter (Microsoft Classification Engine) skips snapshots containing cards or passwords, but [tests found gaps](https://www.pcworld.com/article/2870013/windows-recall-still-screenshots-sensitive-data-at-times-test-shows.html). **Encrypted at rest, with keys in the TPM, processing in a VBS enclave, and Windows Hello gating every session** ([MS blog](https://blogs.windows.com/windowsexperience/2024/09/27/update-on-recall-security-and-privacy-architecture/)). [how it works](https://www.windowscentral.com/software-apps/windows-11/how-does-windows-recall-work) |
| **Rewind → Limitless** | Continuous screen video, heavily compressed (claimed up to 3 750×), **OCR on video**, local Whisper | Searchable OCR text linked to the moment | Q&A sends only the text of relevant moments to an LLM. Now pivoted to cloud plus wearable. [review](https://thesweetsetup.com/a-first-look-at-rewind-ai/), [overview](https://thesweetbits.com/tools/rewind-ai) |
| **Dayflow** (open source, macOS) | Records at 1 FPS; **a VLM summarises every 15 minutes** (Gemini, or local via Ollama/LM Studio) | A timeline of activity cards | VLM-first and batched: cheap per hour, coarse in time. [README](https://cdn.jsdelivr.net/gh/jerryzliu/dayflow@main/README.md), [write-up](https://virtuslab.com/blog/ai/dayflow-your-private-git-log/) |
| **Highlight** | Continuous screen awareness built on Medal's capture layer; "captures but doesn't store" | On demand, as context for chat | [product](https://highlightai.com/product), [news](https://www.vktr.com/the-wire/medal-unveils-highlight-a-revolutionary-ai-assistant-platform-raises-13m-at-333m-valuation) |
| **Granola** | No screen capture at all: mic plus system audio, transcribed live, **audio deleted** | User notes enhanced from the transcript | A high-signal, narrow channel; useful for meetings. [docs](https://docs.granola.ai/help-center/taking-notes/transcription), [analysis](https://zackproser.com/blog/how-does-granola-ai-work) |
| **GUM, General User Models** (Stanford/MSR, 2025) | Screenshot on I/O events; **a VLM (Qwen2.5-VL 72B) transcribes the changes into text** | Confidence-weighted **propositions** about the user, retrieved and **revised** as observations arrive; a privacy **audit** step based on contextual integrity | The closest research analogue to DIscrete's claims. Proposition accuracy 76% overall and **88% at confidence above 0.8** (N=18); Brier score 0.17. [arXiv 2505.10831](https://arxiv.org/abs/2505.10831) |
| **Computer-use agents** (OSWorld) | Screenshot, AX tree, or both (Set-of-Marks) | Per-step observation | Combining AX with the screenshot helps (GPT-4o: 11.2% with a11y vs 5.0% without), but **AX trees cost 3–26 s and thousands of tokens** including hidden elements ([OSWorld-Human](https://arxiv.org/pdf/2506.16042), [OSWorld](https://arxiv.org/pdf/2404.07972)). **Compressed AX trees used 22% of the tokens with +5.1 pp success** ([A11y-Compressor](https://arxiv.org/pdf/2605.00551)). This supports §1a: prune and dedup the tree, don't drop it. |
| **OmniParser v2** | Pure-vision UI parsing: a YOLO detector plus a Florence-2 captioner | Structured elements | 39.6% on ScreenSpot-Pro, about 0.6 s per frame on an A100 ([MS](https://labs.ai.azure.com/innovations/omniparserv2/), [paper](https://arxiv.org/pdf/2408.00203)). Too heavy, and the wrong goal for memory: it finds interactable targets, not content. |
| **ScreenSpot-Pro** | Benchmark | n/a | Targets average 0.07% of the screen; the best grounding models scored 18.9%, and a cascaded zoom search scored 48.1% ([arXiv 2504.07981](https://arxiv.org/pdf/2504.07981)). **VLMs miss small high-resolution detail unless they crop and zoom**, the same failure as the obs 78/79 captions at 1024 px "low" detail. |

**AX vs OCR vs VLM, synthesised** [O, from the sources above plus the measurements here]:

| | Accuracy | Cost | Latency | Privacy |
|---|---|---|---|---|
| AX | Exact text and roles, but includes hidden, offscreen and aria text (26% here), word-split in Chromium, and nothing for canvases, video or images | Free | Tens to hundreds of ms; 250 ms deadline here | Exposes *more* than the screen shows (previews, scrollback) |
| Apple Vision OCR | Very good on rendered text; fails below about 1/32 of image height ([Apple notes](https://developer.apple.com/forums/thread/695693)), so small media text is lost after downscaling; flattens tables | Free, local | About 350 ms at 1600 px here | Only what is visible |
| VLM caption | The only source of *activity and state* ("typing", "paused", "selected"); misses small detail at 1024 px "low"; occasionally misattributes embedded media | About $0.00028 per call here | 9 s average here | Pixels leave the device unless the model runs locally |

Apple's `RecognizeDocumentsRequest` (macOS 26, [WWDC25 session 272](https://developer.apple.com/videos/play/wwdc2025/272/)) returns tables, lists and paragraphs, which would fix the obs 146 table flattening at no API cost.

**Main-content extraction.** On web pages, Readability (F1 0.947) and Trafilatura (0.958) are the reference extractors ([Trafilatura evaluation](https://trafilatura.readthedocs.io/en/latest/evaluation.html)). Using them requires the DOM, which means a browser extension. The extension-free equivalent is to walk only the `AXWebArea` subtree, keep elements whose bounds fall in the viewport, skip landmark and navigation roles, and merge text runs per line.

Chromium and Electron build the AX tree only on request (`AXManualAccessibility`), and doing so has a real performance cost for the inspected app ([Electron docs](https://electronjs.org/fr/docs/latest/tutorial/accessibility), [Chromium a11y overview](https://chromium.googlesource.com/chromium/src/+/56ed97b900f0943480077ee4fa1b94de137e97ff/docs/accessibility/overview.md)). Keep the walk scoped.

---

## 3. Schema review (`store/db.py`) against the objective

**Sound:**

- `observations` keeps ground-truth fields (app, bundle, title, URL, site) apart from model fields (caption, page_title, is_listing, navigated_from).
- `evidence` uses quote-level links.
- `claims` is bi-temporal (`occurred_*`, `valid_*`) with lineage, version and status, plus `action`.
- `links` gives typed edges for causality and trails.
- `memory_events` provides a change log with undo, and `suppressions` handles forgetting.
- The outbox, FTS5 and perf tables are in place.

**Missing, ordered by impact on preferences and "why" questions:**

1. **Sightings and dwell (attention)** [M: obs 203, episode overlap]. A collapsed row loses its visit times. Add `observation_sightings(obs_id, started_at, ended_at, keys, clicks, scrolls)`, written in `_bump`, and compute *active* dwell: foreground time with input within N seconds. Attention is the strongest implicit preference signal available, and it currently cannot be computed.
2. **Pages and documents as first-class objects** [O]. Add `pages(id, canonical_url, site, title, kind[article|repo|video|product|chat|file|listing], first_seen, last_seen, visits, active_dwell_s, engaged)` and `observations.page_id`. Files and documents come from `AXDocument` (TextEdit, Finder, Preview). This answers "what did I read most", "which repos do I return to", "which docs did I edit" without an LLM.
3. **Text provenance** [M: 26% hidden AX tokens, 81% caption-only quotes]. Store `ax_text` and `ocr_text` separately, or as `segments(source, role, visible, bbox)`. Add `evidence.field` (caption | screen_text | title | url) and `evidence.visible`. Add `observations.ax_status` (ok | untrusted | no_window | viewport_mismatch | timeout | empty), plus `ax_nodes`, `ax_ms`, `ocr_ms`, `caption_model` and `pipeline_version` so that rows can be reprocessed.
4. **Revealed choices and actions as structured signals** [M: the Helium setup choices were never recorded]. Add `signals(id, kind[choice|install|purchase|search|explicit_statement|dwell|repeat_visit], subject, object_entity_id, alternatives JSON, obs_id, at, confidence)`, filled by deterministic detectors plus the drafter. Preferences should then be *induced* across days from signals (dream.py). Today they are drafted from a single screen.
5. **Tasks and projects above episodes** [M: one research thread split into 3 episodes]. Add `tasks(id, title, status, first_seen, last_seen)` and an episode–task mapping, or reuse `entities(kind='project')` with episode mentions.
6. **Preference slots** [O]. Add `claims.predicate`, `claims.object_entity_id`, `claims.polarity` and `claims.scope`, so that "prefers DuckDuckGo for search" and "prefers Google" collide on a slot rather than relying on embedding similarity alone.

**Redundant or underused** [M: live store]:

- `claims.reasoning` was empty in 42 of 42 rows (it is used only by gaps and dream).
- `claims.refines_id`, `invalidates_id` and `superseded_by` were unused (0 of 42), and overlap with `links.relation` since m007. Choose one representation.
- `observations.entities` (JSON), `entity_candidates` and `mentions` hold three copies of mention data, and the live store promoted **0 entities** out of 80 candidates.
- `content_hash` vs `text_hash`: two hashes of overlapping text; keep both only if both are queried.
- The observation FTS indexes raw `screen_text`, chrome included. That is fine for recall, but it is noise for ranking; index the chrome-filtered text instead, or in addition.

---

## 4. Ranked recommendations

Effort: S is under a day, M is 1–3 days, L is a week or more. Costs assume the live session's rates.

| # | Recommendation | Backing | Expected impact | Cost / latency | Effort | Where |
|---|---|---|---|---|---|---|
| 1 | **Stop the drafter from dropping rows.** Either close an episode once `st.n >= DRAFT_MAX_OBS`, or draft long episodes in consecutive windows of at most 40 rows, carrying the previous window's summary forward. Always include engaged rows and action-candidate rows. Add a replay test built on episode 5. | [M] 52/92 rows of episode 5, including the whole Helium chain, never reached the model | Restores 23% of episode rows in this session. The Helium download, install, setup choices and the obs 79 trigger become draftable, so `trails.py` can answer "why" | +1 draft and verify per 40 rows (about $0.005); no capture latency | S | `process/pipeline.py:184` (`rows[-T.DRAFT_MAX_OBS:]`), `process/episodes.py:62-112`, `config.py:346` |
| 2 | **Never ingest DIscrete's own output.** The CLI and Bubble write each answer's text to a small `self_outputs` table (with a TTL) that `classify_origin` checks by trigram containment, as it already does in-process. Add `"jarvis "` to the default `SELF_WINDOW_PATTERNS`. | [M] claims 32–34 came from `jarvis ask` output (obs 205, 207) | Removes a feedback loop in which memory confirms itself | Negligible | S | `capture/suppress.py:12-54`, `cli.py` (answer printing), `config.py` `SELF_WINDOW_PATTERNS` |
| 3 | **Rotate the leaked key and purge obs 239** from the snapshot (row and frame). Then encrypt the store at rest (SQLCipher, or a Keychain-wrapped key) and frames (per-file AES, key in the Keychain), gated by Touch ID for exports and debugging views, as Recall does. | [M] 4 key-shaped strings in obs 239; [O] Recall design | Closes the worst-case breach; a prerequisite for "production grade" | Under 5% CPU for encryption [O] | S (rotate) / M (encryption) | `store/db.py:connect`, `capture/screen.py:_save`, `capture/retention.py` |
| 4 | **Clean AX at the source and keep only visible text.** (a) Merge consecutive `AXStaticText` runs that share a parent or line into one line. (b) Read `AXDescription` only when there is no `AXValue`/`AXTitle` and no text children (this drops aria duplicates). (c) For `AXTextArea`/`AXTextView`, read `AXVisibleCharacterRange` → `AXStringForRange` instead of the whole value (Terminal scrollback, editors). (d) Skip landmark and navigation roles (`AXLandmarkNavigation`, `AXList` inside `AXLandmarkBanner`). (e) If truncation is still needed, keep the viewport text first and cut from the tail of the offscreen remainder. | [M] 43% single-word lines, 15% duplicate lines, median 26% of tokens not visible, obs 205 newest output cut, obs 208 hidden preview → 4 claims | Shorter and cleaner text, quotes that read as sentences, no claims from unseen text | Fewer tokens downstream; AX walk time unchanged or lower | M | `capture/accessibility.py:223-273` (`_walk_text`, `read_screen_text`), `screen.py:340-355` |
| 5 | **Record provenance** for text segments (AX/OCR, visible flag), for evidence (field), and for capture status (`ax_status`, timings, model versions). Weight confidence lower when evidence is caption-only. Tell the drafter to "prefer quoting screen text". | [M] 81% of quotes are caption-only; 50 rows with empty AX and no reason | Auditable claims, debuggable capture failures, and the ability to reprocess when models change | Negligible | S–M | `store/db.py` (new migration m008), `process/validation.py:24-45`, `process/prompts.py` (drafter rules), `screen.py:_screen_text` |
| 6 | **Filter chrome and repetition before the drafter and the index** (not before storage). Use an (app, site) line-recurrence filter (X=30%, last 50 captures), then raise `DRAFT_TEXT_CHARS_PER_OBS` to about 2 000 for the remaining content. In `enrich.py`, send the chrome-filtered text in place of the first 4 000 raw characters. | [M] chrome 16% (X=30%) to 26% (X=20%); 43% repeated from recent captures; drafter shown 51% of novel text, caption prompt cut 40–57% of content in long rows | The drafter and the VLM see content rather than navigation | Same or fewer tokens | S | `process/pipeline.py:181-200`, `capture/enrich.py:199-206`, a small `textkit` helper |
| 7 | **Run OCR on native-resolution pixels.** Grab the window at full backing scale, OCR that, then downscale to 1600 for storage and the VLM. Where available, use `RecognizeDocumentsRequest` for tables. Optionally re-OCR the regions of detected video and image elements at 2× crop. | [M] "Helium" unreadable in the obs 78 video at 1600 px and unrecoverable by upscaling, legible in 79/101; table flattened in 146 | Recovers small text in media (the "why" triggers) and table structure | OCR cost rises from about 350 ms to about 0.8–1.5 s off the event loop [O, estimate]; skip on frame duplicates | S–M | `capture/frames.py:85-120` (`_grab_window` width cap), `screen.py:349-353`, `capture/ocr.py` |
| 8 | **Segment episodes by task, not by (app, site).** Count time spent *away from* the episode context in any pair (do not reset on each new pair). Allow semantic-shift splits across pairs. Merge adjacent episodes with high embedding similarity, which fixes episodes 2–4. Add a `tasks` layer. | [M] episodes 2–4 over-split; episode 5 under-split (6 sites, 5+ topics); episode 7 overlap | Coherent episodes, better titles and summaries, fewer cross-topic claims | One embedding comparison per row (vectors already exist) | M | `process/episodes.py:62-112` (lines 77–86) |
| 9 | **Sightings, pages and active dwell** (schema items 1–2). Write a sighting in `_bump`. Derive `pages` and dwell, and use dwell plus repeat visits as preference signals in `dream.py`. | [M] obs 203 interval spans 8 minutes; no dwell possible | Interest and habit inference ("reads OpenRouter model lists daily"); correct intervals | Small storage | M | `store/observations.py:44-52`, `store/db.py` (m008), `process/dream.py` |
| 10 | **Deterministic action and choice detectors** for: "Thank you for downloading", a new file in Finder Downloads with date added today, a `.dmg` window, a setup or installer wizard step with a selected option, sent or posted confirmations, order confirmations. Each marks rows `action_candidate` and writes a `signals` row. | [M] obs 87, 89, 90, 97, 109 carry this verbatim in AX or OCR text; [O] detector design | Actions and choices recorded even when the drafter misses them; trails get their anchors | Regex and rules, under 1 ms per row | M | new `process/actions.py`, called from `pipeline._draft`; `store/db.py` |
| 11 | **Re-scope the caption, but do not switch models blindly.** In the prompt: "distinguish the page from embedded video or images; name apps, brands and products visible inside media". Set `CAPTION_LM_REASONING_EFFORT=minimal` (436 completion tokens for about 60 tokens of answer). Skip captions for rows whose page was already captioned in the last few minutes with less than 20% text change. Benchmark 2–3 fast non-reasoning VLMs on the 15 spot-check frames (listed in §5) before switching. | [M] obs 101 misattributed; 78/79/101 omitted "Helium"; 9 s average latency | Roughly halves caption latency and cost [O]; fewer page-identity errors | Lower | S | `capture/prompts.py`, `.env` / `config.py` caption endpoint, `capture/enrich.py:_apply` |
| 12 | **Production hygiene** [O]. (a) Turn the golden replays (g46, and this session exported as a manifest) into a CI gate. Assertions: the Helium download claim exists, no claim cites a `self` row, chrome share stays under the threshold, nothing is truncated without a flag. (b) Per-stage SLOs with alerts from `perf_samples`: capture.text p90 under 1.2 s, enrich.total p90 under 30 s, segment.lag p90 under 60 s. (c) A health counter for every silent-loss path (drafter window, truncations, AX empty by reason). (d) A browser extension, as an *optional* source of DOM main content via Readability/Trafilatura and the visible scroll range, with AX as the fallback. | [M] the silent losses above; [O] industry practice | Regressions get caught before they ship | n/a | M (a–c) / L (d) | `bench/`, `discrete/evaluation/harness.py`, `store/health.py`, `store/perf.py` |

**Not recommended** [O]:

- Replacing AX with a VLM transcription, as GUM does. It costs about 30× more per row, is slower, and loses exact text for quoting.
- OmniParser-style element parsing. It targets clicking, not memory.
- A global switch to a "better VLM" without the frame benchmark in item 11. The captions were mostly right; their errors were small-detail and scoping problems, which items 7 and 11 address more cheaply.

---

## 5. Reproduction

```
.venv/bin/python bench/audit/inputs_stats.py                       # 1a, 1b, 1g (live)
.venv/bin/python bench/audit/inputs_stats.py --manifest bench/private/g46/manifest.jsonl --max-chars 12000
.venv/bin/python bench/audit/claims_coverage.py                    # 1e, 1f
.venv/bin/python bench/audit/ax_vs_ocr.py                          # 1a fragmentation, 1c hidden text, OCR latency (~90 s)
.venv/bin/python bench/audit/caption_novelty.py                    # 1d
.venv/bin/python bench/audit/show_obs.py 79 87 89 101 205 208      # spot checks
```

Frames checked by eye:

- **Live:** 3, 4, 20, 54, 78, 79, 89, 90, 97, 101, 146, 163, 208, 220.
- **g46:** `d3a94d008181.jpg`.

For the chrome split by (app, site) in 1a, run `inputs_stats.py --group-site`. Without the flag, the script groups by app only (16% at X=20%, 10% at X=30%), which undercounts browser chrome.
