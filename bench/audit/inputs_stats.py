"""Input-side audit statistics over a DIscrete store, read-only (docs/audit_inputs.md).

    .venv/bin/python bench/audit/inputs_stats.py [--db ~/.cache/exp_discrete/discrete.db]
    .venv/bin/python bench/audit/inputs_stats.py --manifest bench/private/g46/manifest.jsonl

Chrome heuristic: a normalized line is "chrome" for an app when it recurs in more
than CHROME_SHARE of that app's stored captures (apps with >= MIN_CAPTURES rows).
Everything else is "content". Short lines (<= 3 words) are reported separately as
a label-shaped proxy that needs no recurrence.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from discrete import textkit as tk  # noqa: E402
from discrete.encryption import open_database  # noqa: E402

MIN_CAPTURES = 5


def load_db(path: str) -> list[dict]:
    conn = open_database(path, read_only=True)
    rows = [dict(r) for r in conn.execute("SELECT * FROM observations ORDER BY started_at, id")]
    conn.close()
    return rows


def load_manifest(path: str) -> list[dict]:
    rows = []
    for n, line in enumerate(open(path)):
        rec = json.loads(line)
        d = dict(rec.get("draft") or {})
        d.setdefault("id", n)
        d.setdefault("origin", "user")
        d["frame_path"] = rec.get("image", "")
        rows.append(d)
    return rows


def lines_of(text: str) -> list[str]:
    return [ln for ln in (text or "").splitlines() if tk.norm_key(ln)]


GROUP_BY_SITE = False


def group_key(r: dict):
    return (r["app"], r.get("site") or "") if GROUP_BY_SITE else r["app"]


def chrome_sets(rows: list[dict], share: float) -> dict:
    by_app: dict = collections.defaultdict(list)
    for r in rows:
        by_app[group_key(r)].append(r)
    out: dict[str, set[str]] = {}
    for app, rs in by_app.items():
        if len(rs) < MIN_CAPTURES:
            out[app] = set()
            continue
        df = collections.Counter()
        for r in rs:
            df.update({tk.norm_key(ln) for ln in lines_of(r["screen_text"])})
        out[app] = {k for k, c in df.items() if c / len(rs) > share}
    return out


def split(text: str, chrome: set[str]) -> tuple[int, int, int]:
    """(chrome chars, content chars, short-label chars) of one screen text."""
    ch = co = short = 0
    for ln in lines_of(text):
        n = len(ln)
        if tk.norm_key(ln) in chrome:
            ch += n
        else:
            co += n
        if len(tk.tokens(ln)) <= 3:
            short += n
    return ch, co, short


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def q(xs: list[float], p: float) -> float:
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/.cache/exp_discrete/discrete.db")
    ap.add_argument("--manifest")
    ap.add_argument("--max-chars", type=int, default=16000)
    ap.add_argument("--group-site", action="store_true", help="chrome recurrence per (app, site), not per app")
    args = ap.parse_args()
    global GROUP_BY_SITE
    GROUP_BY_SITE = args.group_site
    rows = load_manifest(args.manifest) if args.manifest else load_db(args.db)
    user = [r for r in rows if r.get("origin", "user") == "user" and (r.get("screen_text") or "").strip()]
    print(f"rows={len(rows)} user_with_text={len(user)}")

    # ---- (a) size by app and by text_source
    print("\n## size by text_source")
    by_src = collections.defaultdict(list)
    for r in user:
        by_src[r.get("text_source", "")].append(r)
    for src, rs in sorted(by_src.items(), key=lambda x: -len(x[1])):
        ch = [len(r["screen_text"]) for r in rs]
        ln = [len(lines_of(r["screen_text"])) for r in rs]
        print(f"{src or '(none)':8} n={len(rs):3} chars median={statistics.median(ch):6.0f} p90={q(ch, .9):6.0f} "
              f"max={max(ch):6}  lines median={statistics.median(ln):4.0f} p90={q(ln, .9):4.0f}")
    print("\n## size by app")
    by_app = collections.defaultdict(list)
    for r in user:
        by_app[r["app"]].append(r)
    for app, rs in sorted(by_app.items(), key=lambda x: -len(x[1])):
        ch = [len(r["screen_text"]) for r in rs]
        srcs = collections.Counter(r.get("text_source", "") for r in rs)
        print(f"{app[:24]:24} n={len(rs):3} median={statistics.median(ch):6.0f} p90={q(ch, .9):6.0f} {dict(srcs)}")

    # ---- chrome vs content at several thresholds
    print("\n## chrome share (line recurs in > X of the app's captures)")
    for share in (0.2, 0.3, 0.5):
        cs = chrome_sets(user, share)
        tot_ch = tot_co = 0
        per_app = collections.defaultdict(lambda: [0, 0])
        for r in user:
            ch, co, _ = split(r["screen_text"], cs[group_key(r)])
            tot_ch += ch
            tot_co += co
            per_app[group_key(r)][0] += ch
            per_app[group_key(r)][1] += co
        apps = ", ".join(f"{str(a)[:28]}={pct(v[0], sum(v))}" for a, v in sorted(per_app.items(), key=lambda x: -sum(x[1]))[:6])
        print(f"X={share:.0%}: chrome {pct(tot_ch, tot_ch + tot_co)} of chars ({tot_ch}/{tot_ch + tot_co}); {apps}")
    cs = chrome_sets(user, 0.3)
    short = sum(split(r["screen_text"], set())[2] for r in user)
    tot = sum(len("".join(lines_of(r["screen_text"]))) for r in user)
    print(f"short lines (<=3 words): {pct(short, tot)} of chars")
    print("\nexample chrome lines (X=30%):")
    for app, s in sorted(cs.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"  {app}: {len(s)} lines e.g. {sorted(s, key=len, reverse=True)[:12]}")

    # Within-session repetition: share of a row's lines already seen in the previous
    # 10 captures of the same app (what an incremental store would not need again).
    seen_prev = collections.defaultdict(lambda: collections.deque(maxlen=10))
    rep = new = 0
    for r in user:
        prev = set().union(*seen_prev[r["app"]]) if seen_prev[r["app"]] else set()
        ks = [tk.norm_key(ln) for ln in lines_of(r["screen_text"])]
        for k, ln in zip(ks, lines_of(r["screen_text"]), strict=True):
            if k in prev:
                rep += len(ln)
            else:
                new += len(ln)
        seen_prev[r["app"]].append(set(ks))
    print(f"\nlines already present in the previous 10 captures of the same app: {pct(rep, rep + new)} of chars")

    # ---- (b) truncation at every stage
    print("\n## truncation")
    hits = [r for r in user if len(r["screen_text"]) >= args.max_chars - 1]
    print(f"SCREEN_TEXT_MAX_CHARS={args.max_chars}: {len(hits)}/{len(user)} rows at the cap: "
          f"{[(r['id'], r['app']) for r in hits]}")
    for lim, name in ((4000, "caption prompt (enrich.py tk.truncate 4000)"),
                      (1200, "draft excerpt (DRAFT_TEXT_CHARS_PER_OBS)")):
        cut = [r for r in user if len(r["screen_text"]) > lim]
        lost_content = 0
        tot_content = 0
        for r in cut:
            _, co_all, _ = split(r["screen_text"], cs[group_key(r)])
            _, co_kept, _ = split(r["screen_text"][:lim], cs[group_key(r)])
            lost_content += co_all - co_kept
            tot_content += co_all
        print(f"{name}: {len(cut)}/{len(user)} rows exceed; content chars lost in those rows "
              f"{pct(lost_content, tot_content)}")
    # chrome share of the first 1200 chars (what the drafter is shown before cross-row dedup)
    ch1 = co1 = 0
    for r in user:
        ch, co, _ = split(r["screen_text"][:1200], cs[group_key(r)])
        ch1 += ch
        co1 += co
    print(f"first 1200 chars of each row: chrome {pct(ch1, ch1 + co1)}")

    # ---- (g) duplicates and storage
    print("\n## dedup / near-duplicates")
    dups = sum(r.get("dup_count", 1) - 1 for r in rows)
    print(f"collapsed into existing rows (dup_count-1): {dups}; stored rows {len(rows)}")
    near = collections.Counter()
    last: dict[tuple, frozenset] = {}
    for r in user:
        key = (r["app"], r.get("site", ""))
        g = tk.trigrams(r["screen_text"])
        if key in last:
            j = tk.jaccard(g, last[key])
            for t in (0.5, 0.7, 0.8):
                if j >= t:
                    near[t] += 1
        last[key] = g
    print("consecutive same (app,site) pairs with trigram Jaccard >= t: "
          + ", ".join(f"{t}: {near[t]}" for t in (0.5, 0.7, 0.8)) + f"  (of {len(user)})")
    if rows and "started_at" in rows[0]:
        span_h = (max(r["ended_at"] for r in rows) - min(r["started_at"] for r in rows)) / 3600
        text_b = sum(len((r.get("screen_text") or "").encode()) for r in rows)
        cap_b = sum(len((r.get("caption") or "").encode()) for r in rows)
        fr = [r["frame_path"] for r in rows if r.get("frame_path") and Path(r["frame_path"]).exists()]
        fr_b = sum(Path(p).stat().st_size for p in fr)
        print(f"span {span_h:.2f} h; rows/h {len(rows) / span_h:.0f}; screen_text {text_b / 1e6 / span_h:.2f} MB/h; "
              f"captions {cap_b / 1e3 / span_h:.0f} KB/h; frames {len(fr)} files {fr_b / 1e6 / span_h:.1f} MB/h "
              f"(avg {fr_b / max(1, len(fr)) / 1e3:.0f} KB)")
        srcs = collections.Counter(r["source"] for r in rows)
        kinds = collections.Counter(r["event_kind"] for r in rows)
        print(f"source={dict(srcs)} event_kind={dict(kinds)}")


if __name__ == "__main__":
    main()
