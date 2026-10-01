"""Enumeration: "list all the pages I opened …" (planS3 §17).

Bypasses TOP_K and MMR and returns every distinct page, time-ordered. Page
identity, listing-ness and "opened from" come from structured observation
fields (see pages.py), never from caption wording.
"""

from __future__ import annotations

import re

from .. import textkit as tk
from ..config import T
from ..llm import try_embed
from ..models import Hit, QueryPlan, RetrievalResult
from ..store import Store
from .pages import canon_url, is_generic_title, names_overlap, page_identity, page_label

_OPENED_QUERY = re.compile(r"\b(?:open|opened|read|visit|visited|click|clicked|saw|seen)\b", re.I)


def wants_opened_items(plan: QueryPlan) -> bool:
    return bool(_OPENED_QUERY.search(plan.raw))


def _in_interval(row, interval: tuple[float, float] | None) -> bool:
    return not interval or (row["started_at"] < interval[1] and row["ended_at"] >= interval[0])


async def enumerate_pages(store: Store, plan: QueryPlan, result: RetrievalResult, qvec) -> None:
    """Matching alone cannot answer "what did I open from X": the pages opened from a
    listing rarely mention it. So the list is the *trail*: every page that matched
    (status "matched"), pages the vision call saw being reached from a matched source
    (`navigated_from`), and the pages visited in the same app within
    `ENUMERATE_TRAIL_S` after a match (status "trail") (insights.md §7.34)."""
    result.exhaustive = True
    ids: list[int] = []
    subject = " ".join(plan.queries[:8]).strip()
    if subject:
        for q in plan.queries[:8]:
            ids += await store.fts_ids("observation", q, limit=T.ENUMERATE_POOL)
            ids += await store.observation_ids_navigated_from(q, limit=T.ENUMERATE_POOL)
        # Embed the subject, not the raw request: its scaffolding pulls in unrelated rows.
        qvec = await try_embed(subject) if subject != plan.raw else qvec
        if qvec is not None:
            ids += [i for i, s in await store.dense_search("observation", qvec, limit=T.ENUMERATE_POOL)
                    if s >= T.DENSE_HIT_THRESHOLD]
    elif plan.interval:
        # "List the pages I opened today": no subject, so everything in the interval matches.
        ids = await store.observation_ids_started_between(*plan.interval)
    matched = {r["id"]: r for r in await store.live_rows("observation", list(dict.fromkeys(ids)))
               if _in_interval(r, plan.interval)}
    if not matched:
        return
    # A named source ("hacker news", "figma") is matched by page identity: window title,
    # page title, site, URL. Screen text mentions other pages too, so content matches
    # alone marked every page as the source (insights.md §7.36). A topic with no identity
    # match anywhere ("articles about rust") keeps content matches. Per query string, not
    # their union: the planner adds sibling queries, whose union is never in one title.
    wants = [w for w in (set(tk.content_tokens(q)) for q in plan.queries[:8]) if w]
    linked: dict[int, object] = {}
    if wants:
        def cov(text: str) -> float:
            toks = set(tk.tokens(text))
            return max(len(w & toks) / len(w) for w in wants)

        by_ident = {i for i, r in matched.items()
                    if cov(f"{r['window_title']} {r['page_title']} {r['site']} {r['url']}") >= 0.5}
        from_src = {i for i, r in matched.items() if i not in by_ident and cov(r["navigated_from"]) >= 0.5}
        if by_ident or from_src:
            linked = {i: matched[i] for i in from_src}
            matched = {i: r for i, r in matched.items() if i in by_ident}
    apps = {r["app"] for r in matched.values()}

    spans: list[list[float]] = []
    for t in sorted(r["started_at"] for r in matched.values()):
        if spans and t <= spans[-1][1]:
            spans[-1][1] = t + T.ENUMERATE_TRAIL_S
        else:
            spans.append([t, t + T.ENUMERATE_TRAIL_S])
    rows = []
    for a, b in spans:
        b = min(b, plan.interval[1]) if plan.interval else b
        rows += [r for r in await store.observations_started_within(a, b) if r["id"] in matched or r["app"] in apps]
    have = {r["id"] for r in rows}
    rows += [r for i, r in linked.items() if i not in have]
    rows.sort(key=lambda r: r["started_at"])

    visits: list[dict] = []
    visit_of: dict[str, dict] = {}
    for r in rows:
        url = canon_url(r["url"] or "")
        title = r["window_title"] or r["app"]
        ident = page_identity(title, page_title=r["page_title"], url=url, caption=r["caption"] or "")
        v = visit_of.get(ident)
        if v is None:
            v = {"id": r["id"], "title": page_label(title, page_title=r["page_title"], url=url, site=r["site"] or ""),
                 "site": r["site"], "app": r["app"], "url": url, "caption": r["caption"], "first": r["started_at"],
                 "last": r["ended_at"], "matched": False, "is_listing": False, "navigated_from": ""}
            visits.append(v)
            visit_of[ident] = v
        v["last"] = max(v["last"], r["ended_at"])
        v["matched"] = v["matched"] or r["id"] in matched
        v["is_listing"] = v["is_listing"] or bool(r["is_listing"])
        v["navigated_from"] = v["navigated_from"] or r["navigated_from"]
        if r["caption"] and not v["caption"]:
            v["caption"] = r["caption"]

    for v in visits[: T.ENUMERATE_MAX]:
        body = " · ".join(x for x in (v["site"] or v["app"], v["url"]) if x)
        if v["caption"]:
            body += f"\n{tk.truncate(v['caption'], 160)}"
        result.hits.append(Hit("observation", v["id"], v["title"][:140], body, strong=True, untrusted=True,
                               status="matched" if v["matched"] else "trail", url=v["url"],
                               occurred_at=v["first"], occurred_to=v["last"], is_listing=v["is_listing"],
                               navigated_from=v["navigated_from"]))


def shape_enumerate_items(hits: list[Hit], plan: QueryPlan) -> list[Hit]:
    """The lines that answer a 'list what I opened/read' question.

    Raw enumerate hits include listing pages and every same-app page in the trail
    window. For opened-item questions: listings are the source, never an item; a
    page the vision call saw being reached from some *other* page is not an item
    opened from this source; once any page has a recorded source, pages with none
    are only nearby in time; a window with a generic title and no page title
    names nothing."""
    labeled: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        key = tk.norm_key(h.title)
        if key not in seen:
            seen.add(key)
            labeled.append(h)
    if not labeled or not wants_opened_items(plan):
        return labeled
    sources = [h.title for h in labeled if h.status == "matched"]

    def item(h: Hit) -> bool:
        if h.is_listing or is_generic_title(h.title):
            return False
        if h.status == "trail" and h.navigated_from and sources:
            return any(names_overlap(h.navigated_from, s) for s in sources)
        return True

    items = [h for h in labeled if item(h)]
    # Provenance beats proximity: when the vision call recorded that some pages were
    # reached from the source, same-app pages merely open nearby in time are dropped.
    if any(h.status == "trail" and h.navigated_from for h in items):
        items = [h for h in items if h.status == "matched" or h.navigated_from]
    return items or labeled
