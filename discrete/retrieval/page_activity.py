"""Guarded page aggregates and deterministic answers, before any model call."""

from __future__ import annotations

from ..config import T
from ..models import Hit, RetrievalResult
from ..store import Store
from ..timeutil import now


async def page_activity(store: Store, result: RetrievalResult) -> None:
    plan = result.plan
    start, end = plan.interval or (0, now() + 1)
    op = plan.page_operation
    kinds = (("doc", "file") if op == "edited" else
             ("article", "repo", "doc", "file", "other") if op == "read_most" else ())
    # A duration asks for the total, not merely the top N pages.
    rows = await store.list_pages(start=start, end=end, kind="repo" if op == "returns" else "",
                                  limit=2**31 - 1 if op == "duration" else T.PAGE_QUERY_TOP,
                                  order="visits" if op == "returns" else "dwell", search=plan.page_search,
                                  edited=op == "edited", kinds=kinds)
    if op == "returns":
        rows = [p for p in rows if p["visits"] > 1]
    result.exhaustive = True
    for p in rows:
        result.hits.append(Hit("page", p["id"], p["title"],
                               f"{p['canonical_url']} · {p['visits']} visits · {p['dwell_s']:.0f}s foreground · "
                               f"{p['active_dwell_s']:.0f}s active", strong=True,
                               occurred_at=p["first_seen"], occurred_to=p["last_seen"], url=p["canonical_url"]))
    if not rows:
        result.page_answer = "No matching recorded pages in this period. Older captures may need `discrete pages --backfill`."
        return
    if op == "duration":
        seconds = sum(p["dwell_s"] for p in rows)
        active = sum(p["active_dwell_s"] for p in rows)
        result.page_answer = (f"Estimated foreground time on {plan.page_search}: {seconds:.0f}s "
                              f"({seconds / 60:.1f} min), including {active:.0f}s with input.")
        return
    header = {"read_most": "Pages with most estimated foreground time (reading is not confirmed):",
              "returns": "Repositories revisited:",
              "edited": "Documents with typing (edit signal; saving is not verified):"}[op]
    result.page_answer = header + "\n" + "\n".join(
        f"- {p['title']} — {p['canonical_url']} · {p['visits']} visits · "
        f"{p['dwell_s'] / 60:.1f} min ({p['active_dwell_s'] / 60:.1f} active)" for p in rows)
