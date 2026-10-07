"""Plain data shapes shared across layers. Imports nothing from the package but textkit.

Dependency direction: models → store → llm → capture/process/retrieval → bubble/app/cli.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import textkit as tk


@dataclass
class ObservationDraft:
    source: str = "screen"
    origin: str = "user"
    event_kind: str = "click"
    session_id: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    app: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    site: str = ""
    screen_text: str = ""
    text_source: str = ""
    caption: str = ""
    entities: list[str] = field(default_factory=list)
    keys: int = 0
    clicks: int = 0
    scrolls: int = 0
    clipboard: int = 0
    frame_phash: str = ""
    frame_path: str = ""
    # Page identity, returned as structured fields by the vision call (never parsed
    # out of caption prose): the page's own title, whether it is a listing/feed/index,
    # and the page it was reached from.
    page_title: str = ""
    is_listing: bool = False
    navigated_from: str = ""
    # "pending" when a vision caption will be added after the row is stored.
    enrich_state: str = ""
    # Stage 1 pHash tier-0 duplicate: bump the matching row, never insert.
    frame_duplicate: bool = False

    def dedup_key(self) -> str:
        return tk.stable_obs_key(self.app, self.window_title, self.url, self.screen_text)

    def world_text(self) -> str:
        """Text counted in term statistics (planS3 §5): world text only."""
        return world_text(self.window_title, self.site, self.caption, self.screen_text, self.page_title)


def world_text(window_title: str, site: str, caption: str, screen_text: str, page_title: str = "") -> str:
    return "\n".join(p for p in (window_title, page_title, site, caption, screen_text) if p)


@dataclass
class ClaimDraft:
    kind: str
    text: str
    context: str = ""
    reasoning: str = ""
    modality: str = "passive"
    episode_id: int | None = None
    occurred_from: float = 0.0
    occurred_to: float = 0.0
    entities: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # [{observation_id, quote}]


@dataclass
class ClaimWrite:
    """A prepared mutation; model calls and entity resolution precede persistence."""

    draft: ClaimDraft
    operation: str = "added"
    target_id: int | None = None
    refines_id: int | None = None
    contested_id: int | None = None
    vector: list[float] | None = None
    text_hash: str = ""
    mentions: list[int] = field(default_factory=list)


@dataclass
class QueryPlan:
    raw: str
    intent: str = "recall"
    queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    interval: tuple[float, float] | None = None
    entity_kind_filter: str = ""
    needs_store: bool = True


@dataclass
class Hit:
    ref_kind: str
    ref_id: int
    title: str
    body: str
    score: float = 0.0
    coverage: float = 0.0
    echo: float = 0.0
    dense_sim: float = 0.0
    rerank_score: float | None = None
    strong: bool = False
    untrusted: bool = False
    modality: str = ""
    confidence: float = 0.0
    status: str = ""
    occurred_at: float = 0.0
    occurred_to: float = 0.0
    claim_kind: str = ""
    url: str = ""
    is_listing: bool = False
    navigated_from: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"

    def in_interval(self, interval: tuple[float, float] | None) -> bool:
        """planS3 §4.1: episodes, observations and events outside the interval are dropped."""
        if not interval or self.ref_kind == "entity" or (self.ref_kind == "claim" and self.claim_kind != "event"):
            return True
        start, end = interval
        return self.occurred_at < end and (self.occurred_to or self.occurred_at) >= start


@dataclass
class RetrievalResult:
    query: str
    plan: QueryPlan
    hits: list[Hit] = field(default_factory=list)
    exhaustive: bool = False
    digest: str = ""

    def of(self, kind: str) -> list[Hit]:
        return [h for h in self.hits if h.ref_kind == kind]
