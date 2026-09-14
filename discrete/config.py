"""Configuration: OpenAI-compatible endpoint cascade and every tunable in planS1-3.

All model calls go to APIs (plans Rev 2: no locally hosted models). Each
component may point at its own endpoint; unset components fall back to
CORE_LM_*. Blank env values count as unset. Tunables are read from
`DISCRETE_<NAME>` env vars, with the plan defaults below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def first(*names: str) -> str | None:
    for n in names:
        v = env(n)
        if v:
            return v
    return None


def _typed(name: str, default):
    raw = env(f"DISCRETE_{name}")
    if raw is None:
        return default
    try:
        if isinstance(default, bool):
            return raw.lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (list, tuple)):
            return [p.strip() for p in raw.split(",") if p.strip()]
    except ValueError:
        return default
    return raw


@dataclass(frozen=True)
class Endpoint:
    name: str
    base_url: str | None
    api_key: str | None
    model: str | None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def describe(self) -> str:
        if not self.configured:
            return f"{self.name}: (not configured)"
        host = (self.base_url or "https://api.openai.com/v1").split("//")[-1].split("/")[0]
        return f"{self.name}: {self.model} @ {host}"


_TAIL_BASE = ("CORE_LM_API_BASE",)
_TAIL_KEY = ("CORE_LM_API_KEY", "OPENAI_API_KEY")
_TAIL_MODEL = ("CORE_LM_MODEL_NAME",)


def _ep(name: str, *prefixes: str, model_fallback: bool = True) -> Endpoint:
    bases = [f"{p}_API_BASE" for p in prefixes] + list(_TAIL_BASE)
    keys = [f"{p}_API_KEY" for p in prefixes] + list(_TAIL_KEY)
    models = [f"{p}_MODEL_NAME" for p in prefixes] + (list(_TAIL_MODEL) if model_fallback else [])
    return Endpoint(name, first(*bases), first(*keys), first(*models))


def caption_endpoint() -> Endpoint:
    """Vision model for Stage 1 enrichment (planS1 §6.2)."""
    return _ep("caption", "CAPTION_LM")


def insight_endpoint() -> Endpoint:
    """Episode drafting, relation classification, entity adjudication (planS2)."""
    return _ep("insight", "INSIGHT_LM")


def consolidate_endpoint() -> Endpoint:
    """Offline consolidation (planS2 §7); may be a stronger model."""
    return _ep("consolidate", "CONSOLIDATE_LM", "INSIGHT_LM")


def utility_endpoint() -> Endpoint:
    """Query planning, listwise rerank fallback (planS3)."""
    return _ep("utility", "UTILITY_LM")


def embedding_endpoint() -> Endpoint:
    """Embeddings. Never falls back to a chat model."""
    return _ep("embedding", "EMBED_LM", model_fallback=False)


def rerank_endpoint() -> Endpoint:
    """OpenRouter-style /rerank (planS3 §6.1). Never falls back to a chat model."""
    return _ep("rerank", "RERANK_LM", model_fallback=False)


def all_endpoints() -> list[Endpoint]:
    return [
        caption_endpoint(),
        insight_endpoint(),
        consolidate_endpoint(),
        utility_endpoint(),
        embedding_endpoint(),
        rerank_endpoint(),
    ]


USER_NAME = env("USER_NAME", "the user") or "the user"


def data_dir() -> Path:
    p = Path(os.path.expanduser(env("DISCRETE_DATA_DIR", "~/.cache/discrete") or "~/.cache/discrete"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return Path(os.path.expanduser(env("DISCRETE_DB") or str(data_dir() / "discrete.db")))


def frames_dir() -> Path:
    p = data_dir() / "frames"
    p.mkdir(parents=True, exist_ok=True)
    return p


def golden_dir() -> Path:
    return ROOT / "golden"


class T:
    """Every tunable from planS1-3, with the plan's defaults."""

    # ---- Stage 1 (planS1 §11)
    MOUSE_DEBOUNCE_S = _typed("MOUSE_DEBOUNCE_S", 2.0)
    KEY_IDLE_DEBOUNCE_S = _typed("KEY_IDLE_DEBOUNCE_S", 6.0)
    KEYBOARD_CAPTURE = _typed("KEYBOARD_CAPTURE", True)
    SESSION_GAP_S = _typed("SESSION_GAP_S", 300.0)
    IDLE_THRESHOLD_S = _typed("IDLE_THRESHOLD_S", 300.0)
    CAPTURE_MAX_RETRIES = _typed("CAPTURE_MAX_RETRIES", 3)
    CAPTURE_RETRY_BASE_S = _typed("CAPTURE_RETRY_BASE_S", 4.0)
    FRAME_PHASH_MAX_DIST = _typed("FRAME_PHASH_MAX_DIST", 6)
    AX_WALK_MAX_NODES = _typed("AX_WALK_MAX_NODES", 4000)
    AX_WALK_TIMEOUT_MS = _typed("AX_WALK_TIMEOUT_MS", 250)
    AX_TEXT_MIN_CHARS = _typed("AX_TEXT_MIN_CHARS", 80)
    SCREEN_TEXT_MAX_CHARS = _typed("SCREEN_TEXT_MAX_CHARS", 16000)
    HEARTBEAT_TICK_S = _typed("HEARTBEAT_TICK_S", 60.0)
    HEARTBEAT_TEXT_CHANGE = _typed("HEARTBEAT_TEXT_CHANGE", 0.5)
    HEARTBEAT_MIN_GAP_S = _typed("HEARTBEAT_MIN_GAP_S", 120.0)
    HEARTBEAT_MAX_GAP_S = _typed("HEARTBEAT_MAX_GAP_S", 600.0)
    FRAME_MAX_WIDTH = _typed("FRAME_MAX_WIDTH", 1600)
    FRAME_JPEG_QUALITY = _typed("FRAME_JPEG_QUALITY", 65)
    POLL_S = _typed("POLL_S", 0.5)
    SELF_APPS = [s.lower() for s in _typed("SELF_APPS", [])]
    CANVAS_APPS = [s.lower() for s in _typed(
        "CANVAS_APPS",
        ["figma", "illustrator", "photoshop", "sketch", "screen sharing", "remote desktop", "affinity"],
    )]
    LLM_TIMEOUT_S = _typed("LLM_TIMEOUT_S", 60.0)

    # ---- Stage 2 (planS2 §13)
    DEDUP_JACCARD_THRESHOLD = _typed("DEDUP_JACCARD_THRESHOLD", 0.80)
    DEDUP_MINHASH_PERMUTATIONS = _typed("DEDUP_MINHASH_PERMUTATIONS", 128)
    DEDUP_SHORT_SHINGLES = _typed("DEDUP_SHORT_SHINGLES", 30)
    EPISODE_SWITCH_MIN_S = _typed("EPISODE_SWITCH_MIN_S", 90.0)
    EPISODE_SHIFT_SIM = _typed("EPISODE_SHIFT_SIM", 0.55)
    EPISODE_MAX_S = _typed("EPISODE_MAX_S", 1800.0)
    EPISODE_IDLE_CLOSE_S = _typed("EPISODE_IDLE_CLOSE_S", 300.0)
    CLAIM_MAX_CHARS = _typed("CLAIM_MAX_CHARS", 240)
    DRAFT_TEXT_CHARS_PER_OBS = _typed("DRAFT_TEXT_CHARS_PER_OBS", 1200)
    DRAFT_MAX_EVENTS = _typed("DRAFT_MAX_EVENTS", 5)
    DRAFT_MAX_DURABLE = _typed("DRAFT_MAX_DURABLE", 5)
    DRAFT_MAX_OBS = _typed("DRAFT_MAX_OBS", 40)
    EVIDENCE_MIN_OVERLAP = _typed("EVIDENCE_MIN_OVERLAP", 0.2)
    CANDIDATE_LIMIT = _typed("CANDIDATE_LIMIT", 8)
    RRF_K = _typed("RRF_K", 60.0)
    CONSOLIDATION_INTERVAL_S = _typed("CONSOLIDATION_INTERVAL_S", 3600.0)
    CONSOLIDATE_SIM = _typed("CONSOLIDATE_SIM", 0.82)
    CONSOLIDATE_MAX_CLUSTER = _typed("CONSOLIDATE_MAX_CLUSTER", 8)
    LINEAGE_MAX_VERSIONS = _typed("LINEAGE_MAX_VERSIONS", 12)
    INDUCTION_WINDOW_DAYS = _typed("INDUCTION_WINDOW_DAYS", 7.0)
    UNSUPPORTED_TTL_DAYS = _typed("UNSUPPORTED_TTL_DAYS", 14.0)
    CONFIDENCE_SCALE = _typed("CONFIDENCE_SCALE", 3.0)
    FACT_HALF_LIFE_DAYS = _typed("FACT_HALF_LIFE_DAYS", 180.0)
    PREF_HALF_LIFE_DAYS = _typed("PREF_HALF_LIFE_DAYS", 90.0)
    EVIDENCE_MAX_SUPPORTS = _typed("EVIDENCE_MAX_SUPPORTS", 12)
    ENTITY_MIN_EPISODES = _typed("ENTITY_MIN_EPISODES", 2)
    FUZZY_AUTO = _typed("FUZZY_AUTO", 96.0)
    FUZZY_ADJUDICATE = _typed("FUZZY_ADJUDICATE", 88.0)
    ENTITY_EMBED_SIM = _typed("ENTITY_EMBED_SIM", 0.80)
    ENTITY_PAGES_PER_RUN = _typed("ENTITY_PAGES_PER_RUN", 25)
    FRAME_TTL_DAYS = _typed("FRAME_TTL_DAYS", 3.0)
    SCREEN_TEXT_TTL_DAYS = _typed("SCREEN_TEXT_TTL_DAYS", 90.0)
    OUTBOX_MAX_AGE_S = _typed("OUTBOX_MAX_AGE_S", 600.0)
    EMBED_BATCH = _typed("EMBED_BATCH", 64)
    EMBED_TEXT_CHARS = _typed("EMBED_TEXT_CHARS", 2000)

    # ---- Stage 3 (planS3 §11)
    DENSE_HIT_THRESHOLD = _typed("DENSE_HIT_THRESHOLD", 0.5)
    CANDIDATE_POOL = _typed("CANDIDATE_POOL", 60)
    GRAPH_EXPAND = _typed("GRAPH_EXPAND", False)
    EXHAUSTIVE_MAX_SPAN_S = _typed("EXHAUSTIVE_MAX_SPAN_S", 86400.0)
    ECHO_THRESHOLD = _typed("ECHO_THRESHOLD", 0.55)
    COVERAGE_FLOOR = _typed("COVERAGE_FLOOR", 0.12)
    COVERAGE_STRONG = _typed("COVERAGE_STRONG", 0.55)
    COLD_START_ROWS = _typed("COLD_START_ROWS", 2000)
    RECENCY_HALF_LIFE_DAYS = _typed("RECENCY_HALF_LIFE_DAYS", 30.0)
    MMR_LAMBDA = _typed("MMR_LAMBDA", 0.7)
    ENTITY_PROP_OVERLAP_THRESHOLD = _typed("ENTITY_PROP_OVERLAP_THRESHOLD", 0.6)
    RERANK_POOL = _typed("RERANK_POOL", 50)
    RERANK_MAX_TOKENS = _typed("RERANK_MAX_TOKENS", 384)
    RERANK_STRONG = _typed("RERANK_STRONG", 0.4)
    LLM_RERANK = _typed("LLM_RERANK", False)
    LLM_RERANK_POOL = _typed("LLM_RERANK_POOL", 24)
    QUERY_LLM_PLAN = _typed("QUERY_LLM_PLAN", True)
    TOP_K = _typed("TOP_K", 8)
    EVAL_K = _typed("EVAL_K", 8)
    EVAL_REGRESSION_ALPHA = _typed("EVAL_REGRESSION_ALPHA", 0.05)
    EVAL_MIN_CASES_FOR_SIG = _typed("EVAL_MIN_CASES_FOR_SIG", 100)
    EVAL_TOLERANCE = _typed("EVAL_TOLERANCE", 0.05)
    STORE_CARD_TTL_S = _typed("STORE_CARD_TTL_S", 120.0)
