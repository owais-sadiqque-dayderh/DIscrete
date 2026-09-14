"""Online episode segmentation (planS2 §3.2).

Boundaries: session change / idle; a sustained (app, site) switch, placed
retroactively at the switch point; two consecutive semantic-shift
observations; the episode length cap. A brief alt-tab that returns to the
episode's context before `EPISODE_SWITCH_MIN_S` is absorbed.

Tracker state lives in memory. On restart, orphaned open episodes are made
ready by `Store.recover_open_episodes` (planS2 Rev 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, NotConfigured, embed
from ..store import Store
from ..timeutil import now

log = logging.getLogger("discrete.process.episodes")


@dataclass
class _Pending:
    """Observations since the context left the episode's (app, site)."""

    pair: tuple[str, str]
    since: float
    obs: list[tuple[int, object]] = field(default_factory=list)  # (id, row)


@dataclass
class EpisodeState:
    episode_id: int
    session_id: str
    started_at: float
    last_obs_at: float
    pair: tuple[str, str]
    n: int = 0
    centroid: np.ndarray | None = None
    switch: _Pending | None = None
    low_sim: list[tuple[int, object, np.ndarray]] = field(default_factory=list)


def pair_of(row) -> tuple[str, str]:
    return (row["app"] or "", row["site"] or "")


class EpisodeTracker:
    def __init__(self, store: Store, *, use_embeddings: bool = True):
        self.store = store
        self.state: EpisodeState | None = None
        self.use_embeddings = use_embeddings

    async def on_observation(self, obs_id: int, row, *, session_id: str) -> list[int]:
        """Assign an observation; returns episode ids that were closed (now ready)."""
        closed: list[int] = []
        if self.state is not None and self.state.session_id != session_id:
            closed += await self.close_current()
        if self.state is None:
            await self._open(obs_id, row, session_id)
            return closed

        st = self.state
        pair = pair_of(row)
        t = row["started_at"]
        vec = await self._embed(row)

        if pair == st.pair:
            st.switch = None  # a brief excursion returned: absorbed
        elif st.switch is not None and st.switch.pair == pair:
            st.switch.obs.append((obs_id, row))
            if t - st.switch.since >= T.EPISODE_SWITCH_MIN_S:
                moved = st.switch.obs
                closed += await self._split(moved[:-1], obs_id, row, session_id)
                return closed
        else:
            st.switch = _Pending(pair=pair, since=t, obs=[(obs_id, row)])

        # Semantic shift: two consecutive low-similarity observations (same context).
        if vec is not None and pair == st.pair:
            sim = float(st.centroid @ vec) if st.centroid is not None else 1.0
            if sim < T.EPISODE_SHIFT_SIM:
                st.low_sim.append((obs_id, row, vec))
                if len(st.low_sim) >= 2:
                    moved = [(i, r) for i, r, _ in st.low_sim[:-1]]
                    closed += await self._split(moved, obs_id, row, session_id)
                    return closed
            else:
                st.low_sim.clear()

        await self.store.assign_episode([obs_id], st.episode_id)
        st.n += 1
        st.last_obs_at = max(st.last_obs_at, row["ended_at"])
        if vec is not None and pair == st.pair and not st.low_sim:
            st.centroid = _update(st.centroid, vec, st.n)
        if st.last_obs_at - st.started_at >= T.EPISODE_MAX_S:
            closed += await self.close_current()
        return closed

    async def _open(self, obs_id: int, row, session_id: str) -> None:
        eid = await self.store.open_episode(session_id, row["started_at"])
        await self.store.assign_episode([obs_id], eid)
        vec = await self._embed(row)
        self.state = EpisodeState(
            episode_id=eid, session_id=session_id, started_at=row["started_at"],
            last_obs_at=row["ended_at"], pair=pair_of(row), n=1, centroid=vec,
        )

    async def _split(self, moved: list[tuple[int, object]], obs_id: int, row, session_id: str) -> list[int]:
        """Boundary at the switch point: `moved` + this observation start the next
        episode. They are reassigned before the old episode's interval is computed."""
        old = self.state.episode_id
        first_id, first_row = moved[0] if moved else (obs_id, row)
        await self._open(first_id, first_row, session_id)
        rest = [i for i, _ in moved[1:]] + ([obs_id] if moved else [])
        if rest:
            await self.store.assign_episode(rest, self.state.episode_id)
            self.state.n += len(rest)
            self.state.last_obs_at = max(self.state.last_obs_at, row["ended_at"])
            self.state.pair = pair_of(row)
        await self.store.mark_episode_ready(old)
        return [old]

    async def _embed(self, row) -> np.ndarray | None:
        if not self.use_embeddings:
            return None
        text = f"{row['caption']}\n{tk.truncate(row['screen_text'], 1000)}".strip()
        if not text:
            return None
        try:
            v = np.asarray((await embed([text]))[0], dtype=np.float32)
        except (NotConfigured, *EXPECTED_ERRORS):
            return None
        return v / max(float(np.linalg.norm(v)), 1e-9)

    async def close_current(self) -> list[int]:
        if not self.state:
            return []
        eid = self.state.episode_id
        self.state = None
        await self.store.mark_episode_ready(eid)
        return [eid]

    async def close_idle(self) -> list[int]:
        if self.state and now() - self.state.last_obs_at >= T.EPISODE_IDLE_CLOSE_S:
            return await self.close_current()
        return []


def _update(centroid: np.ndarray | None, vec: np.ndarray, n: int) -> np.ndarray:
    if centroid is None:
        return vec
    c = centroid * (n - 1) / n + vec / n
    return c / max(float(np.linalg.norm(c)), 1e-9)
