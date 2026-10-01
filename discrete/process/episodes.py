"""Online episode segmentation (planS2 §3.2).

Boundaries: session change / idle; a sustained (app, site) switch, placed
retroactively at the switch point; two consecutive semantic-shift
observations; the episode length cap. A brief alt-tab that returns to the
episode's context before `EPISODE_SWITCH_MIN_S` is absorbed.

Tracker state lives in memory. On restart, orphaned open episodes are made
ready by `Store.recover_open_episodes` (planS2 Rev 3).

The tracker never calls the embeddings API itself: the caller passes the
observation's stored vector (process/embed_worker.embed_row), so each
observation is embedded exactly once. Without a vector, only the (app, site)
and time rules apply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import T
from ..store import Store
from ..timeutil import now

log = logging.getLogger("discrete.process.episodes")


@dataclass
class _Pending:
    """Observations since the context left the episode's (app, site)."""

    pair: tuple[str, str]
    since: float
    obs: list[tuple[int, object, np.ndarray | None]] = field(default_factory=list)  # (id, row, vec)


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
    low_sim: list[tuple[int, object, np.ndarray | None]] = field(default_factory=list)


def pair_of(row) -> tuple[str, str]:
    return (row["app"] or "", row["site"] or "")


class EpisodeTracker:
    def __init__(self, store: Store):
        self.store = store
        self.state: EpisodeState | None = None

    async def on_observation(self, obs_id: int, row, *, session_id: str,
                             vec: np.ndarray | None = None) -> list[int]:
        """Assign an observation; returns episode ids that were closed (now ready).
        `vec` is the observation's unit embedding, if it has one."""
        closed: list[int] = []
        if self.state is not None and self.state.session_id != session_id:
            closed += await self.close_current()
        if self.state is None:
            await self._open(obs_id, row, session_id, vec)
            return closed

        st = self.state
        pair = pair_of(row)
        t = row["started_at"]

        if pair == st.pair:
            st.switch = None  # a brief excursion returned: absorbed
        elif st.switch is not None and st.switch.pair == pair:
            st.switch.obs.append((obs_id, row, vec))
            if t - st.switch.since >= T.EPISODE_SWITCH_MIN_S:
                moved = st.switch.obs
                closed += await self._split(moved[:-1], obs_id, row, session_id, vec)
                return closed
        else:
            st.switch = _Pending(pair=pair, since=t, obs=[(obs_id, row, vec)])

        # Semantic shift: two consecutive low-similarity observations (same context).
        if vec is not None and pair == st.pair:
            sim = float(st.centroid @ vec) if st.centroid is not None else 1.0
            if sim < T.EPISODE_SHIFT_SIM:
                st.low_sim.append((obs_id, row, vec))
                if len(st.low_sim) >= 2:
                    moved = st.low_sim[:-1]
                    closed += await self._split(moved, obs_id, row, session_id, vec)
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

    async def _open(self, obs_id: int, row, session_id: str, vec: np.ndarray | None) -> None:
        eid = await self.store.open_episode(session_id, row["started_at"])
        await self.store.assign_episode([obs_id], eid)
        self.state = EpisodeState(
            episode_id=eid, session_id=session_id, started_at=row["started_at"],
            last_obs_at=row["ended_at"], pair=pair_of(row), n=1, centroid=vec,
        )

    async def _split(self, moved: list[tuple[int, object, np.ndarray | None]], obs_id: int, row,
                     session_id: str, vec: np.ndarray | None) -> list[int]:
        """Boundary at the switch point: `moved` + this observation start the next
        episode. They are reassigned before the old episode's interval is computed."""
        assert self.state is not None
        old = self.state.episode_id
        first_id, first_row, first_vec = moved[0] if moved else (obs_id, row, vec)
        await self._open(first_id, first_row, session_id, first_vec)
        rest = [i for i, _, _ in moved[1:]] + ([obs_id] if moved else [])
        if rest:
            await self.store.assign_episode(rest, self.state.episode_id)
            self.state.n += len(rest)
            self.state.last_obs_at = max(self.state.last_obs_at, row["ended_at"])
            self.state.pair = pair_of(row)
        await self.store.mark_episode_ready(old)
        return [old]

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
