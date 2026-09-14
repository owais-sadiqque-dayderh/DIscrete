"""Online episode segmentation (planS2 §3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import textkit as tk
from ..config import T
from ..llm import EXPECTED_ERRORS, embed
from ..store import Store
from ..timeutil import now

log = logging.getLogger("discrete.process.episodes")


@dataclass
class EpisodeState:
    episode_id: int
    session_id: str
    started_at: float
    last_obs_at: float
    apps: set[str] = field(default_factory=set)
    sites: set[str] = field(default_factory=set)
    obs_ids: list[int] = field(default_factory=list)
    switch_at: float | None = None
    switch_pair: tuple[str, str] | None = None
    low_sim_streak: int = 0
    centroid: list[float] | None = None


class EpisodeTracker:
    def __init__(self, store: Store):
        self.store = store
        self.state: EpisodeState | None = None

    async def on_observation(self, obs_id: int, obs_row, *, session_id: str, idle_transition: bool = False) -> list[int]:
        """Assign obs to episode; return episode ids that closed and need drafting."""
        closed: list[int] = []
        if idle_transition or self._session_changed(session_id):
            closed.extend(await self._close_current())
            self.state = None

        if self.state is None:
            self.state = EpisodeState(
                episode_id=await self.store.open_episode(session_id, obs_row["started_at"]),
                session_id=session_id,
                started_at=obs_row["started_at"],
                last_obs_at=obs_row["started_at"],
            )

        st = self.state
        st.obs_ids.append(obs_id)
        st.last_obs_at = obs_row["ended_at"]
        if obs_row["app"]:
            st.apps.add(obs_row["app"])
        if obs_row["site"]:
            st.sites.add(obs_row["site"])
        await self.store.assign_episode([obs_id], st.episode_id)

        if await self._should_close(st, obs_row):
            closed.extend(await self._close_current())
            self.state = None
        else:
            await self._update_centroid(st, obs_row)
        return closed

    def _session_changed(self, session_id: str) -> bool:
        return self.state is not None and self.state.session_id != session_id

    async def _should_close(self, st: EpisodeState, obs) -> bool:
        if st.last_obs_at - st.started_at >= T.EPISODE_MAX_S:
            return True
        pair = (obs["app"] or "", obs["site"] or "")
        if st.switch_pair and st.switch_pair != pair:
            if st.switch_at and (obs["started_at"] - st.switch_at) >= T.EPISODE_SWITCH_MIN_S:
                return True
        else:
            st.switch_pair = pair
            st.switch_at = obs["started_at"]

        sim = await self._sim_to_centroid(st, obs)
        if sim is not None and sim < T.EPISODE_SHIFT_SIM:
            st.low_sim_streak += 1
        else:
            st.low_sim_streak = 0
        if st.low_sim_streak >= 2:
            return True
        return False

    async def _sim_to_centroid(self, st: EpisodeState, obs) -> float | None:
        text = f"{obs['caption']}\n{tk.truncate(obs['screen_text'], 1000)}"
        try:
            vec = (await embed([text]))[0]
        except EXPECTED_ERRORS:
            return None
        if st.centroid is None:
            st.centroid = vec
            return 1.0
        sim = tk.cosine(st.centroid, vec)
        n = len(st.obs_ids)
        st.centroid = [(a * (n - 1) + b) / n for a, b in zip(st.centroid, vec)]
        return sim

    async def _update_centroid(self, st: EpisodeState, obs) -> None:
        await self._sim_to_centroid(st, obs)

    async def _close_current(self) -> list[int]:
        if not self.state:
            return []
        eid = self.state.episode_id
        await self.store.mark_episode_ready(eid)
        self.state = None
        return [eid]

    async def close_idle(self) -> list[int]:
        if not self.state:
            return []
        if now() - self.state.last_obs_at >= T.EPISODE_IDLE_CLOSE_S:
            return await self._close_current()
        return []
