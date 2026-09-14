"""Shared session boundaries across capture components."""

from __future__ import annotations

from uuid import uuid4


class SessionTracker:
    def __init__(self) -> None:
        self._id = str(uuid4())

    @property
    def session_id(self) -> str:
        return self._id

    def new_session(self) -> str:
        self._id = str(uuid4())
        return self._id
