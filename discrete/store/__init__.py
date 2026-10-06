"""Storage layer: every SQL statement in the package lives under this directory.

`Store` is the composition of one piece per concern; all pieces share the
connections, locks and write helpers in `base.StoreBase`.
"""

from __future__ import annotations

from .drafts import DraftConflict, DraftsMixin
from .entities import EntitiesMixin
from .episodes import EpisodesMixin
from .gaps import GapsMixin
from .health import HealthMixin
from .index import KINDS, IndexMixin
from .observations import ObservationsMixin
from .perf import PerfMixin


class Store(ObservationsMixin, EpisodesMixin, DraftsMixin, EntitiesMixin, IndexMixin, GapsMixin, HealthMixin,
            PerfMixin):
    pass


__all__ = ["KINDS", "DraftConflict", "Store"]
