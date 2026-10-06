"""Storage layer: every SQL statement in the package lives under this directory.

`Store` is the composition of one piece per concern; all pieces share the
connections, locks and write helpers in `base.StoreBase`.
"""

from __future__ import annotations

from .corrections import CorrectionsMixin
from .drafts import DraftConflict, DraftsMixin
from .entities import EntitiesMixin
from .episodes import EpisodesMixin
from .gaps import GapsMixin
from .health import HealthMixin
from .history import RevertConflict
from .index import KINDS, IndexMixin
from .links import LinksMixin
from .observations import ObservationsMixin
from .perf import PerfMixin


class Store(ObservationsMixin, EpisodesMixin, DraftsMixin, CorrectionsMixin, LinksMixin, EntitiesMixin, IndexMixin,
            GapsMixin, HealthMixin, PerfMixin):
    pass


__all__ = ["KINDS", "DraftConflict", "RevertConflict", "Store"]
