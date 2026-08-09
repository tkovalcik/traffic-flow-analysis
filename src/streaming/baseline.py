"""EWMA baselines over per-window vehicle counts.

Pure state + math, no Kafka: the stream processor (hand-written consumer/window
core) calls into this once per closed window. Baselines are tracked per
(camera_id, direction) so a quiet side street never dilutes a busy mainline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaselineStats:
    """EWMA state for one key. `windows_seen` gates alerting until warmed up."""

    ewma: float = 0.0
    windows_seen: int = 0


@dataclass
class EwmaBaseline:
    """Exponentially weighted moving average of window counts per key.

    alpha: weight of the newest window (0.3 → ~last 3 windows dominate).
    """

    alpha: float = 0.3
    _stats: dict[tuple[str, str], BaselineStats] = field(default_factory=dict)

    def observe(self, camera_id: str, direction: str, count: int) -> BaselineStats:
        """Fold one closed window's count into the baseline for its key.

        Returns the stats as they were BEFORE this observation — alert rules
        compare the new count against the history, not against itself.
        """
        key = (camera_id, direction)
        stats = self._stats.get(key, BaselineStats())
        before = BaselineStats(ewma=stats.ewma, windows_seen=stats.windows_seen)
        if stats.windows_seen == 0:
            updated = BaselineStats(ewma=float(count), windows_seen=1)
        else:
            updated = BaselineStats(
                ewma=self.alpha * count + (1 - self.alpha) * stats.ewma,
                windows_seen=stats.windows_seen + 1,
            )
        self._stats[key] = updated
        return before

    def stats_for(self, camera_id: str, direction: str) -> BaselineStats:
        return self._stats.get((camera_id, direction), BaselineStats())
