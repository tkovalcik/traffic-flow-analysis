"""Event-time tumbling windows over vehicle events. Pure state + time, no Kafka.

The window core the consumer loop delegates to: bucket events onto a clock-aligned
grid, close a window once a per-camera watermark passes its end, and drop events
whose window already closed. Windows close on event time only — never wall clock —
so replayed output is identical at --speed 0 in CI and --speed 60 in the demo.

Watermarks are per camera so one silent camera cannot stall another's windows;
stream_time is the cross-camera clock the staleness rule compares against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from src.streaming.contracts import VehicleEvent
from src.streaming.outputs import VolumeRow

DEFAULT_WINDOW_SECONDS = 900
DEMO_WINDOW_SECONDS = 60
DEFAULT_LATENESS_SECONDS = 30.0

WindowKey = tuple[str, str, datetime]


@dataclass(frozen=True)
class ClosedWindow:
    """One finished window for a (camera, direction), counted by vehicle class."""

    camera_id: str
    direction: str
    window_start: datetime
    window_end: datetime
    counts: dict[str, int]

    @property
    def total(self) -> int:
        """Vehicles across all classes — what the baseline and alert rules see."""
        return sum(self.counts.values())


def to_volume_rows(window: ClosedWindow) -> list[VolumeRow]:
    """Expand a closed window into one volume-table row per vehicle class."""
    return [
        VolumeRow(
            window_start=window.window_start,
            window_end=window.window_end,
            camera_id=window.camera_id,
            direction=window.direction,
            vehicle_class=vehicle_class,
            count=window.counts[vehicle_class],
        )
        for vehicle_class in sorted(window.counts)
    ]


def window_bounds(ts: datetime, window_seconds: int) -> tuple[datetime, datetime]:
    """The clock-aligned window containing ts, as (start, end)."""
    epoch_s = int(ts.timestamp())
    start = datetime.fromtimestamp(epoch_s - epoch_s % window_seconds, tz=UTC)
    return start, start + timedelta(seconds=window_seconds)


@dataclass
class TumblingWindows:
    """Accumulates events into clock-aligned windows, closing them on watermarks."""

    window_seconds: int = DEFAULT_WINDOW_SECONDS
    lateness_seconds: float = DEFAULT_LATENESS_SECONDS
    late_dropped: dict[str, int] = field(default_factory=dict)
    _open: dict[WindowKey, dict[str, int]] = field(default_factory=dict)
    _last_event: dict[str, datetime] = field(default_factory=dict)
    _closed_through: dict[str, datetime] = field(default_factory=dict)

    def add(self, event: VehicleEvent) -> list[ClosedWindow]:
        """Count one event, returning any windows its watermark just closed.

        Closed windows come back ascending by (window_start, camera_id,
        direction) so replays regenerate an identical table; an event whose
        window already closed is dropped into late_dropped instead.
        """
        start, end = window_bounds(event.ts_event, self.window_seconds)
        closed_through = self._closed_through.get(event.camera_id)
        if closed_through is not None and end <= closed_through:
            self.late_dropped[event.camera_id] = self.late_dropped.get(event.camera_id, 0) + 1
            return []

        counts = self._open.setdefault((event.camera_id, event.direction.value, start), {})
        counts[event.vehicle_class.value] = counts.get(event.vehicle_class.value, 0) + 1

        last = self._last_event.get(event.camera_id)
        if last is None or event.ts_event > last:
            self._last_event[event.camera_id] = event.ts_event
        return self._close_due(event.camera_id)

    def _close_due(self, camera_id: str) -> list[ClosedWindow]:
        """Emit this camera's windows that now sit fully behind its watermark."""
        mark = self.watermark(camera_id)
        if mark is None:
            return []
        span = timedelta(seconds=self.window_seconds)
        due = [key for key in self._open if key[0] == camera_id and key[2] + span <= mark]
        # Empty windows never entered _open, so track the boundary itself —
        # otherwise a late event could reopen a stretch already reported as past.
        self._closed_through[camera_id] = window_bounds(mark, self.window_seconds)[0]
        return self._emit(due)

    def _emit(self, keys: list[WindowKey]) -> list[ClosedWindow]:
        """Pop the given open windows in the documented deterministic order."""
        closed = []
        for key in sorted(keys, key=lambda k: (k[2], k[0], k[1])):
            camera_id, direction, start = key
            closed.append(
                ClosedWindow(
                    camera_id=camera_id,
                    direction=direction,
                    window_start=start,
                    window_end=start + timedelta(seconds=self.window_seconds),
                    counts=self._open.pop(key),
                )
            )
        return closed

    def flush(self) -> list[ClosedWindow]:
        """Close every remaining open window at end of stream, same ordering.

        The trailing window has no later event to push a watermark past its end,
        so without this the tail of a capture is silently never reported.
        """
        return self._emit(list(self._open))

    @property
    def stream_time(self) -> datetime | None:
        """Highest ts_event seen on any camera — the staleness rule's `now`."""
        return max(self._last_event.values()) if self._last_event else None

    def watermark(self, camera_id: str) -> datetime | None:
        """One camera's watermark: its last ts_event minus the lateness bound."""
        last = self._last_event.get(camera_id)
        return None if last is None else last - timedelta(seconds=self.lateness_seconds)

    def last_event_at(self, camera_id: str) -> datetime | None:
        """One camera's highest seen ts_event, or None if it never produced."""
        return self._last_event.get(camera_id)

    def cameras(self) -> list[str]:
        """Every camera seen so far, for liveness checks over quiet cameras."""
        return sorted(self._last_event)
