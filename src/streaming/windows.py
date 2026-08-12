"""Event-time tumbling windows over vehicle events. Pure state + time, no Kafka.

The window core the consumer loop (task 2.1) delegates to: bucket events onto a
clock-aligned grid, close a window once a per-camera watermark passes its end,
and drop events whose window already closed. Windows close on event time only —
never wall clock — so replayed output is identical at --speed 0 in CI and at
--speed 60 in the demo.

Signatures and data shapes are fixed by tests/test_windows.py; the state machine
itself is unimplemented on purpose (see CLAUDE.md: the team hand-writes it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.streaming.contracts import VehicleEvent
from src.streaming.outputs import VolumeRow

DEFAULT_WINDOW_SECONDS = 900
DEMO_WINDOW_SECONDS = 60
DEFAULT_LATENESS_SECONDS = 30.0


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
    raise NotImplementedError("task 2.1")


@dataclass
class TumblingWindows:
    """Accumulates events into clock-aligned windows, closing them on watermarks.

    Watermarks are tracked per camera (camera_id, so one silent camera cannot
    stall another's windows) and sit lateness_seconds behind that camera's
    highest seen ts_event.
    """

    window_seconds: int = DEFAULT_WINDOW_SECONDS
    lateness_seconds: float = DEFAULT_LATENESS_SECONDS
    late_dropped: dict[str, int] = field(default_factory=dict)

    def add(self, event: VehicleEvent) -> list[ClosedWindow]:
        """Count one event, returning any windows its watermark just closed.

        Windows come back ascending by (window_start, camera_id, direction) so
        the volume table is byte-identical across runs. An event whose window
        already closed is dropped and tallied in late_dropped instead.
        """
        raise NotImplementedError("task 2.1")

    def flush(self) -> list[ClosedWindow]:
        """Close every remaining open window at end of stream, same ordering.

        Without this the trailing window never closes: no further event exists
        to push the watermark past its end.
        """
        raise NotImplementedError("task 2.1")

    @property
    def stream_time(self) -> datetime | None:
        """Highest ts_event seen on any camera — the staleness rule's `now`."""
        raise NotImplementedError("task 2.1")

    def watermark(self, camera_id: str) -> datetime | None:
        """One camera's watermark: its last ts_event minus the lateness bound."""
        raise NotImplementedError("task 2.1")

    def last_event_at(self, camera_id: str) -> datetime | None:
        """One camera's highest seen ts_event, or None if it never produced."""
        raise NotImplementedError("task 2.1")
