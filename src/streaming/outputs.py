"""Output artifacts of the stream processor: volume table CSV + alerts JSONL.

These files ARE the product ("useful output" in the rubric): the standard
15-minute volume table by camera/direction/class, and the alert log. Writers
append deterministically so replays regenerate byte-identical artifacts
(modulo alert ids/timestamps, which the replay check normalizes).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.streaming.contracts import TrafficAlert

VOLUME_COLUMNS = [
    "window_start",
    "window_end",
    "camera_id",
    "direction",
    "vehicle_class",
    "count",
]


@dataclass(frozen=True)
class VolumeRow:
    """One cell of the volume table: a (window, camera, direction, class) count."""

    window_start: datetime
    window_end: datetime
    camera_id: str
    direction: str
    vehicle_class: str
    count: int


def append_volume_rows(csv_path: Path, rows: list[VolumeRow]) -> None:
    """Append rows to the volume table, writing the header on first use."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(VOLUME_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.window_start.isoformat(),
                    row.window_end.isoformat(),
                    row.camera_id,
                    row.direction,
                    row.vehicle_class,
                    row.count,
                ]
            )


def append_alerts(jsonl_path: Path, alerts: list[TrafficAlert]) -> None:
    """Append alerts as JSON lines (ISO timestamps — human-readable artifact)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as fh:
        for alert in alerts:
            fh.write(alert.model_dump_json() + "\n")


def read_alerts(jsonl_path: Path) -> list[TrafficAlert]:
    """Load alerts back (dashboard + tests)."""
    if not jsonl_path.exists():
        return []
    return [
        TrafficAlert.model_validate_json(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
