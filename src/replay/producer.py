"""Replay recorded crossing events into Kafka on their original event time.

Drives the deterministic reviewer demo: reads a recorded JSONL of vehicle events
and republishes them through the shared producer, preserving each event's
ts_event and rewriting ts_publish to the moment it reaches the broker. --speed
compresses the wall-clock gaps so a 15-minute capture replays in seconds, and
--late-fraction holds a share of events back past their window boundary so the
consumer's late-event path actually runs. --mirror-camera adds a second,
clearly synthetic camera that --drop-after can silence mid-stream, which is the
only way a single-camera recording can exercise the camera_stale rule.

Usage:
    uv run python -m src.replay.producer --speed 60
    uv run python -m src.replay.producer --late-fraction 0.05 --speed 120
    uv run python -m src.replay.producer --mirror-camera tva43_mirror --drop-after
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from src.streaming.contracts import VehicleEvent
from src.streaming.producer import DeliveryStats, EventProducer

DEFAULT_FILE = Path("data/sample/replay_tva43_15min.jsonl")
DEFAULT_SPEED = 60.0
DEFAULT_LATE_SECONDS = 120.0
DEFAULT_SEED = 0
DEFAULT_DROP_AFTER = 300.0


def load_events(path: Path, limit: int | None = None) -> list[VehicleEvent]:
    """Parse a recorded events JSONL into validated VehicleEvents."""
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        events.append(VehicleEvent.from_avro_dict(json.loads(line)))
        if limit is not None and len(events) >= limit:
            break
    return events


def emission_schedule(
    events: list[VehicleEvent],
    late_fraction: float = 0.0,
    late_seconds: float = DEFAULT_LATE_SECONDS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[float, VehicleEvent]]:
    """(seconds after the first event, event) pairs in the order they are sent.

    Recorded files are perfectly ordered by ts_event, so nothing is ever late on
    its own. late_fraction defers a seeded random share past its window, which
    reorders emission while leaving each event's ts_event untouched.
    """
    if not events:
        return []
    start = events[0].ts_event
    rng = random.Random(seed)
    schedule = []
    for event in events:
        offset = (event.ts_event - start).total_seconds()
        if late_fraction > 0 and rng.random() < late_fraction:
            offset += late_seconds
        schedule.append((offset, event))
    schedule.sort(key=lambda pair: pair[0])
    return schedule


def mirror_events(
    events: list[VehicleEvent],
    camera_id: str,
    drop_after_seconds: float | None = None,
) -> list[VehicleEvent]:
    """Copy the capture onto a second camera, optionally silenced part way in.

    A lone camera's stream_time is its own last event, so its silence gap is
    always zero and camera_stale can never fire. A mirror that stops early
    leaves the real camera advancing the clock, which is what the rule needs.
    """
    if not events:
        return []
    start = events[0].ts_event
    mirrored = []
    for event in events:
        if drop_after_seconds is not None:
            if (event.ts_event - start).total_seconds() > drop_after_seconds:
                continue
        mirrored.append(event.model_copy(update={"camera_id": camera_id, "event_id": str(uuid4())}))
    return mirrored


def replay(
    schedule: list[tuple[float, VehicleEvent]],
    producer: EventProducer,
    speed: float = DEFAULT_SPEED,
    sleeper: Callable[[float], None] = time.sleep,
) -> DeliveryStats:
    """Publish the schedule, event-time gaps divided by speed (0 = no waiting)."""
    previous = 0.0
    for offset, event in schedule:
        if speed > 0:
            delay = (offset - previous) / speed
            if delay > 0:
                sleeper(delay)
        previous = offset
        producer.publish(event.model_copy(update={"ts_publish": datetime.now(UTC)}))
    return producer.flush()


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--limit", type=int, help="Replay only the first N events")
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Event-time compression; 0 replays as fast as possible",
    )
    parser.add_argument(
        "--late-fraction", type=float, default=0.0, help="Share of events delayed past their window"
    )
    parser.add_argument("--late-seconds", type=float, default=DEFAULT_LATE_SECONDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Lateness draw seed")
    parser.add_argument(
        "--mirror-camera",
        help="Replay the capture a second time under this camera id (synthetic)",
    )
    parser.add_argument(
        "--drop-after",
        type=float,
        nargs="?",
        const=DEFAULT_DROP_AFTER,
        help="Silence the mirrored camera this many seconds in, so camera_stale fires",
    )
    args = parser.parse_args(argv)

    if args.drop_after is not None and not args.mirror_camera:
        raise SystemExit("--drop-after needs --mirror-camera; the real capture is never truncated")

    events = load_events(args.file, args.limit)
    if not events:
        raise SystemExit(f"no events in {args.file}")
    if args.mirror_camera:
        mirrored = mirror_events(events, args.mirror_camera, args.drop_after)
        silence = f", silent after {args.drop_after:.0f}s" if args.drop_after is not None else ""
        print(f"mirroring {len(mirrored)} events onto {args.mirror_camera}{silence}")
        events = events + mirrored
    schedule = emission_schedule(events, args.late_fraction, args.late_seconds, args.seed)
    span_s = schedule[-1][0]
    pace = f"at {args.speed:.0f}x (~{span_s / args.speed:.0f}s)" if args.speed > 0 else "unpaced"
    print(f"replaying {len(events)} events spanning {span_s / 60:.1f} min {pace}")

    started = time.monotonic()
    stats = replay(schedule, EventProducer(), args.speed)
    elapsed = time.monotonic() - started
    print(
        f"delivered {stats.delivered}, failed {stats.failed}, pending {stats.pending} "
        f"in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
