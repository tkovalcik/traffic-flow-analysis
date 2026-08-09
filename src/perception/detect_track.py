"""YOLO11n + ByteTrack over video, emitting validated VehicleEvents at counting lines.

The AI boundary of the whole project lives here: frames in, schema-validated
crossing events out. Works on live streams and recorded clips alike; for clips
recorded by src.replay.record, the sidecar metadata anchors event timestamps to
the original capture time, so replayed detections carry true event time.

Usage:
    uv run python -m src.perception.detect_track \
        --video data/captures/<clip>.mp4 --metadata data/captures/<clip>.json \
        --camera tv516 --line "0.05,0.55,0.95,0.55:EB:WB" --out events.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.perception.crossing import CountingLine, LineCrossingCounter, parse_line_spec
from src.streaming.contracts import TravelDirection, VehicleClass, VehicleEvent

COCO_TO_VEHICLE = {
    2: VehicleClass.car,
    3: VehicleClass.motorcycle,
    5: VehicleClass.bus,
    7: VehicleClass.truck,
}

DEFAULT_LINE = "0.05,0.55,0.95,0.55:EB:WB"


@dataclass
class TrackObservation:
    frame_index: int
    track_id: int
    vehicle_class: VehicleClass
    confidence: float
    center: tuple[float, float]  # normalized (cx, cy)


def iter_tracked(
    video_source: str,
    model_name: str = "yolo11n.pt",
    conf: float = 0.35,
    device: str | None = None,
) -> Iterator[TrackObservation]:
    """Run YOLO+ByteTrack over a source, yielding per-frame track observations."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.track(
        source=video_source,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=sorted(COCO_TO_VEHICLE),
        conf=conf,
        device=device,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box_id, cls_id, box_conf, xywhn in zip(
            boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()
        ):
            vclass = COCO_TO_VEHICLE.get(int(cls_id))
            if vclass is None:
                continue
            cx, cy = float(xywhn[0]), float(xywhn[1])
            yield TrackObservation(
                frame_index=frame_index,
                track_id=int(box_id),
                vehicle_class=vclass,
                confidence=float(box_conf),
                center=(cx, cy),
            )


def clip_time_anchor(metadata_path: Path | None) -> tuple[datetime, float]:
    """(first-frame UTC timestamp, fps) for event-time reconstruction of a clip."""
    if metadata_path is None:
        return datetime.now(UTC), 15.0
    meta = json.loads(metadata_path.read_text())
    rec = meta["recorded"]
    anchor = datetime.fromisoformat(rec["first_frame_utc"])
    return anchor, float(rec["fps_writer"])


def events_from_video(
    video_path: Path,
    camera_id: str,
    lines: list[CountingLine],
    metadata_path: Path | None = None,
    model_name: str = "yolo11n.pt",
    conf: float = 0.35,
    device: str | None = None,
) -> list[VehicleEvent]:
    """Full perception pass over a clip: detections → tracks → crossing events."""
    anchor, fps = clip_time_anchor(metadata_path)
    counter = LineCrossingCounter(lines)
    latest: dict[int, TrackObservation] = {}
    events: list[VehicleEvent] = []
    for obs in iter_tracked(str(video_path), model_name, conf, device):
        latest[obs.track_id] = obs
        for crossing in counter.update(obs.track_id, obs.center):
            ts_event = anchor + timedelta(seconds=obs.frame_index / fps)
            events.append(
                VehicleEvent(
                    camera_id=camera_id,
                    ts_event=ts_event,
                    ts_publish=datetime.now(UTC),
                    track_id=crossing.track_id,
                    vehicle_class=obs.vehicle_class,
                    direction=crossing.direction,
                    confidence=obs.confidence,
                )
            )
    return events


def track_stats(
    video_path: Path,
    model_name: str = "yolo11n.pt",
    conf: float = 0.35,
    device: str | None = None,
) -> dict:
    """Detection/track summary for a clip — used to calibrate counting lines.

    The center percentiles show where tracked vehicles actually live in the
    image, i.e. where a counting line must be placed to intersect traffic.
    """
    observations = 0
    tracks: set[int] = set()
    by_class: dict[str, int] = {}
    xs: list[float] = []
    ys: list[float] = []
    for obs in iter_tracked(str(video_path), model_name, conf, device):
        observations += 1
        tracks.add(obs.track_id)
        by_class[obs.vehicle_class.value] = by_class.get(obs.vehicle_class.value, 0) + 1
        xs.append(obs.center[0])
        ys.append(obs.center[1])

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(int(q * len(ordered)), len(ordered) - 1)], 3)

    return {
        "observations": observations,
        "unique_tracks": len(tracks),
        "by_class": by_class,
        "center_x_p10_p50_p90": [pct(xs, q) for q in (0.1, 0.5, 0.9)],
        "center_y_p10_p50_p90": [pct(ys, q) for q in (0.1, 0.5, 0.9)],
    }


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Clip sidecar JSON (event-time anchor)")
    parser.add_argument("--camera", required=True)
    parser.add_argument(
        "--stats", action="store_true", help="Print track statistics instead of emitting events"
    )
    parser.add_argument(
        "--line",
        action="append",
        help=f"x1,y1,x2,y2:POS:NEG (normalized; default {DEFAULT_LINE})",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/events.jsonl"))
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument(
        "--conf", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
    )
    args = parser.parse_args(argv)

    if args.stats:
        print(json.dumps(track_stats(args.video, args.model, args.conf), indent=2))
        return

    lines = [parse_line_spec(s) for s in (args.line or [DEFAULT_LINE])]
    events = events_from_video(args.video, args.camera, lines, args.metadata, args.model, args.conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for event in events:
            fh.write(json.dumps(event.to_avro_dict()) + "\n")

    by_direction: dict[TravelDirection, int] = {}
    for e in events:
        by_direction[e.direction] = by_direction.get(e.direction, 0) + 1
    summary = ", ".join(f"{d.value}={n}" for d, n in sorted(by_direction.items())) or "none"
    print(f"{len(events)} crossing events ({summary}) -> {args.out}")


if __name__ == "__main__":
    main()
