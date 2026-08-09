"""Render an annotated detection video: boxes, track ids, counting lines, counts.

Deterministic visualization of exactly what the perception pipeline sees — the
validation tool for camera/line calibration and the raw material for demo
videos. Overlays per frame: class-colored boxes with track ids, the counting
line(s) with direction labels, a running per-direction count banner, and the
reconstructed event-time clock.

Usage:
    uv run python -m src.perception.render \
        --video data/captures/<clip>.mp4 --metadata data/captures/<clip>.json \
        --camera tva43 --out outputs/tva43_annotated.mp4
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from src.perception.camera_config import resolve_lines
from src.perception.crossing import CountingLine, LineCrossingCounter
from src.perception.detect_track import (
    DEFAULT_LINE,
    TrackObservation,
    clip_time_anchor,
    iter_frames_tracked,
)
from src.streaming.contracts import VehicleClass

# BGR colors per class — fixed so every render reads the same.
CLASS_COLORS: dict[VehicleClass, tuple[int, int, int]] = {
    VehicleClass.car: (96, 200, 96),
    VehicleClass.truck: (60, 140, 255),
    VehicleClass.bus: (200, 90, 200),
    VehicleClass.motorcycle: (255, 200, 60),
}
LINE_COLOR = (0, 220, 255)
BANNER_COLOR = (24, 24, 24)
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT_COLOR = (0, 60, 255)

TrackedFrame = tuple[np.ndarray, int, list[TrackObservation]]


@dataclass
class RenderResult:
    out_path: Path
    frames: int
    counts: dict[str, int] = field(default_factory=dict)
    codec: str = ""


def _open_writer(out_path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Prefer H.264 (browser-playable) and fall back to MPEG-4 part 2."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"no usable codec for {out_path}")


def _outlined_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.8,
) -> None:
    """Text with a dark outline so it stays readable over any pavement."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_line(frame: np.ndarray, line: CountingLine) -> None:
    h, w = frame.shape[:2]
    p1 = (int(line.p1[0] * w), int(line.p1[1] * h))
    p2 = (int(line.p2[0] * w), int(line.p2[1] * h))
    cv2.line(frame, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
    # Direction labels sit on their own side of the line: crossing toward a
    # label means traveling in that label's direction. Unit normal (-ly, lx)
    # points into the line's positive half-plane.
    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    lx, ly = p2[0] - p1[0], p2[1] - p1[1]
    norm = (lx**2 + ly**2) ** 0.5 or 1.0
    nx, ny = -ly / norm, lx / norm
    offset = max(28, int(0.035 * h))
    for direction, sign in ((line.positive_direction, 1), (line.negative_direction, -1)):
        org = (int(mid_x + sign * nx * offset) - 18, int(mid_y + sign * ny * offset) + 8)
        _outlined_text(frame, direction.value, org, LINE_COLOR)


def _draw_box(frame: np.ndarray, obs: TrackObservation, highlighted: bool) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = obs.box
    pt1 = (int(x1 * w), int(y1 * h))
    pt2 = (int(x2 * w), int(y2 * h))
    color = HIGHLIGHT_COLOR if highlighted else CLASS_COLORS[obs.vehicle_class]
    cv2.rectangle(frame, pt1, pt2, color, 3 if highlighted else 2, cv2.LINE_AA)
    label = f"#{obs.track_id} {obs.vehicle_class.value} {obs.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (pt1[0], max(12, pt1[1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_banner(frame: np.ndarray, lines_text: list[str], corner: str = "tr") -> None:
    """Semi-transparent info box. Default top-right: Caltrans cameras burn their
    own timestamp into the top-left/bottom-left, and we must not cover it."""
    h, w = frame.shape[:2]
    box_w, box_h = 330, 18 + 20 * len(lines_text)
    x0 = 0 if corner in ("tl", "bl") else w - box_w
    y0 = 0 if corner in ("tl", "tr") else h - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), BANNER_COLOR, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, text in enumerate(lines_text):
        cv2.putText(
            frame,
            text,
            (x0 + 10, y0 + 22 + 20 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )


def annotate_stream(
    tracked_frames: Iterable[TrackedFrame],
    lines: list[CountingLine],
    fps: float,
    anchor: datetime,
    out_path: Path,
    camera_id: str = "",
    banner: bool = True,
    banner_corner: str = "tr",
) -> RenderResult:
    """Write an annotated video from (frame, index, observations) tuples."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counter = LineCrossingCounter(lines)
    counts: dict[str, int] = {}
    highlight_until: dict[int, int] = {}  # track_id -> frame index highlight expires
    writer: cv2.VideoWriter | None = None
    codec = ""
    frames = 0

    for frame, index, observations in tracked_frames:
        if writer is None:
            h, w = frame.shape[:2]
            writer, codec = _open_writer(out_path, fps, (w, h))
        for line in lines:
            _draw_line(frame, line)
        for obs in observations:
            for crossing in counter.update(obs.track_id, obs.center):
                counts[crossing.direction.value] = counts.get(crossing.direction.value, 0) + 1
                highlight_until[obs.track_id] = index + int(fps)
            _draw_box(frame, obs, highlighted=highlight_until.get(obs.track_id, -1) >= index)
        if banner:
            clock = (anchor + timedelta(seconds=index / fps)).strftime("%Y-%m-%d %H:%M:%SZ")
            count_text = (
                "  ".join(f"{d}:{n}" for d, n in sorted(counts.items())) or "no crossings yet"
            )
            _draw_banner(frame, [f"{camera_id}  {clock}", f"counts  {count_text}"], banner_corner)
        writer.write(frame)
        frames += 1

    if writer is not None:
        writer.release()
    if frames == 0:
        raise RuntimeError("no frames to render")
    return RenderResult(out_path=out_path, frames=frames, counts=counts, codec=codec)


def _yolo_frames(
    video: Path, model: str, conf: float, device: str | None
) -> Iterator[TrackedFrame]:
    yield from iter_frames_tracked(str(video), model, conf, device)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Clip sidecar JSON (event-time anchor)")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--line", action="append", help=f"Line spec (default {DEFAULT_LINE})")
    parser.add_argument("--out", type=Path, help="Default: outputs/<video-stem>_annotated.mp4")
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument(
        "--conf", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
    )
    parser.add_argument(
        "--no-banner", action="store_true", help="Boxes and lines only (class-video mode)"
    )
    parser.add_argument("--banner-corner", choices=["tl", "tr", "bl", "br"], default="tr")
    args = parser.parse_args(argv)

    anchor, fps = clip_time_anchor(args.metadata)
    if args.metadata is None:
        anchor = datetime.now(UTC)
    lines = resolve_lines(args.camera, args.line, DEFAULT_LINE)
    out = args.out or Path("outputs") / f"{args.video.stem}_annotated.mp4"
    result = annotate_stream(
        _yolo_frames(args.video, args.model, args.conf, None),
        lines,
        fps,
        anchor,
        out,
        camera_id=args.camera,
        banner=not args.no_banner,
        banner_corner=args.banner_corner,
    )
    counts = ", ".join(f"{d}={n}" for d, n in sorted(result.counts.items())) or "none"
    print(
        f"rendered {result.frames} frames ({result.codec}) crossings: {counts} -> {result.out_path}"
    )


if __name__ == "__main__":
    main()
