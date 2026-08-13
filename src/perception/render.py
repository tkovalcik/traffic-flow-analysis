"""Render an annotated detection video: boxes, track ids, counting lines, counts.

Deterministic visualization of exactly what the perception pipeline sees — the
validation tool for camera/line calibration and the raw material for demo
videos. Overlays per frame: class-colored boxes with track ids, the counting
line(s) with direction labels, a running per-direction count banner, and the
reconstructed event-time clock.

Presentation mode adds a staged timeline (--stage) that starts on the raw
stream, fades the detection overlay in, then fades the video out so only the
detections remain — the visual argument that events, not pixels, are what flow
through Kafka. --trail-seconds gives each track a vanishing motion trail and
every counted crossing flashes a red pop on the line.

Usage:
    uv run python -m src.perception.render \
        --video data/captures/<clip>.mp4 --metadata data/captures/<clip>.json \
        --camera tva43 --out outputs/tva43_annotated.mp4
    uv run python -m src.perception.render \
        --video data/captures/<clip>.mp4 --metadata data/captures/<clip>.json \
        --camera tva43 --stage raw=8,overlay=14,dark=10 --trail-seconds 2.5 \
        --banner-label "I-580 @ Grand/Lakeshore, Oakland" --banner-tz America/Los_Angeles
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
POP_COLOR = (40, 40, 255)  # the red flash marking one emitted VehicleEvent
POP_SECONDS = 1.2
TRAIL_FLOOR = 0.05  # a trail has visually vanished once it decays to 5%

TrackedFrame = tuple[np.ndarray, int, list[TrackObservation]]


@dataclass(frozen=True)
class StagePlan:
    """Piecewise presentation timeline: raw video → +overlay → detections only.

    Transitions are linear ramps of `fade` seconds ending at each boundary, so
    every stage holds steady for the advertised duration before the next blend.
    """

    raw: float
    overlay: float
    dark: float
    fade: float = 1.5

    @property
    def total(self) -> float:
        return self.raw + self.overlay + self.dark

    def gains(self, t: float) -> tuple[float, float]:
        """(video_gain, overlay_gain) at t seconds — each in [0, 1]."""
        overlay_gain = _ramp_up(t, start=self.raw, fade=self.fade)
        video_gain = 1.0 - _ramp_up(t, start=self.raw + self.overlay, fade=self.fade)
        return video_gain, overlay_gain


def _ramp_up(t: float, start: float, fade: float) -> float:
    """0 before the ramp, 1 from `start` on, linear over the `fade` before it."""
    if fade <= 0:
        return 1.0 if t >= start else 0.0
    return min(1.0, max(0.0, (t - (start - fade)) / fade))


def parse_stage_spec(spec: str, fade: float) -> StagePlan:
    """Parse "raw=8,overlay=14,dark=10" into a StagePlan."""
    durations = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in ("raw", "overlay", "dark"):
            raise ValueError(f"unknown stage {name.strip()!r} (want raw/overlay/dark)")
        durations[name.strip()] = float(value)
    missing = {"raw", "overlay", "dark"} - set(durations)
    if missing:
        raise ValueError(f"stage spec missing {sorted(missing)}")
    return StagePlan(durations["raw"], durations["overlay"], durations["dark"], fade)


def trail_decay(fps: float, trail_seconds: float) -> float:
    """Per-frame decay factor that fades a trail to TRAIL_FLOOR in trail_seconds."""
    return float(TRAIL_FLOOR ** (1.0 / max(1.0, fps * trail_seconds)))


def pop_strength(age_seconds: float, pop_seconds: float = POP_SECONDS) -> float:
    """1 → 0 linear fade of a crossing pop over its lifetime; 0 when expired."""
    if age_seconds < 0 or age_seconds >= pop_seconds:
        return 0.0
    return 1.0 - age_seconds / pop_seconds


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
    mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    lx, ly = p2[0] - p1[0], p2[1] - p1[1]
    norm = (lx**2 + ly**2) ** 0.5 or 1.0
    nx, ny = -ly / norm, lx / norm  # unit normal into the positive half-plane
    offset = max(28, int(0.035 * h))

    flow = line.flow_direction()
    if flow is not None:
        # Motion-gated line: it measures exactly ONE direction. Single label,
        # placed on the side its traffic crosses toward.
        sign = 1 if flow == line.positive_direction else -1
        org = (int(mid_x + sign * nx * offset) - 18, int(mid_y + sign * ny * offset) + 8)
        _outlined_text(frame, flow.value, org, LINE_COLOR)
        return
    # Ungated line counts both ways: label each side with the direction a
    # crossing onto that side is assigned.
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


class TrailCanvas:
    """Persistent glow layer for track trails: decays every frame, never clears.

    Segments are drawn at full class color and multiplied down each frame, so a
    car leaves a tail that vanishes over ~trail_seconds — state the pixels
    remember, the way the stream processor remembers windows.
    """

    def __init__(self, shape: tuple[int, int, int], decay: float):
        self._canvas = np.zeros(shape, dtype=np.float32)
        self._decay = decay
        self._last_center: dict[int, tuple[int, int]] = {}

    def step(self, observations: list[TrackObservation], size: tuple[int, int]) -> None:
        w, h = size
        self._canvas *= self._decay
        seen = set()
        for obs in observations:
            seen.add(obs.track_id)
            center = (int(obs.center[0] * w), int(obs.center[1] * h))
            previous = self._last_center.get(obs.track_id)
            if previous is not None:
                color = CLASS_COLORS[obs.vehicle_class]
                cv2.line(self._canvas, previous, center, color, 3, cv2.LINE_AA)
            self._last_center[obs.track_id] = center
        # Forget tracks that vanished so a re-used id cannot join two vehicles.
        for track_id in list(self._last_center):
            if track_id not in seen:
                del self._last_center[track_id]

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Additively blend the glow onto a frame (saturating, in place)."""
        return cv2.add(frame, self._canvas.astype(np.uint8))


def _draw_pops(
    frame: np.ndarray,
    pops: list[tuple[tuple[int, int], int]],
    index: int,
    fps: float,
) -> None:
    """Red flash + expanding ring where a crossing was just counted."""
    for center, at_index in pops:
        strength = pop_strength((index - at_index) / fps)
        if strength <= 0:
            continue
        radius = 6 + int(26 * (1.0 - strength))
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, POP_COLOR, 2, cv2.LINE_AA)
        cv2.circle(overlay, center, 5, POP_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, strength, frame, 1.0 - strength, 0, frame)


def _prune_pops(pops: list[tuple[tuple[int, int], int]], index: int, fps: float) -> None:
    pops[:] = [p for p in pops if pop_strength((index - p[1]) / fps) > 0]


def _draw_banner(frame: np.ndarray, lines_text: list[str], corner: str = "tr") -> None:
    """Semi-transparent info box. Default top-right: Caltrans cameras burn their
    own timestamp into the top-left/bottom-left, and we must not cover it."""
    h, w = frame.shape[:2]
    widths = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for text in lines_text]
    box_w, box_h = max(330, 20 + max(widths, default=0)), 18 + 20 * len(lines_text)
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


def _banner_clock(anchor: datetime, seconds: float, tz_name: str | None) -> str:
    """The reconstructed event-time clock; local w/ weekday when a tz is given."""
    at = anchor + timedelta(seconds=seconds)
    if tz_name:
        return at.astimezone(ZoneInfo(tz_name)).strftime("%a %b %-d · %-I:%M:%S %p %Z")
    return at.strftime("%Y-%m-%d %H:%M:%SZ")


def annotate_stream(
    tracked_frames: Iterable[TrackedFrame],
    lines: list[CountingLine],
    fps: float,
    anchor: datetime,
    out_path: Path,
    camera_id: str = "",
    banner: bool = True,
    banner_corner: str = "tr",
    banner_label: str = "",
    banner_tz: str | None = None,
    stage_plan: StagePlan | None = None,
    trail_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> RenderResult:
    """Write an annotated video from (frame, index, observations) tuples.

    With stage_plan the render opens on the raw stream, blends the detection
    layer in, then fades the video to black under it; trails and crossing pops
    ride the detection layer. Without it, behavior is the classic overlay.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counter = LineCrossingCounter(lines)
    counts: dict[str, int] = {}
    highlight_until: dict[int, int] = {}  # track_id -> frame index highlight expires
    pops: list[tuple[tuple[int, int], int]] = []  # (pixel center, frame index)
    trails: TrailCanvas | None = None
    writer: cv2.VideoWriter | None = None
    codec = ""
    frames = 0
    stop_after = stage_plan.total if stage_plan is not None else max_seconds

    for frame, index, observations in tracked_frames:
        t = index / fps
        if stop_after is not None and t >= stop_after:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer, codec = _open_writer(out_path, fps, (w, h))
            if trail_seconds > 0:
                trails = TrailCanvas(frame.shape, trail_decay(fps, trail_seconds))
        h, w = frame.shape[:2]

        # Counting is unconditional — the banner tally must not depend on stages.
        for obs in observations:
            for crossing in counter.update(obs.track_id, obs.center):
                counts[crossing.direction.value] = counts.get(crossing.direction.value, 0) + 1
                highlight_until[obs.track_id] = index + int(fps)
                pops.append(((int(obs.center[0] * w), int(obs.center[1] * h)), index))
        if trails is not None:
            trails.step(observations, (w, h))
        _prune_pops(pops, index, fps)

        video_gain, overlay_gain = (1.0, 1.0) if stage_plan is None else stage_plan.gains(t)
        base = (
            frame if video_gain >= 1.0 else (frame.astype(np.float32) * video_gain).astype(np.uint8)
        )
        composed = base
        if overlay_gain > 0:
            overlaid = base.copy()
            for line in lines:
                _draw_line(overlaid, line)
            for obs in observations:
                _draw_box(overlaid, obs, highlighted=highlight_until.get(obs.track_id, -1) >= index)
            if trails is not None:
                overlaid = trails.composite(overlaid)
            _draw_pops(overlaid, pops, index, fps)
            composed = (
                overlaid
                if overlay_gain >= 1.0
                else cv2.addWeighted(overlaid, overlay_gain, base, 1.0 - overlay_gain, 0)
            )
        if banner:
            clock = _banner_clock(anchor, t, banner_tz)
            text_lines = [f"{camera_id}  {clock}"]
            if banner_label:
                text_lines.insert(0, banner_label)
            if overlay_gain > 0.5:
                count_text = (
                    "  ".join(f"{d}:{n}" for d, n in sorted(counts.items())) or "no crossings yet"
                )
                text_lines.append(f"counts  {count_text}")
            _draw_banner(composed, text_lines, banner_corner)
        writer.write(composed)
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
    parser.add_argument(
        "--banner-label", default="", help='Location line for the banner, e.g. "I-80, Emeryville"'
    )
    parser.add_argument(
        "--banner-tz",
        default=None,
        help="IANA timezone: show the clock as local time with weekday (traffic reads by it)",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help='Staged presentation timeline, e.g. "raw=8,overlay=14,dark=10" (seconds)',
    )
    parser.add_argument(
        "--stage-fade", type=float, default=1.5, help="Seconds per stage transition"
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=0.0,
        help="Vanishing motion-trail length per track (0 disables)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="Stop after this much clip time"
    )
    args = parser.parse_args(argv)

    anchor, fps = clip_time_anchor(args.metadata)
    if args.metadata is None:
        anchor = datetime.now(UTC)
    lines = resolve_lines(args.camera, args.line, DEFAULT_LINE)
    out = args.out or Path("outputs") / f"{args.video.stem}_annotated.mp4"
    stage_plan = None if args.stage is None else parse_stage_spec(args.stage, args.stage_fade)
    result = annotate_stream(
        _yolo_frames(args.video, args.model, args.conf, None),
        lines,
        fps,
        anchor,
        out,
        camera_id=args.camera,
        banner=not args.no_banner,
        banner_corner=args.banner_corner,
        banner_label=args.banner_label,
        banner_tz=args.banner_tz,
        stage_plan=stage_plan,
        trail_seconds=args.trail_seconds,
        max_seconds=args.max_seconds,
    )
    counts = ", ".join(f"{d}={n}" for d, n in sorted(result.counts.items())) or "none"
    print(
        f"rendered {result.frames} frames ({result.codec}) crossings: {counts} -> {result.out_path}"
    )


if __name__ == "__main__":
    main()
