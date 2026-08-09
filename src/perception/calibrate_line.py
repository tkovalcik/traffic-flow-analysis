"""Propose counting lines from observed track motion.

Instead of hand-placing lines, analyze where vehicles actually travel: cluster
tracks into the two opposing flows, then for each flow propose a line

- perpendicular to that flow's mean motion direction,
- positioned inside the flow's own lane band (biased toward the lower half of
  the frame where detections are largest/most reliable),
- spanning the flow's full lateral extent plus margin, so no lane escapes
  off-screen uncounted.

Direction labels are derived from the camera's known compass orientation
(`--up-frame`, e.g. EB when receding traffic is eastbound), so each proposed
line emits the correct EB/WB (or NB/SB) labels.

Usage:
    uv run python -m src.perception.calibrate_line \
        --video data/captures/<clip>.mp4 --max-frames 3600 --up-frame EB \
        --preview outputs/review/<camera>_line_proposal.jpg

Paste the printed specs into configs/cameras.json after reviewing the preview.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.streaming.contracts import TravelDirection

OPPOSITE = {
    TravelDirection.EB: TravelDirection.WB,
    TravelDirection.WB: TravelDirection.EB,
    TravelDirection.NB: TravelDirection.SB,
    TravelDirection.SB: TravelDirection.NB,
}

MIN_TRACK_POINTS = 5
MIN_NET_DISPLACEMENT = 0.03  # normalized units — drop parked/jittering tracks


@dataclass
class TrackPath:
    track_id: int
    points: list[tuple[float, float]]

    @property
    def motion(self) -> tuple[float, float]:
        (x0, y0), (x1, y1) = self.points[0], self.points[-1]
        return (x1 - x0, y1 - y0)

    @property
    def displacement(self) -> float:
        dx, dy = self.motion
        return math.hypot(dx, dy)


@dataclass
class ProposedLine:
    spec: str
    tracks: int
    mean_motion_deg: float  # image-coords angle of the flow this line counts


def collect_paths(observations) -> list[TrackPath]:
    """Group (track_id, center) observations into filtered track paths."""
    by_id: dict[int, list[tuple[float, float]]] = {}
    for obs in observations:
        by_id.setdefault(obs.track_id, []).append(obs.center)
    paths = [TrackPath(tid, pts) for tid, pts in by_id.items() if len(pts) >= MIN_TRACK_POINTS]
    return [p for p in paths if p.displacement >= MIN_NET_DISPLACEMENT]


def split_flows(paths: list[TrackPath]) -> tuple[list[TrackPath], list[TrackPath]]:
    """Split tracks into two opposing flows using the strongest track as reference."""
    if not paths:
        return [], []
    ref = max(paths, key=lambda p: p.displacement).motion
    with_ref, against_ref = [], []
    for p in paths:
        dx, dy = p.motion
        (with_ref if dx * ref[0] + dy * ref[1] >= 0 else against_ref).append(p)
    return with_ref, against_ref


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def propose_line(
    flow: list[TrackPath],
    up_frame: TravelDirection,
    lateral_margin: float = 0.06,
) -> ProposedLine | None:
    """One counting line for one flow: perpendicular, in-band, full lateral span."""
    if len(flow) < 3:
        return None
    # Mean unit motion of the flow (image coords: +y is DOWN the frame).
    mx = my = 0.0
    for p in flow:
        dx, dy = p.motion
        norm = p.displacement or 1.0
        mx, my = mx + dx / norm, my + dy / norm
    norm = math.hypot(mx, my) or 1.0
    mx, my = mx / norm, my / norm

    # Anchor: lateral center of the band, longitudinally biased toward the
    # nearer (lower-in-frame) 60th percentile of this flow's own points.
    points = [pt for p in flow for pt in p.points]
    lateral = [-my * x + mx * y for x, y in points]  # coordinate along the line
    ys = [y for _x, y in points]
    xs = [x for x, _y in points]
    cy = _quantile(ys, 0.6)
    cx = _quantile(xs, 0.5)

    # Endpoints: span the flow's lateral extent (p05..p95) plus margin.
    lat_c = -my * cx + mx * cy
    lat_lo = _quantile(lateral, 0.05) - lateral_margin
    lat_hi = _quantile(lateral, 0.95) + lateral_margin
    px, py = -my, mx  # unit vector along the line (perpendicular to motion)

    def endpoint(lat: float) -> tuple[float, float]:
        x = cx + (lat - lat_c) * px
        y = cy + (lat - lat_c) * py
        return (min(max(x, 0.02), 0.98), min(max(y, 0.02), 0.98))

    (x1, y1), (x2, y2) = endpoint(lat_lo), endpoint(lat_hi)

    # This flow's compass label: up-frame motion (my < 0) means `up_frame`.
    flow_direction = up_frame if my < 0 else OPPOSITE[up_frame]
    # The flow crosses onto the line's positive side iff cross(L, m) > 0.
    line_dx, line_dy = x2 - x1, y2 - y1
    crosses_positive = (line_dx * my - line_dy * mx) > 0
    pos = flow_direction if crosses_positive else OPPOSITE[flow_direction]
    neg = OPPOSITE[pos]

    # The flow's mean motion rides along in the spec so the crossing counter can
    # motion-gate: only tracks moving with this flow may fire this line.
    spec = f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}:{pos.value}:{neg.value}:{mx:.3f},{my:.3f}"
    angle = math.degrees(math.atan2(my, mx))
    return ProposedLine(spec=spec, tracks=len(flow), mean_motion_deg=round(angle, 1))


def propose_lines_for_clip(
    video: Path,
    up_frame: TravelDirection,
    max_frames: int | None = None,
    model_name: str = "yolo11n.pt",
    conf: float = 0.35,
) -> list[ProposedLine]:
    """YOLO+ByteTrack pass over (part of) a clip → proposed line per flow."""
    from itertools import islice

    from src.perception.detect_track import iter_frames_tracked

    frames = iter_frames_tracked(str(video), model_name, conf)
    if max_frames is not None:
        frames = islice(frames, max_frames)
    observations = [obs for _f, _i, obs_list in frames for obs in obs_list]
    proposals = []
    for flow in split_flows(collect_paths(observations)):
        line = propose_line(flow, up_frame)
        if line:
            proposals.append(line)
    return proposals


def render_preview(video: Path, proposals: list[ProposedLine], out_path: Path) -> None:
    """Draw proposed lines over a frame from the clip for human review."""
    import cv2

    from src.perception.crossing import parse_line_spec
    from src.perception.render import _draw_line

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.5))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {video}")
    for proposal in proposals:
        _draw_line(frame, parse_line_spec(proposal.spec))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--up-frame",
        default="EB",
        choices=[d.value for d in TravelDirection],
        help="Compass direction of traffic receding up-frame (see direction-verification)",
    )
    parser.add_argument("--max-frames", type=int, help="Analyze only the first N frames")
    parser.add_argument("--preview", type=Path, help="Write a review image with the lines drawn")
    parser.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument(
        "--conf", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
    )
    args = parser.parse_args(argv)

    proposals = propose_lines_for_clip(
        args.video, TravelDirection(args.up_frame), args.max_frames, args.model, args.conf
    )
    if not proposals:
        raise SystemExit("no usable flows found — clip too short or too little traffic?")
    for p in proposals:
        print(f'"{p.spec}"  # {p.tracks} tracks, flow angle {p.mean_motion_deg} deg')
    if args.preview:
        render_preview(args.video, proposals, args.preview)
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
