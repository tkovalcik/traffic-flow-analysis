"""Perspective ruler: locate lane-line dashes and mark real-world distance.

Prototype of the speed-estimation design's calibration stage. Given a clean
road image (see median_frame.py) and a rough guide polyline traced along ONE
dashed lane line, this tool:

1. samples brightness along the guide (with a small perpendicular search
   window, so the guide only needs to be roughly right),
2. finds the bright runs = painted dashes, with sub-step arc positions,
3. treats each dash START as a tick exactly one dash cycle apart (California
   multilane >45 mph: 12 ft stripe + 36 ft gap = 48 ft cycle),
4. renders the ruler: ticks with cumulative feet, and the local scale
   (feet-per-pixel) computed from each consecutive tick pair — the number
   that visibly grows with distance from the camera and is exactly what the
   full homography fit will consume.

Usage:
    uv run python -m src.perception.speed.dash_ruler outputs/review/<median>.jpg \
        --guide "0.49,0.93 0.63,0.81 0.76,0.71 0.86,0.62 0.98,0.51" \
        --cycle-ft 48 --out outputs/review/tva43_ruler.jpg
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PERP_WINDOW_PX = 5  # brightness search half-window around the guide
STEP_PX = 2.0  # arc-length sampling step


@dataclass
class Dash:
    start_arc: float  # arc-length position (px) where the painted dash begins
    end_arc: float
    start_xy: tuple[float, float]  # pixel coords of the dash start

    @property
    def length_px(self) -> float:
        return self.end_arc - self.start_arc


def _polyline_samples(points: list[tuple[float, float]], step: float):
    """Even arc-length samples along a polyline: (arc, (x, y), unit_normal)."""
    samples = []
    arc = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue
        ux, uy = (x2 - x1) / seg_len, (y2 - y1) / seg_len
        nx, ny = -uy, ux
        pos = 0.0
        while pos < seg_len:
            samples.append((arc + pos, (x1 + ux * pos, y1 + uy * pos), (nx, ny)))
            pos += step
        arc += seg_len
    return samples


def find_dashes(
    image: np.ndarray,
    guide_norm: list[tuple[float, float]],
    min_dash_px: float = 4.0,
    window_px: int = PERP_WINDOW_PX,
    thresh_k: float = 0.5,
) -> list[Dash]:
    """Bright runs along the guide polyline = painted dashes."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    guide_px = [(x * w, y * h) for x, y in guide_norm]
    samples = _polyline_samples(guide_px, STEP_PX)

    # Max brightness in a small perpendicular window tolerates guide error.
    values = []
    for _arc, (x, y), (nx, ny) in samples:
        best = 0.0
        for offset in range(-window_px, window_px + 1):
            px, py = int(round(x + nx * offset)), int(round(y + ny * offset))
            if 0 <= px < w and 0 <= py < h:
                best = max(best, float(gray[py, px]))
        values.append(best)
    values = np.array(values)
    threshold = values.mean() + thresh_k * values.std()

    dashes: list[Dash] = []
    run_start = None
    for i, bright in enumerate(values >= threshold):
        if bright and run_start is None:
            run_start = i
        elif not bright and run_start is not None:
            start, end = samples[run_start], samples[i - 1]
            if end[0] - start[0] >= min_dash_px:
                dashes.append(Dash(start[0], end[0], start[1]))
            run_start = None
    if run_start is not None:
        start, end = samples[run_start], samples[-1]
        if end[0] - start[0] >= min_dash_px:
            dashes.append(Dash(start[0], end[0], start[1]))
    return dashes


def render_ruler(
    image: np.ndarray, dashes: list[Dash], cycle_ft: float, dash_ft: float
) -> tuple[np.ndarray, list[str]]:
    """Draw ticks at dash starts with cumulative feet and local ft-per-pixel."""
    out = image.copy()
    table = []
    for i, dash in enumerate(dashes):
        x, y = int(dash.start_xy[0]), int(dash.start_xy[1])
        cv2.circle(out, (x, y), 5, (0, 60, 255), -1, cv2.LINE_AA)
        cumulative_ft = i * cycle_ft
        label = f"{cumulative_ft:.0f}ft"
        if i + 1 < len(dashes):
            cycle_px = dashes[i + 1].start_arc - dash.start_arc
            ft_per_px = cycle_ft / cycle_px
            label += f"  {ft_per_px:.2f} ft/px"
            table.append(
                f"tick {i}: cumulative {cumulative_ft:>4.0f} ft | cycle {cycle_px:6.1f} px "
                f"| scale {ft_per_px:.3f} ft/px | dash {dash.length_px:5.1f} px "
                f"(~{dash.length_px * ft_per_px:4.1f} ft vs {dash_ft:.0f} ft standard)"
            )
        for thickness, color in ((4, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(
                out,
                label,
                (x + 10, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return out, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Clean road image (median frame)")
    parser.add_argument(
        "--guide", required=True, help="Polyline 'x1,y1 x2,y2 ...' (normalized) along ONE dash line"
    )
    parser.add_argument("--cycle-ft", type=float, default=48.0, help="Dash cycle (CA fwy: 48)")
    parser.add_argument("--dash-ft", type=float, default=12.0, help="Painted length (CA fwy: 12)")
    parser.add_argument("--window", type=int, default=PERP_WINDOW_PX, help="Perp search ±px")
    parser.add_argument("--thresh-k", type=float, default=0.5, help="Threshold = mean + k*std")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    guide = [tuple(float(v) for v in pt.split(",")) for pt in args.guide.split()]
    dashes = find_dashes(image, guide, window_px=args.window, thresh_k=args.thresh_k)
    if len(dashes) < 3:
        raise SystemExit(f"only {len(dashes)} dashes found — adjust the guide polyline")
    ruler, table = render_ruler(image, dashes, args.cycle_ft, args.dash_ft)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), ruler)
    print("\n".join(table))
    print(f"{len(dashes)} dashes -> {args.out}")


if __name__ == "__main__":
    main()
