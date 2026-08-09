"""Lane-marking map: detect dashes, group into lane lines, fit curves, lay 12-ft ticks.

Phase A of the speed-estimation design, as three reviewable stages on a clean
median-frame image (see median_frame.py):

  stage 1  every detected dash, color-grouped by lane line, gaps drawn as thin
           connectors — "here are the markings and gaps I found"
  stage 2  a smooth parametric curve fitted through each lane line's dashes —
           the lane delineation, following the roadway's curve
  stage 3  each curve subdivided into 12-ft increments derived purely from the
           dash geometry (CA: 12 ft paint + 36 ft gap = 48 ft cycle). Visual
           self-check: a dash must span one tick interval, a gap three.

No satellite imagery, no known camera location — the striping standard itself
is the ruler.

Usage:
    uv run python -m src.perception.speed.lane_map outputs/review/<median>.jpg \
        --out-dir outputs/review --prefix tva43
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# BGR palette for lane-line groups (cycled if more chains than colors).
CHAIN_COLORS = [
    (80, 220, 80),
    (60, 160, 255),
    (230, 100, 230),
    (255, 210, 60),
    (60, 60, 230),
    (200, 200, 80),
]
UNGROUPED_COLOR = (140, 140, 140)


@dataclass
class DashBlob:
    centroid: tuple[float, float]
    p_start: tuple[float, float]  # endpoint nearer the image bottom (near field)
    p_end: tuple[float, float]
    length_px: float
    contour: np.ndarray = field(repr=False)

    @property
    def axis(self) -> tuple[float, float]:
        dx = self.p_end[0] - self.p_start[0]
        dy = self.p_end[1] - self.p_start[1]
        norm = math.hypot(dx, dy) or 1.0
        return (dx / norm, dy / norm)


def detect_dashes(
    image: np.ndarray,
    min_len: float = 5.0,
    max_len: float = 120.0,
    max_width: float = 14.0,
    tophat_px: int = 15,
    thresh: float = 30.0,
) -> list[DashBlob]:
    """Find short elongated bright blobs = painted dashes (no guide needed).

    Top-hat filtering isolates small bright structures regardless of the
    pavement's own brightness gradient; solid edge lines are rejected by the
    max_len cut, noise by the elongation/size cuts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_px, tophat_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dashes = []
    for contour in contours:
        if len(contour) < 4:
            continue
        (cx, cy), (rw, rh), _angle = cv2.minAreaRect(contour)
        length, width = max(rw, rh), min(rw, rh)
        if not (min_len <= length <= max_len and width <= max_width):
            continue
        if width > 0 and length / width < 1.6:
            continue
        pts = contour.reshape(-1, 2).astype(np.float64)
        centered = pts - pts.mean(axis=0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        proj = centered @ direction
        lo, hi = pts[proj.argmin()], pts[proj.argmax()]
        # Orient start = nearer the bottom of the frame (near field first).
        p1, p2 = (tuple(lo), tuple(hi)) if lo[1] > hi[1] else (tuple(hi), tuple(lo))
        dashes.append(
            DashBlob(centroid=(cx, cy), p_start=p1, p_end=p2, length_px=length, contour=contour)
        )
    return dashes


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(dot))


def chain_dashes(
    dashes: list[DashBlob],
    max_gap_factor: float = 6.0,
    min_gap_floor: float = 18.0,
    angle_tol_deg: float = 32.0,
    min_chain: int = 3,
    merge: bool = False,
) -> list[list[DashBlob]]:
    """Greedily link dashes into lane lines, walking near-field to far-field.

    From each unused dash (largest first — near-field anchors are most
    reliable), repeatedly take the closest unused dash that lies roughly along
    the current travel direction within a gap budget scaled to local dash size
    (gaps are ~3x the paint length, and both shrink with distance together).
    """
    unused = sorted(dashes, key=lambda d: -d.length_px)
    used: set[int] = set()
    chains: list[list[DashBlob]] = []

    for seed in unused:
        if id(seed) in used:
            continue
        chain = [seed]
        used.add(id(seed))
        direction = (-seed.axis[0], -seed.axis[1]) if seed.axis[1] > 0 else seed.axis
        while True:
            current = chain[-1]
            budget = max(min_gap_floor, max_gap_factor * max(current.length_px, 6.0))
            best, best_dist = None, budget
            for cand in dashes:
                if id(cand) in used:
                    continue
                vx = cand.centroid[0] - current.centroid[0]
                vy = cand.centroid[1] - current.centroid[1]
                dist = math.hypot(vx, vy)
                if dist == 0 or dist > best_dist:
                    continue
                step = (vx / dist, vy / dist)
                if _angle_between(step, direction) > angle_tol_deg:
                    continue
                cand_axis = cand.axis
                axis_angle = min(
                    _angle_between(cand_axis, step),
                    _angle_between((-cand_axis[0], -cand_axis[1]), step),
                )
                if axis_angle > angle_tol_deg:
                    continue
                best, best_dist = cand, dist
            if best is None:
                break
            vx = best.centroid[0] - current.centroid[0]
            vy = best.centroid[1] - current.centroid[1]
            norm = math.hypot(vx, vy)
            direction = (vx / norm, vy / norm)
            chain.append(best)
            used.add(id(best))
        if len(chain) >= min_chain:
            chains.append(chain)
        else:
            for dash in chain:
                used.discard(id(dash))
    if merge:
        # Experimental: can stitch fragments of one line, but also mis-joins
        # adjacent lines into zigzags — off by default until made side-aware.
        chains = _merge_collinear_chains(chains, angle_tol_deg)
    # Near-field-first ordering within each chain is construction order; sort
    # chains left-to-right by their near-field anchor for stable colors.
    chains.sort(key=lambda c: c[0].centroid[0])
    return chains


def _merge_collinear_chains(
    chains: list[list[DashBlob]], angle_tol_deg: float
) -> list[list[DashBlob]]:
    """Join chain fragments of the same physical lane line end-to-start.

    Perspective stretches near-field gaps beyond the per-dash search budget, so
    one painted line often arrives as 2-3 chains; if chain B starts where chain
    A ends (aligned direction, gap under ~3 local cycles), they are one line.
    """
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(chains):
            tail = a[-1]
            # Local cycle estimate at A's far end.
            if len(a) >= 2:
                prev = a[-2]
                local_cycle = math.hypot(
                    tail.centroid[0] - prev.centroid[0], tail.centroid[1] - prev.centroid[1]
                )
            else:
                local_cycle = 6.0 * tail.length_px
            for j, b in enumerate(chains):
                if i == j:
                    continue
                head = b[0]
                vx = head.centroid[0] - tail.centroid[0]
                vy = head.centroid[1] - tail.centroid[1]
                dist = math.hypot(vx, vy)
                if dist == 0 or dist > 3.0 * max(local_cycle, 12.0):
                    continue
                step = (vx / dist, vy / dist)
                if (
                    _angle_between(step, tail.axis) > angle_tol_deg
                    and _angle_between(step, (-tail.axis[0], -tail.axis[1])) > angle_tol_deg
                ):
                    continue
                if (
                    _angle_between(step, head.axis) > angle_tol_deg
                    and _angle_between(step, (-head.axis[0], -head.axis[1])) > angle_tol_deg
                ):
                    continue
                chains[i] = a + b
                del chains[j]
                merged = True
                break
            if merged:
                break
    return chains


@dataclass
class LaneCurve:
    chain: list[DashBlob]
    ts: np.ndarray = field(repr=False)  # dense parameter samples
    xs: np.ndarray = field(repr=False)
    ys: np.ndarray = field(repr=False)
    arc: np.ndarray = field(repr=False)  # cumulative arc length at each sample

    def point_at_arc(self, s: float) -> tuple[float, float]:
        x = float(np.interp(s, self.arc, self.xs))
        y = float(np.interp(s, self.arc, self.ys))
        return (x, y)

    def tangent_at_arc(self, s: float) -> tuple[float, float]:
        i = int(np.searchsorted(self.arc, s).clip(1, len(self.arc) - 1))
        dx = self.xs[i] - self.xs[i - 1]
        dy = self.ys[i] - self.ys[i - 1]
        norm = math.hypot(dx, dy) or 1.0
        return (dx / norm, dy / norm)


def fit_lane_curve(chain: list[DashBlob], degree: int = 2) -> LaneCurve:
    """Parametric polynomial through each dash's endpoints, by chord length."""
    points = []
    for dash in chain:
        points.extend([dash.p_start, dash.p_end])
    pts = np.array(points, dtype=np.float64)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    deg = min(degree, len(pts) - 1)
    fx = np.polynomial.polynomial.polyfit(chord, pts[:, 0], deg)
    fy = np.polynomial.polynomial.polyfit(chord, pts[:, 1], deg)
    ts = np.linspace(chord[0], chord[-1], 400)
    xs = np.polynomial.polynomial.polyval(ts, fx)
    ys = np.polynomial.polynomial.polyval(ts, fy)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return LaneCurve(chain=chain, ts=ts, xs=xs, ys=ys, arc=arc)


def feet_along_curve(curve: LaneCurve, dash_ft: float, gap_ft: float) -> np.ndarray:
    """Map curve arc length to real-world feet using the dash cycle.

    Each dash contributes two calibration samples: its start (cumulative cycle
    feet) and end (+dash_ft). A spacing more than ~1.6 cycles wide means the
    detector missed a dash — we account a double cycle there.
    """
    cycle_ft = dash_ft + gap_ft

    def arc_of(point: tuple[float, float]) -> float:
        d2 = (curve.xs - point[0]) ** 2 + (curve.ys - point[1]) ** 2
        return float(curve.arc[int(d2.argmin())])

    samples_arc, samples_ft = [], []
    feet = 0.0
    prev_start_arc: float | None = None
    expected_cycle_px: float | None = None
    for dash in curve.chain:
        start_arc, end_arc = arc_of(dash.p_start), arc_of(dash.p_end)
        if prev_start_arc is not None:
            cycle_px = start_arc - prev_start_arc
            cycles = 1
            if expected_cycle_px and cycle_px > 1.6 * expected_cycle_px:
                cycles = round(cycle_px / expected_cycle_px)
            feet += cycle_ft * max(1, cycles)
            expected_cycle_px = cycle_px / max(1, cycles)
        prev_start_arc = start_arc
        samples_arc.extend([start_arc, end_arc])
        samples_ft.extend([feet, feet + dash_ft])
    order = np.argsort(samples_arc)
    arcs = np.array(samples_arc)[order]
    fts = np.array(samples_ft)[order]
    # Feet at every dense curve sample, interpolated (linear between knowns —
    # locally exact enough; the knowns are only ~12-36 ft apart on the road).
    return np.interp(curve.arc, arcs, fts)


# ---------------------------------------------------------------- rendering


def render_stage1(image: np.ndarray, chains: list[list[DashBlob]], leftovers: list[DashBlob]):
    out = image.copy()
    for dash in leftovers:
        cv2.drawContours(out, [dash.contour], -1, UNGROUPED_COLOR, 1, cv2.LINE_AA)
    for i, chain in enumerate(chains):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        for dash in chain:
            cv2.drawContours(out, [dash.contour], -1, color, 2, cv2.LINE_AA)
        for a, b in zip(chain, chain[1:], strict=False):
            p1 = (int(a.p_end[0]), int(a.p_end[1]))
            p2 = (int(b.p_start[0]), int(b.p_start[1]))
            cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)  # the gap connector
        anchor = chain[0].p_start
        cv2.putText(
            out,
            f"lane line {i + 1} ({len(chain)} dashes)",
            (int(anchor[0]) - 30, int(anchor[1]) + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def render_stage2(image: np.ndarray, curves: list[LaneCurve]):
    out = image.copy()
    for i, curve in enumerate(curves):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        pts = np.stack([curve.xs, curve.ys], axis=1).astype(np.int32)
        cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
    return out


def render_stage3(image: np.ndarray, curves: list[LaneCurve], dash_ft: float, gap_ft: float):
    out = render_stage2(image, curves)
    tables = []
    for i, curve in enumerate(curves):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        feet = feet_along_curve(curve, dash_ft, gap_ft)
        max_ft = float(feet[-1])
        for tick_ft in np.arange(0.0, max_ft + 1e-6, dash_ft):
            s = float(np.interp(tick_ft, feet, curve.arc))
            x, y = curve.point_at_arc(s)
            tx, ty = curve.tangent_at_arc(s)
            nx, ny = -ty, tx
            half = 7 if tick_ft % (dash_ft + gap_ft) else 11  # cycle starts bigger
            cv2.line(
                out,
                (int(x - nx * half), int(y - ny * half)),
                (int(x + nx * half), int(y + ny * half)),
                color,
                2,
                cv2.LINE_AA,
            )
            if tick_ft % (2 * (dash_ft + gap_ft)) == 0:  # label every other cycle
                for thickness, tcolor in ((3, (0, 0, 0)), (1, (255, 255, 255))):
                    cv2.putText(
                        out,
                        f"{tick_ft:.0f}ft",
                        (int(x + nx * 14), int(y + ny * 14) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        tcolor,
                        thickness,
                        cv2.LINE_AA,
                    )
        near = np.interp([0.0, dash_ft + gap_ft], feet, curve.arc)
        far_lo = max(0.0, max_ft - (dash_ft + gap_ft))
        far = np.interp([far_lo, max_ft], feet, curve.arc)
        tables.append(
            f"lane line {i + 1}: {len(curve.chain)} dashes, {max_ft:.0f} ft mapped | "
            f"near cycle {near[1] - near[0]:.0f}px, far cycle {far[1] - far[0]:.0f}px | "
            f"scale {((dash_ft + gap_ft) / (near[1] - near[0])):.2f} -> "
            f"{((dash_ft + gap_ft) / max(far[1] - far[0], 1e-6)):.2f} ft/px"
        )
    return out, tables


def main() -> None:
    from src.perception.calibrate_line import review_stamp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Clean road image (median frame)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review"))
    parser.add_argument("--prefix", default="lanes")
    parser.add_argument("--dash-ft", type=float, default=12.0)
    parser.add_argument("--gap-ft", type=float, default=36.0)
    parser.add_argument("--tophat", type=int, default=15)
    parser.add_argument("--thresh", type=float, default=30.0)
    parser.add_argument("--min-chain", type=int, default=4)
    parser.add_argument(
        "--roi",
        help="Road region polygon 'x1,y1 x2,y2 ...' (normalized); dashes outside are ignored",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    dashes = detect_dashes(image, tophat_px=args.tophat, thresh=args.thresh)
    if args.roi:
        h, w = image.shape[:2]
        polygon = np.array(
            [
                [float(v) * s for v, s in zip(pt.split(","), (w, h), strict=True)]
                for pt in args.roi.split()
            ],
            dtype=np.float32,
        )
        dashes = [d for d in dashes if cv2.pointPolygonTest(polygon, d.centroid, False) >= 0]
    chains = chain_dashes(dashes, min_chain=args.min_chain)
    chained_ids = {id(d) for chain in chains for d in chain}
    leftovers = [d for d in dashes if id(d) not in chained_ids]
    curves = [fit_lane_curve(chain) for chain in chains]

    stamp = review_stamp()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    stage3, tables = render_stage3(image, curves, args.dash_ft, args.gap_ft)
    for stage_num, rendered in (
        (1, render_stage1(image, chains, leftovers)),
        (2, render_stage2(image, curves)),
        (3, stage3),
    ):
        path = args.out_dir / f"{args.prefix}_lanes_stage{stage_num}_{stamp}.jpg"
        cv2.imwrite(str(path), rendered)
        paths.append(path)

    print(f"{len(dashes)} dashes -> {len(chains)} lane lines ({len(leftovers)} unmatched)")
    print("\n".join(tables))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
