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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
        # Lane paint is WHITE: bright and unsaturated. Kills foliage/horizon
        # clutter that passes the shape tests.
        blob = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        mean_s = float(cv2.mean(hsv[:, :, 1], mask=blob)[0])
        mean_v = float(cv2.mean(hsv[:, :, 2], mask=blob)[0])
        if mean_s > 70 or mean_v < 120:
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


def detect_dashes_multiscale(
    image: np.ndarray,
    y_split: float = 0.55,
    upscale: float = 2.0,
    **kwargs,
) -> list[DashBlob]:
    """detect_dashes plus a zoomed pass over the far field (above y_split).

    Distant dashes shrink to a few pixels and fall below the base detector's
    size cuts; upscaling that region first gives them enough pixels — the same
    slicing/zooming principle SAHI uses for small-object detection.
    """
    base = detect_dashes(image, **kwargs)
    h = image.shape[0]
    split_px = int(h * y_split)
    if split_px < 20:
        return base
    far = cv2.resize(image[:split_px], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    far_kwargs = dict(kwargs)
    far_kwargs["tophat_px"] = max(7, int(kwargs.get("tophat_px", 15) * 0.7))
    far_kwargs["thresh"] = kwargs.get("thresh", 30.0) * 0.8
    extra = []
    for dash in detect_dashes(far, **far_kwargs):
        scaled = DashBlob(
            centroid=(dash.centroid[0] / upscale, dash.centroid[1] / upscale),
            p_start=(dash.p_start[0] / upscale, dash.p_start[1] / upscale),
            p_end=(dash.p_end[0] / upscale, dash.p_end[1] / upscale),
            length_px=dash.length_px / upscale,
            contour=(dash.contour.astype(np.float64) / upscale).astype(np.int32),
        )
        duplicate = any(
            math.hypot(scaled.centroid[0] - b.centroid[0], scaled.centroid[1] - b.centroid[1]) < 5
            for b in base
        )
        if not duplicate:
            extra.append(scaled)
    return base + extra


def chain_score(chain: list[DashBlob]) -> float:
    """Confidence of a lane-line chain: many dashes, long paint, smooth spacing.

    Perspective makes true cycle spacings shrink smoothly along a real lane
    line; erratic spacing ratios mean the chain hopped between features.
    """
    if len(chain) < 3:
        return 0.0
    spacings = [
        math.hypot(b.centroid[0] - a.centroid[0], b.centroid[1] - a.centroid[1])
        for a, b in zip(chain, chain[1:], strict=False)
    ]
    ratios = [b / a for a, b in zip(spacings, spacings[1:], strict=False) if a > 0]
    if ratios:
        mean_r = sum(ratios) / len(ratios)
        var = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
        cv = math.sqrt(var) / mean_r if mean_r > 0 else 1.0
    else:
        cv = 0.0
    regularity = 1.0 / (1.0 + 3.0 * cv)
    mean_len = sum(d.length_px for d in chain) / len(chain)
    return len(chain) * mean_len * regularity


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
        chains = merge_by_curve(chains)
    # Near-field-first ordering within each chain is construction order; sort
    # chains left-to-right by their near-field anchor for stable colors.
    chains.sort(key=lambda c: c[0].centroid[0])
    return chains


def _step(a: DashBlob, b: DashBlob) -> tuple[float, float]:
    """Unit vector from dash A's centroid to dash B's."""
    vx = b.centroid[0] - a.centroid[0]
    vy = b.centroid[1] - a.centroid[1]
    norm = math.hypot(vx, vy) or 1.0
    return (vx / norm, vy / norm)


def _axis_angle(step: tuple[float, float], axis: tuple[float, float]) -> float:
    """Angle between a step vector and an undirected axis, folded into [0, 90]."""
    angle = _angle_between(step, axis)
    return min(angle, 180.0 - angle)


def link_verdict(a: DashBlob, b: DashBlob, axis_tol_deg: float = 22.0) -> str:
    """Judge one chain link by parallelism to the dashes' own painted strokes.

    On a real dashed lane line the step from one dash to the next runs the way
    the paint itself points. good = parallel to both strokes; suspicious =
    parallel to exactly one (a kink, or an interloper on one side); bad =
    parallel to neither (this link must not exist).
    """
    step = _step(a, b)
    ok_a = _axis_angle(step, a.axis) <= axis_tol_deg
    ok_b = _axis_angle(step, b.axis) <= axis_tol_deg
    if ok_a and ok_b:
        return "good"
    if ok_a or ok_b:
        return "suspicious"
    return "bad"


def _link_deviation(a: DashBlob, b: DashBlob) -> float:
    """Worst step-vs-stroke angle of a link — 'good' iff <= the axis tolerance."""
    step = _step(a, b)
    return max(_axis_angle(step, a.axis), _axis_angle(step, b.axis))


def _turn_flags(chain: list[DashBlob], floor_deg: float, mad_k: float) -> list[int]:
    """Indices of interior dashes whose turning angle is a robust outlier.

    median + k*MAD adapts to genuinely curved roads (where every turning angle
    is nonzero); the absolute floor keeps perspective-inflated far-field
    angles from flagging on perfectly straight chains (MAD ~ 0).
    """
    if len(chain) < 3:
        return []
    steps = [_step(a, b) for a, b in zip(chain, chain[1:], strict=False)]
    angles = [_angle_between(s1, s2) for s1, s2 in zip(steps, steps[1:], strict=False)]
    med = float(np.median(angles))
    mad = float(np.median([abs(t - med) for t in angles]))
    return [
        j + 1 for j, theta in enumerate(angles) if theta > floor_deg and theta > med + mad_k * mad
    ]


def filter_chain_consistency(
    chains: list[list[DashBlob]],
    axis_tol_deg: float = 22.0,
    turn_floor_deg: float = 20.0,
    turn_mad_k: float = 3.0,
    min_chain: int = 3,
) -> tuple[list[list[DashBlob]], list[DashBlob]]:
    """Enforce local sequence consistency inside every chain.

    Primary check: every link must be parallel to its dashes' stroke axes
    (link_verdict) — this judges endpoint dashes too, via their single link,
    which turning angle alone cannot see. Secondary: turning-angle outliers at
    interior dashes. Repairs: eject a dash when its neighbors then reconnect
    cleanly (it was an interloper), otherwise split the chain at the offending
    link (it was two different features); an endpoint whose only link can't be
    repaired is dropped back to unmatched. Each chain runs to a fixpoint.
    Returns (surviving chains, dropped dashes).
    """
    done: list[list[DashBlob]] = []
    dropped: list[DashBlob] = []
    work = [list(chain) for chain in chains]
    while work:
        cur = work.pop()

        # Endpoint dashes are judged by their single link: a non-good end link
        # means either the end dash is wrong or its neighbor is an interloper —
        # eject whichever removal reconnects the chain cleanly.
        trimming = True
        while trimming:
            trimming = False
            if len(cur) >= 2 and link_verdict(cur[0], cur[1], axis_tol_deg) != "good":
                if len(cur) >= 3 and link_verdict(cur[0], cur[2], axis_tol_deg) == "good":
                    dropped.append(cur.pop(1))
                else:
                    dropped.append(cur.pop(0))
                trimming = True
            if len(cur) >= 2 and link_verdict(cur[-2], cur[-1], axis_tol_deg) != "good":
                if len(cur) >= 3 and link_verdict(cur[-3], cur[-1], axis_tol_deg) == "good":
                    dropped.append(cur.pop(-2))
                else:
                    dropped.append(cur.pop())
                trimming = True

        # First interior bad link (both dashes have another neighbor).
        bad_i = next(
            (
                i
                for i in range(1, len(cur) - 2)
                if link_verdict(cur[i], cur[i + 1], axis_tol_deg) == "bad"
            ),
            None,
        )
        if bad_i is not None:
            i = bad_i
            # Removing which side of the link reconnects the chain better?
            dev_drop_i = _link_deviation(cur[i - 1], cur[i + 1])
            dev_drop_next = _link_deviation(cur[i], cur[i + 2])
            if min(dev_drop_i, dev_drop_next) <= axis_tol_deg:
                dropped.append(cur.pop(i if dev_drop_i <= dev_drop_next else i + 1))
                work.append(cur)
            else:
                work.extend([cur[: i + 1], cur[i + 1 :]])
            continue

        flags = _turn_flags(cur, turn_floor_deg, turn_mad_k)
        if flags:
            j = flags[0]
            if _link_deviation(cur[j - 1], cur[j + 1]) <= axis_tol_deg:
                dropped.append(cur.pop(j))
                work.append(cur)
            elif _link_deviation(cur[j - 1], cur[j]) >= _link_deviation(cur[j], cur[j + 1]):
                work.extend([cur[:j], cur[j:]])
            else:
                work.extend([cur[: j + 1], cur[j + 1 :]])
            continue

        if len(cur) >= min_chain:
            done.append(cur)
        else:
            dropped.extend(cur)
    done.sort(key=lambda c: c[0].centroid[0])
    return done, dropped


def _extended_curve_samples(chain: list[DashBlob], degree: int = 2, n: int = 600):
    """Dense (xs, ys, ts, total) of the chain's curve, extrapolated both ways."""
    points = []
    for dash in chain:
        points.extend([dash.p_start, dash.p_end])
    pts = np.array(points, dtype=np.float64)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    deg = min(degree, len(pts) - 1)
    fx = np.polynomial.polynomial.polyfit(chord, pts[:, 0], deg)
    fy = np.polynomial.polynomial.polyfit(chord, pts[:, 1], deg)
    total = float(chord[-1])
    ts = np.linspace(-0.8 * total, 2.4 * total, n)
    xs = np.polynomial.polynomial.polyval(ts, fx)
    ys = np.polynomial.polynomial.polyval(ts, fy)
    return xs, ys, ts, total


def merge_by_curve(chains: list[list[DashBlob]], tol_factor: float = 0.5) -> list[list[DashBlob]]:
    """Join fragments that lie ON each other's extrapolated curve.

    Adjacent lane lines run parallel a full lane width apart, so requiring the
    candidate's dashes to sit within a fraction of a dash length of the
    extended curve merges same-line fragments while rejecting the cross-line
    zigzags that naive end-to-end merging produced.
    """
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(chains):
            if len(a) < 3:
                continue
            xs, ys, ts, total = _extended_curve_samples(a)
            for j, b in enumerate(chains):
                if i == j:
                    continue
                dists, params = [], []
                for dash in b:
                    d2 = (xs - dash.centroid[0]) ** 2 + (ys - dash.centroid[1]) ** 2
                    k = int(d2.argmin())
                    dists.append(math.sqrt(float(d2[k])))
                    params.append(float(ts[k]))
                tol = max(4.0, tol_factor * (sum(d.length_px for d in b) / len(b)))
                if max(dists) > tol:
                    continue
                after = all(p > 0.9 * total for p in params)
                before = all(p < 0.1 * total for p in params)
                if not (after or before):
                    continue
                chains[i] = a + b if after else b + a
                del chains[j]
                changed = True
                break
            if changed:
                break
    return chains


def filter_parallel_to_anchor(
    chains: list[list[DashBlob]], angle_tol_deg: float = 28.0
) -> tuple[list[list[DashBlob]], list[list[DashBlob]]]:
    """Keep the best-scoring chain and only chains roughly parallel to it.

    Lane lines of one roadway are parallel curves; anything else (barrier
    edges, horizon clutter, sign posts) runs its own way. Tangents are
    compared at matching image depths (same y) — the perspective-safe proxy.
    Returns (kept, rejected).
    """
    if len(chains) <= 1:
        return chains, []
    anchor = max(chains, key=chain_score)
    xs, ys, _, _ = _extended_curve_samples(anchor)
    kept, rejected = [], []
    for chain in chains:
        if chain is anchor:
            kept.append(chain)
            continue
        deviations = []
        for a, b in zip(chain, chain[1:], strict=False):
            vx = b.centroid[0] - a.centroid[0]
            vy = b.centroid[1] - a.centroid[1]
            norm = math.hypot(vx, vy) or 1.0
            step = (vx / norm, vy / norm)
            mid_y = (a.centroid[1] + b.centroid[1]) / 2
            k = int(np.abs(ys - mid_y).argmin())
            k0, k1 = max(0, k - 1), min(len(xs) - 1, k + 1)
            tnorm = math.hypot(xs[k1] - xs[k0], ys[k1] - ys[k0]) or 1.0
            tangent = ((xs[k1] - xs[k0]) / tnorm, (ys[k1] - ys[k0]) / tnorm)
            angle = _angle_between(step, tangent)
            deviations.append(min(angle, 180.0 - angle))
        mean_dev = sum(deviations) / len(deviations)
        (kept if mean_dev <= angle_tol_deg else rejected).append(chain)
    return kept, rejected


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
    parser.add_argument("--min-chain", type=int, default=3)
    parser.add_argument(
        "--roi",
        help="Road region polygon 'x1,y1 x2,y2 ...' (normalized); dashes outside are ignored",
    )
    parser.add_argument(
        "--far-boost",
        action="store_true",
        help="Second detection pass over an upscaled far field (small-object recovery)",
    )
    parser.add_argument("--y-split", type=float, default=0.55, help="Far-field boundary")
    parser.add_argument(
        "--no-parallel-filter",
        action="store_true",
        help="Keep chains regardless of parallelism to the best-scoring anchor",
    )
    parser.add_argument(
        "--no-consistency-filter",
        action="store_true",
        help="Skip the link-parallelism / turning-angle chain cleanup",
    )
    parser.add_argument(
        "--scene-mask",
        type=Path,
        help="Road-surface mask PNG from scene_mask.py "
        "(default: configs/scene_masks/<prefix>.png when it exists)",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    mask_path = args.scene_mask or Path("configs/scene_masks") / f"{args.prefix}.png"
    scene_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if scene_mask is not None and scene_mask.shape != image.shape[:2]:
        raise SystemExit(f"scene mask {mask_path} shape {scene_mask.shape} != image")
    if args.far_boost:
        dashes = detect_dashes_multiscale(
            image, y_split=args.y_split, tophat_px=args.tophat, thresh=args.thresh
        )
    else:
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
    if scene_mask is not None:
        before = len(dashes)
        dashes = [
            d
            for d in dashes
            if scene_mask[
                min(int(d.centroid[1]), scene_mask.shape[0] - 1),
                min(int(d.centroid[0]), scene_mask.shape[1] - 1),
            ]
            > 0
        ]
        print(f"scene mask {mask_path.name}: {before} -> {len(dashes)} dashes on pavement")
    chains = chain_dashes(dashes, min_chain=args.min_chain, merge=False)
    if args.no_consistency_filter:
        ejected: list[DashBlob] = []
    else:
        chains, ejected = filter_chain_consistency(chains, min_chain=args.min_chain)
    chains = merge_by_curve(chains)
    if not args.no_parallel_filter:
        chains, rejected = filter_parallel_to_anchor(chains)
    else:
        rejected = []
    chains.sort(key=lambda c: c[0].centroid[0])
    chained_ids = {id(d) for chain in chains for d in chain}
    leftovers = [d for d in dashes if id(d) not in chained_ids]
    curves = [fit_lane_curve(chain) for chain in chains]

    stamp = review_stamp()
    args.out_dir = args.out_dir / stamp  # one folder per review round
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

    print(
        f"{len(dashes)} dashes -> {len(chains)} lane lines "
        f"({len(leftovers)} unmatched, {len(ejected)} ejected as chain-inconsistent, "
        f"{len(rejected)} chains rejected as non-parallel)"
    )
    for i, chain in enumerate(chains):
        print(f"  lane line {i + 1}: score {chain_score(chain):.0f}")
    print("\n".join(tables))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
