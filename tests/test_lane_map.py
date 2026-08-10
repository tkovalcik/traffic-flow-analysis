"""Tests for lane-marking detection/grouping/calibration on synthetic imagery."""

import math

import numpy as np
import pytest

from src.perception.speed.lane_map import (
    DashBlob,
    chain_dashes,
    chain_score,
    detect_dashes,
    feet_along_curve,
    filter_chain_consistency,
    fit_lane_curve,
    link_verdict,
    merge_by_curve,
)

cv2 = pytest.importorskip("cv2")

DASH_LEN = 30
GAP_LEN = 90  # 12ft : 36ft ratio, as on CA freeways


def make_two_dashed_lines(n_dashes=5):
    """Two vertical dashed lines at x=120 and x=280 on gray pavement."""
    image = np.full((640, 400, 3), 90, dtype=np.uint8)
    for x in (120, 280):
        y = 600
        for _ in range(n_dashes):
            cv2.rectangle(image, (x - 2, y - DASH_LEN), (x + 2, y), (255, 255, 255), -1)
            y -= DASH_LEN + GAP_LEN
    return image


def test_detects_all_dashes():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    assert len(dashes) == 10
    lengths = [d.length_px for d in dashes]
    assert all(length == pytest.approx(DASH_LEN, abs=4) for length in lengths)


def test_chains_group_by_physical_line():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    chains = chain_dashes(dashes)
    assert len(chains) == 2
    for chain in chains:
        assert len(chain) == 5
        xs = [d.centroid[0] for d in chain]
        assert max(xs) - min(xs) < 8  # all dashes from ONE vertical line
        ys = [d.centroid[1] for d in chain]
        assert ys == sorted(ys, reverse=True)  # ordered near-field -> far-field


def test_curve_fit_passes_through_dashes():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    chain = chain_dashes(dashes)[0]
    curve = fit_lane_curve(chain)
    for dash in chain:
        d2 = (curve.xs - dash.centroid[0]) ** 2 + (curve.ys - dash.centroid[1]) ** 2
        assert float(np.sqrt(d2.min())) < 3.0


def test_feet_mapping_matches_dash_geometry():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    chain = chain_dashes(dashes)[0]
    curve = fit_lane_curve(chain)
    feet = feet_along_curve(curve, dash_ft=12.0, gap_ft=36.0)
    # Constant scale here: 120 px per 48 ft cycle -> 0.4 ft/px everywhere,
    # so 12 ft of road must always be ~30 px of arc.
    arc_at_12ft = float(np.interp(12.0, feet, curve.arc))
    arc_at_60ft = float(np.interp(60.0, feet, curve.arc)) - float(np.interp(48.0, feet, curve.arc))
    assert arc_at_12ft == pytest.approx(DASH_LEN, abs=5)
    assert arc_at_60ft == pytest.approx(DASH_LEN, abs=5)
    assert float(feet[-1]) == pytest.approx(4 * 48 + 12, abs=6)  # 5 dashes span 204 ft


def test_colored_blobs_rejected_as_non_paint():
    image = make_two_dashed_lines()
    # Green elongated blobs (foliage-like) alongside the white dashes.
    for y in (100, 220, 340):
        cv2.rectangle(image, (348, y), (352, y + DASH_LEN), (40, 200, 40), -1)
    dashes = detect_dashes(image, tophat_px=11, thresh=40)
    assert len(dashes) == 10  # only the white paint survives
    assert all(d.centroid[0] < 340 for d in dashes)


def test_chain_score_prefers_regular_spacing():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    regular = chain_dashes(dashes)[0]
    # An irregular chain: same dashes but with one dropped, breaking the rhythm.
    irregular = [regular[0], regular[1], regular[3], regular[4]]
    assert chain_score(regular) > chain_score(irregular)


def test_merge_joins_same_line_but_not_adjacent_line():
    dashes = detect_dashes(make_two_dashed_lines(), tophat_px=11, thresh=40)
    left, right = chain_dashes(dashes)
    # Fragments of ONE physical line merge back into it...
    frag_a, frag_b = left[:3], left[3:]
    merged = merge_by_curve([frag_a, frag_b])
    assert len(merged) == 1 and len(merged[0]) == 5
    # ...but two different (parallel, 160px apart) lines never merge.
    assert len(merge_by_curve([left, right])) == 2


def make_dash(cx, cy, angle_deg=90.0, length=30.0):
    """Synthetic dash stroke; angle_deg is the paint direction (90 = vertical)."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)  # 90 deg points up-frame
    half = length / 2.0
    a = (cx - dx * half, cy - dy * half)
    b = (cx + dx * half, cy + dy * half)
    p_start, p_end = (a, b) if a[1] > b[1] else (b, a)
    return DashBlob(
        centroid=(float(cx), float(cy)),
        p_start=p_start,
        p_end=p_end,
        length_px=length,
        contour=np.zeros((4, 1, 2), dtype=np.int32),
    )


def test_link_verdict_parallel_to_both_one_or_neither():
    below, above = make_dash(100, 600), make_dash(100, 480)
    assert link_verdict(below, above) == "good"  # step runs along both strokes
    # Sideways step between two vertical strokes: parallel to neither.
    assert link_verdict(make_dash(100, 600), make_dash(220, 600)) == "bad"
    # Step along one stroke but perpendicular to the other's paint.
    assert link_verdict(below, make_dash(100, 480, angle_deg=0)) == "suspicious"


def test_interloper_dash_ejected():
    line = [make_dash(100, y) for y in (600, 480, 360, 240, 120, 0)]
    interloper = make_dash(160, 310)  # off-line: steps to it parallel to no stroke
    chains, dropped = filter_chain_consistency([line[:3] + [interloper] + line[3:]])
    assert chains == [line]
    assert dropped == [interloper]


def test_wrong_end_dash_dropped():
    line = [make_dash(100, y) for y in (600, 480, 360, 240, 120)]
    perp_end = make_dash(100, 20, angle_deg=0)  # paint perpendicular to travel
    chains, dropped = filter_chain_consistency([line + [perp_end]])
    assert chains == [line]
    assert dropped == [perp_end]


def test_curved_chain_untouched():
    chain = []
    for y in range(600, -1, -120):
        x = 100 + 0.0008 * (600 - y) ** 2  # gentle rightward parabola
        tangent_deg = 90 - math.degrees(math.atan(2 * 0.0008 * (600 - y)))
        chain.append(make_dash(x, y, angle_deg=tangent_deg))
    chains, dropped = filter_chain_consistency([chain])
    assert chains == [chain]
    assert dropped == []


def test_glued_lines_split_at_bad_link():
    left = [make_dash(100, y) for y in (600, 480, 360)]
    right = [make_dash(300, y) for y in (300, 180, 60)]
    chains, dropped = filter_chain_consistency([left + right])
    assert chains == [left, right]  # sorted left-to-right by the filter
    assert dropped == []


def test_offset_dash_caught_by_turning_angle():
    # Links to the offset dash stay within axis tolerance (~18 deg), so only
    # the turning-angle check (37 deg spike vs a straight chain) can see it.
    line = [make_dash(100, y) for y in range(1080, 60, -120)]
    chain = list(line)
    offset = make_dash(140, 600)
    chain[4] = offset
    chains, dropped = filter_chain_consistency([chain])
    assert chains == [line[:4] + line[5:]]
    assert dropped == [offset]
