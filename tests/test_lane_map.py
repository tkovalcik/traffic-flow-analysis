"""Tests for lane-marking detection/grouping/calibration on synthetic imagery."""

import numpy as np
import pytest

from src.perception.speed.lane_map import (
    chain_dashes,
    detect_dashes,
    feet_along_curve,
    fit_lane_curve,
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
