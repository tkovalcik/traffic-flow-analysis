"""Tests for the speed-calibration prototypes: median frame and dash ruler."""

import numpy as np
import pytest

from src.perception.speed.dash_ruler import find_dashes, render_ruler
from src.perception.speed.median_frame import median_frame

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def traffic_video(tmp_path):
    """Static gray road + a bright 'vehicle' square moving across 30 frames."""
    path = tmp_path / "traffic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    for i in range(30):
        frame = np.full((64, 96, 3), 90, dtype=np.uint8)
        x = 3 * i
        frame[24:40, x : x + 12] = 250  # the moving vehicle
        writer.write(frame)
    writer.release()
    return path


def test_median_frame_removes_moving_vehicle(traffic_video):
    result = median_frame(traffic_video, samples=30)
    assert result.shape == (64, 96, 3)
    # Everywhere the vehicle passed through, the median must be pavement.
    assert int(result[30, 40, 0]) < 130
    assert abs(int(result[5, 5, 0]) - 90) < 12  # background preserved (codec noise aside)


def make_dash_image(starts, dash_len=30, y=100):
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    for x in starts:
        cv2.rectangle(image, (x, y - 2), (x + dash_len, y + 2), (255, 255, 255), -1)
    return image


def test_find_dashes_locates_painted_runs():
    starts = [20, 120, 220, 320]
    image = make_dash_image(starts)
    guide = [(0.0, 0.5), (1.0, 0.5)]  # straight through the dashes
    dashes = find_dashes(image, guide, window_px=4, thresh_k=0.3)
    assert len(dashes) == 4
    for dash, expected_x in zip(dashes, starts, strict=True):
        assert dash.start_xy[0] == pytest.approx(expected_x, abs=4)
        assert dash.length_px == pytest.approx(30, abs=5)


def test_find_dashes_cycles_are_uniform_for_straight_line():
    image = make_dash_image([20, 120, 220, 320])
    dashes = find_dashes(image, [(0.0, 0.5), (1.0, 0.5)], window_px=4, thresh_k=0.3)
    cycles = [b.start_arc - a.start_arc for a, b in zip(dashes, dashes[1:], strict=False)]
    assert all(c == pytest.approx(100, abs=5) for c in cycles)


def test_render_ruler_annotates_without_mutating_input():
    image = make_dash_image([20, 120, 220])
    dashes = find_dashes(image, [(0.0, 0.5), (1.0, 0.5)], window_px=4, thresh_k=0.3)
    before = image.copy()
    out, table = render_ruler(image, dashes, cycle_ft=48.0, dash_ft=12.0)
    assert out.shape == image.shape
    assert np.array_equal(image, before)  # copy, not in-place
    assert len(table) == len(dashes) - 1
    assert "ft/px" in table[0]
