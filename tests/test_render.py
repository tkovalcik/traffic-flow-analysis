"""Tests for the annotated-video renderer (no YOLO — scripted observations)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from src.perception.crossing import parse_line_spec
from src.perception.detect_track import TrackObservation
from src.perception.render import annotate_stream
from src.streaming.contracts import VehicleClass

cv2 = pytest.importorskip("cv2")

LINE = parse_line_spec("0.05,0.55,0.95,0.55:EB:WB")
ANCHOR = datetime(2026, 8, 9, 8, 39, 0, tzinfo=UTC)


def scripted_frames(n=12, size=(96, 64)):
    """One track walking down across the line; frame 7+ has no detections."""
    w, h = size
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        observations = []
        if i < 7:
            cy = 0.2 + i * 0.1  # crosses 0.55 between frames 3 and 4
            observations = [
                TrackObservation(
                    frame_index=i,
                    track_id=1,
                    vehicle_class=VehicleClass.car,
                    confidence=0.9,
                    center=(0.5, cy),
                    box=(0.45, cy - 0.05, 0.55, cy + 0.05),
                )
            ]
        yield frame, i, observations


def test_renders_all_frames_and_counts_crossing(tmp_path):
    out = tmp_path / "annotated.mp4"
    result = annotate_stream(scripted_frames(), [LINE], fps=10.0, anchor=ANCHOR, out_path=out)

    assert result.frames == 12
    assert result.counts == {"EB": 1}
    assert out.exists() and out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (64, 96, 3)


def test_render_with_no_detections_still_writes_video(tmp_path):
    def empty_frames():
        for i in range(5):
            yield np.zeros((48, 64, 3), dtype=np.uint8), i, []

    out = tmp_path / "empty.mp4"
    result = annotate_stream(empty_frames(), [LINE], fps=10.0, anchor=ANCHOR, out_path=out)
    assert result.frames == 5
    assert result.counts == {}


def test_render_empty_input_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no frames"):
        annotate_stream(iter(()), [LINE], fps=10.0, anchor=ANCHOR, out_path=tmp_path / "x.mp4")


@pytest.mark.parametrize("corner", ["tl", "tr", "bl", "br"])
def test_banner_corners_render(tmp_path, corner):
    out = tmp_path / f"banner_{corner}.mp4"
    result = annotate_stream(
        scripted_frames(3), [LINE], fps=10.0, anchor=ANCHOR, out_path=out, banner_corner=corner
    )
    assert result.frames == 3


def test_no_banner_mode(tmp_path):
    out = tmp_path / "clean.mp4"
    result = annotate_stream(
        scripted_frames(), [LINE], fps=10.0, anchor=ANCHOR, out_path=out, banner=False
    )
    assert result.frames == 12
    assert result.counts == {"EB": 1}  # counting still works, just not displayed
