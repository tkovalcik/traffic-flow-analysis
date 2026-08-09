"""Tests for the clip recorder: file output and metadata provenance."""

import json

import numpy as np
import pytest

from src.perception.cameras import CameraInfo
from src.replay.record import record_clip

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def camera_with_file_stream(tmp_path):
    """CameraInfo whose 'stream' is a 30-frame local file — recorder can't tell."""
    path = str(tmp_path / "stream.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
    writer.release()
    return CameraInfo(
        camera_id="tvtest",
        name="TVTEST -- I-00 : Nowhere",
        route="I-00",
        direction="East",
        county="Testing",
        place="Testville",
        latitude=37.0,
        longitude=-122.0,
        stream_url=path,
        static_url="",
        in_service=True,
        raw={"location": {"locationName": "TVTEST -- I-00 : Nowhere"}, "inService": "true"},
    )


def test_records_clip_with_metadata(camera_with_file_stream, tmp_path):
    clip = record_clip(camera_with_file_stream, seconds=2.0, out_dir=tmp_path / "out")

    assert clip.video_path.exists() and clip.video_path.stat().st_size > 0
    assert clip.frames == 20  # 2s @ 10 fps — frame bound, not the whole 30-frame file

    meta = json.loads(clip.metadata_path.read_text())
    assert meta["schema_version"] == 1
    assert meta["camera_id"] == "tvtest"
    assert meta["recorded"]["frames"] == 20
    assert meta["recorded"]["fps_writer"] == pytest.approx(10.0)
    assert meta["recorded"]["width"] == 64
    assert meta["recorded"]["first_frame_utc"] is not None
    # Full inventory record rides along for provenance
    assert meta["source"]["inventory_record"]["inService"] == "true"


def test_recorded_clip_is_readable_video(camera_with_file_stream, tmp_path):
    clip = record_clip(camera_with_file_stream, seconds=1.0, out_dir=tmp_path / "out")
    cap = cv2.VideoCapture(str(clip.video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (48, 64, 3)


def test_raises_when_stream_never_delivers(tmp_path, camera_with_file_stream):
    dead = camera_with_file_stream
    dead.stream_url = str(tmp_path / "missing.mp4")
    with pytest.raises(RuntimeError, match="no frames"):
        record_clip(dead, seconds=1.0, out_dir=tmp_path / "out", backoff_s=0.0)
