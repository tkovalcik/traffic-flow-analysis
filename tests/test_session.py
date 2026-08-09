"""Tests for the multi-camera capture session orchestrator."""

import json

import numpy as np
import pytest

from src.perception.camera_config import DEFAULT_CONFIG_PATH
from src.perception.cameras import CameraInfo
from src.replay.session import configured_camera_ids, run_session

cv2 = pytest.importorskip("cv2")


def make_file_camera(tmp_path, camera_id: str, frames: int = 25) -> CameraInfo:
    path = str(tmp_path / f"{camera_id}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 3, dtype=np.uint8))
    writer.release()
    return CameraInfo(
        camera_id=camera_id,
        name=f"{camera_id.upper()} -- I-00 : Test",
        route="I-00",
        direction="East",
        county="Test",
        place="Testville",
        latitude=37.0,
        longitude=-122.0,
        stream_url=path,
        static_url="",
        in_service=True,
        raw={"inService": "true"},
    )


def test_session_records_all_cameras_and_writes_manifest(tmp_path):
    cams = [make_file_camera(tmp_path, "cama"), make_file_camera(tmp_path, "camb")]
    manifest = run_session(cams, seconds=1.0, out_dir=tmp_path / "captures")

    assert len(manifest["clips"]) == 2
    assert all(clip["ok"] for clip in manifest["clips"])
    assert all(clip["frames"] == 10 for clip in manifest["clips"])  # 1s @ 10fps

    manifest_path = tmp_path / "captures" / manifest["session_id"] / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["session_id"] == manifest["session_id"]


def test_one_dead_camera_does_not_sink_the_session(tmp_path):
    good = make_file_camera(tmp_path, "good")
    dead = make_file_camera(tmp_path, "dead")
    dead.stream_url = str(tmp_path / "missing.mp4")

    manifest = run_session([good, dead], seconds=1.0, out_dir=tmp_path / "captures", backoff_s=0.0)
    by_id = {c["camera_id"]: c for c in manifest["clips"]}
    assert by_id["good"]["ok"] is True
    assert by_id["dead"]["ok"] is False
    assert "ERROR" in by_id["dead"]["error"]


def test_configured_camera_ids_reads_committed_config():
    ids = configured_camera_ids(DEFAULT_CONFIG_PATH)
    assert "tva43" in ids and "tv516" in ids
    assert all(not i.startswith("_") for i in ids)
