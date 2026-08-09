"""Tests for per-camera line configuration."""

import json
from pathlib import Path

from src.perception.camera_config import DEFAULT_CONFIG_PATH, load_camera_lines, resolve_lines
from src.streaming.contracts import TravelDirection


def test_committed_config_parses_for_validation_camera():
    lines = load_camera_lines("tva43", DEFAULT_CONFIG_PATH)
    assert lines is not None and len(lines) == 2  # one motion-calibrated line per flow
    for line in lines:
        assert {line.positive_direction, line.negative_direction} == {
            TravelDirection.EB,
            TravelDirection.WB,
        }
        for x, y in (line.p1, line.p2):
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_unknown_camera_returns_none():
    assert load_camera_lines("tv000", DEFAULT_CONFIG_PATH) is None


def test_resolution_order(tmp_path):
    config = tmp_path / "cameras.json"
    config.write_text(json.dumps({"cam1": {"lines": ["0.1,0.3,0.9,0.3:NB:SB"]}}))
    fallback = "0.05,0.55,0.95,0.55:EB:WB"

    cli = resolve_lines("cam1", ["0.2,0.4,0.8,0.4:SB:NB"], fallback, config)
    assert cli[0].positive_direction == TravelDirection.SB  # CLI wins

    configured = resolve_lines("cam1", None, fallback, config)
    assert configured[0].positive_direction == TravelDirection.NB  # config next

    fell_back = resolve_lines("cam2", None, fallback, config)
    assert fell_back[0].positive_direction == TravelDirection.EB  # fallback last


def test_missing_config_file_falls_back(tmp_path):
    lines = resolve_lines("tva43", None, "0.0,0.5,1.0,0.5:EB:WB", tmp_path / "nope.json")
    assert lines[0].positive_direction == TravelDirection.EB
    assert load_camera_lines("tva43", Path(tmp_path / "nope.json")) is None
