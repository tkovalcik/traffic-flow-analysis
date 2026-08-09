"""Tests for volume-table CSV and alerts JSONL writers."""

from datetime import UTC, datetime

from src.streaming.contracts import AlertType, TrafficAlert
from src.streaming.outputs import VolumeRow, append_alerts, append_volume_rows, read_alerts

W_START = datetime(2026, 8, 9, 15, 0, tzinfo=UTC)
W_END = datetime(2026, 8, 9, 15, 15, tzinfo=UTC)


def make_row(count=7, direction="EB", vehicle_class="car"):
    return VolumeRow(
        window_start=W_START,
        window_end=W_END,
        camera_id="tva43",
        direction=direction,
        vehicle_class=vehicle_class,
        count=count,
    )


def test_volume_csv_header_and_rows(tmp_path):
    path = tmp_path / "volumes.csv"
    append_volume_rows(path, [make_row(), make_row(count=2, vehicle_class="truck")])
    lines = path.read_text().splitlines()
    assert lines[0] == "window_start,window_end,camera_id,direction,vehicle_class,count"
    assert lines[1] == "2026-08-09T15:00:00+00:00,2026-08-09T15:15:00+00:00,tva43,EB,car,7"
    assert len(lines) == 3


def test_volume_csv_append_does_not_duplicate_header(tmp_path):
    path = tmp_path / "volumes.csv"
    append_volume_rows(path, [make_row()])
    append_volume_rows(path, [make_row(count=9)])
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert lines.count("window_start,window_end,camera_id,direction,vehicle_class,count") == 1


def test_alerts_jsonl_roundtrip(tmp_path):
    path = tmp_path / "alerts.jsonl"
    alert = TrafficAlert(
        camera_id="tva43",
        alert_type=AlertType.volume_spike,
        ts_alert=W_END,
        window_start=W_START,
        window_end=W_END,
        direction="EB",
        observed_count=50,
        baseline=20.0,
        message="test spike",
    )
    append_alerts(path, [alert])
    restored = read_alerts(path)
    assert restored == [alert]


def test_read_alerts_missing_file_is_empty(tmp_path):
    assert read_alerts(tmp_path / "nope.jsonl") == []
