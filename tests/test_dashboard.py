"""Tests for the minimal dashboard endpoints."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import camera_health, create_app
from src.streaming.contracts import AlertType, TrafficAlert
from src.streaming.outputs import VolumeRow, append_alerts, append_volume_rows

# Firmly in the past so wall-clock-based health always reads "quiet" here;
# the "ok" case is covered via camera_health() with an injected now.
W_START = datetime(2026, 8, 1, 15, 0, tzinfo=UTC)
W_END = datetime(2026, 8, 1, 15, 15, tzinfo=UTC)


@pytest.fixture
def client(tmp_path):
    volumes = tmp_path / "volumes.csv"
    alerts = tmp_path / "alerts.jsonl"
    append_volume_rows(
        volumes,
        [
            VolumeRow(W_START, W_END, "tva43", "EB", "car", 12),
            VolumeRow(W_START, W_END, "tva43", "WB", "car", 8),
        ],
    )
    append_alerts(
        alerts,
        [
            TrafficAlert(
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
        ],
    )
    return TestClient(create_app(volumes, alerts))


def test_volumes_endpoint_newest_first(client):
    rows = client.get("/api/volumes").json()
    assert len(rows) == 2
    assert rows[0]["direction"] == "WB"  # appended last → served first
    assert rows[0]["count"] == "8"


def test_alerts_endpoint(client):
    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "volume_spike"
    assert alerts[0]["camera_id"] == "tva43"


def test_health_endpoint_flags_old_windows(client):
    health = client.get("/api/health").json()
    assert health == [
        {"camera_id": "tva43", "last_window_end": W_END.isoformat(), "status": "quiet"}
    ]  # window ended long before "now"


def test_health_marks_fresh_cameras_ok():
    rows = [{"camera_id": "tva43", "window_end": W_END.isoformat()}]
    fresh_now = datetime(2026, 8, 1, 15, 20, tzinfo=UTC)
    assert camera_health(rows, now=fresh_now)[0]["status"] == "ok"


def test_index_renders(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Traffic Flow" in page.text
    assert "tva43" in page.text


def test_empty_files_render_gracefully(tmp_path):
    empty_client = TestClient(create_app(tmp_path / "v.csv", tmp_path / "a.jsonl"))
    assert empty_client.get("/api/volumes").json() == []
    assert empty_client.get("/api/alerts").json() == []
    assert empty_client.get("/").status_code == 200
