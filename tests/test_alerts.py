"""Tests for alert rules: spike/drop gating and camera staleness."""

from datetime import UTC, datetime, timedelta

from src.streaming.alerts import check_camera_staleness, evaluate_window
from src.streaming.baseline import BaselineStats
from src.streaming.contracts import AlertType

W_START = datetime(2026, 8, 9, 15, 0, tzinfo=UTC)
W_END = datetime(2026, 8, 9, 15, 15, tzinfo=UTC)


def window_alert(count, baseline: BaselineStats):
    return evaluate_window("cam1", "EB", W_START, W_END, count, baseline)


def test_no_alert_during_warmup():
    assert window_alert(500, BaselineStats(ewma=10.0, windows_seen=2)) is None


def test_spike_alert_fires():
    alert = window_alert(50, BaselineStats(ewma=20.0, windows_seen=5))
    assert alert is not None
    assert alert.alert_type == AlertType.volume_spike
    assert alert.observed_count == 50
    assert alert.baseline == 20.0
    assert alert.direction == "EB"
    assert "2.5x" in alert.message


def test_drop_alert_fires():
    alert = window_alert(5, BaselineStats(ewma=40.0, windows_seen=5))
    assert alert is not None
    assert alert.alert_type == AlertType.volume_drop


def test_normal_band_is_quiet():
    assert window_alert(25, BaselineStats(ewma=20.0, windows_seen=5)) is None


def test_tiny_baselines_do_not_spike():
    # 2 vehicles vs EWMA 0.5 is a 4x ratio but meaningless on an empty road;
    # the baseline floor keeps it quiet.
    assert window_alert(2, BaselineStats(ewma=0.5, windows_seen=5)) is None


def test_staleness_alert():
    now = datetime(2026, 8, 9, 16, 0, tzinfo=UTC)
    fresh = check_camera_staleness("cam1", now - timedelta(minutes=2), now)
    assert fresh is None
    stale = check_camera_staleness("cam1", now - timedelta(minutes=9), now)
    assert stale is not None
    assert stale.alert_type == AlertType.camera_stale
    assert stale.direction is None
    assert "540s" in stale.message


def test_staleness_with_no_events_ever_is_quiet():
    # A camera that never produced anything has no last-seen to compare against;
    # bootstrap liveness is the processor's job, not this rule's.
    now = datetime(2026, 8, 9, 16, 0, tzinfo=UTC)
    assert check_camera_staleness("cam1", None, now) is None
