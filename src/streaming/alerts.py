"""Alert rules over closed windows and camera liveness. Pure functions, no Kafka.

Rules:
- volume_spike / volume_drop: a window's count deviates from the EWMA baseline
  by more than a factor, after a warm-up number of windows.
- camera_stale: a camera produced no events for longer than a threshold —
  distinguishes "quiet road" (events with zero crossings still arrive... they
  don't; absence of events IS the signal) from "camera/pipeline down".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.streaming.baseline import BaselineStats
from src.streaming.contracts import AlertType, TrafficAlert

WARMUP_WINDOWS = 3
SPIKE_FACTOR = 2.0
DROP_FACTOR = 0.35
BASELINE_FLOOR = 3.0  # ignore spike/drop math on near-empty baselines
STALE_AFTER = timedelta(minutes=5)


def evaluate_window(
    camera_id: str,
    direction: str,
    window_start: datetime,
    window_end: datetime,
    count: int,
    baseline_before: BaselineStats,
    spike_factor: float = SPIKE_FACTOR,
    drop_factor: float = DROP_FACTOR,
    warmup_windows: int = WARMUP_WINDOWS,
) -> TrafficAlert | None:
    """Compare one closed window's count against its pre-observation baseline."""
    if baseline_before.windows_seen < warmup_windows:
        return None
    baseline = max(baseline_before.ewma, BASELINE_FLOOR)
    ratio = count / baseline
    if ratio >= spike_factor:
        alert_type, verb = AlertType.volume_spike, "above"
    elif ratio <= drop_factor:
        alert_type, verb = AlertType.volume_drop, "below"
    else:
        return None
    return TrafficAlert(
        alert_id=str(uuid4()),
        camera_id=camera_id,
        alert_type=alert_type,
        ts_alert=datetime.now(UTC),
        window_start=window_start,
        window_end=window_end,
        direction=direction,
        observed_count=count,
        baseline=round(baseline_before.ewma, 2),
        message=(
            f"{camera_id} {direction}: {count} vehicles/window is {ratio:.1f}x "
            f"{verb} EWMA baseline {baseline_before.ewma:.1f}"
        ),
    )


def check_camera_staleness(
    camera_id: str,
    last_event_at: datetime | None,
    now: datetime,
    stale_after: timedelta = STALE_AFTER,
) -> TrafficAlert | None:
    """camera_stale alert when a camera has been silent for too long."""
    if last_event_at is None or now - last_event_at < stale_after:
        return None
    silent_s = int((now - last_event_at).total_seconds())
    return TrafficAlert(
        alert_id=str(uuid4()),
        camera_id=camera_id,
        alert_type=AlertType.camera_stale,
        ts_alert=now,
        window_start=last_event_at,
        window_end=now,
        direction=None,
        observed_count=None,
        baseline=None,
        message=f"{camera_id}: no vehicle events for {silent_s}s",
    )
