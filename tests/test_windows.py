"""Window semantics for the stream processor: alignment, watermarks, late events."""

from datetime import UTC, datetime, timedelta

from src.replay.producer import emission_schedule
from src.streaming.alerts import check_camera_staleness
from src.streaming.contracts import AlertType, TravelDirection, VehicleClass, VehicleEvent
from src.streaming.windows import (
    DEMO_WINDOW_SECONDS,
    ClosedWindow,
    TumblingWindows,
    to_volume_rows,
    window_bounds,
)

START = datetime(2026, 8, 9, 18, 46, 30, tzinfo=UTC)  # first event of the sample capture
W1_START = datetime(2026, 8, 9, 18, 45, tzinfo=UTC)
W1_END = datetime(2026, 8, 9, 19, 0, tzinfo=UTC)
W2_END = datetime(2026, 8, 9, 19, 15, tzinfo=UTC)


def at(
    offset_s: float,
    camera: str = "tva43",
    direction: TravelDirection = TravelDirection.EB,
    vehicle_class: VehicleClass = VehicleClass.car,
    track_id: int = 1,
) -> VehicleEvent:
    """One event offset_s seconds after the sample capture's first crossing."""
    ts = START + timedelta(seconds=offset_s)
    return VehicleEvent(
        camera_id=camera,
        ts_event=ts,
        ts_publish=ts,
        track_id=track_id,
        vehicle_class=vehicle_class,
        direction=direction,
        confidence=0.9,
    )


def test_windows_are_clock_aligned_not_first_event_aligned():
    # A capture starting at 18:46:30 still reports on the 18:45 grid, so tables
    # from different cameras and sessions line up row for row.
    assert window_bounds(START, 900) == (W1_START, W1_END)


def test_demo_windows_align_to_the_minute():
    assert window_bounds(START, DEMO_WINDOW_SECONDS) == (
        datetime(2026, 8, 9, 18, 46, tzinfo=UTC),
        datetime(2026, 8, 9, 18, 47, tzinfo=UTC),
    )


def test_window_stays_open_while_the_watermark_is_inside_it():
    windows = TumblingWindows()
    assert windows.add(at(0)) == []
    assert windows.add(at(700)) == []


def test_lateness_bound_holds_a_window_open_past_its_end():
    windows = TumblingWindows(lateness_seconds=30)
    windows.add(at(0))
    assert windows.add(at(820)) == [], "19:00:10 leaves the watermark at 18:59:40"
    closed = windows.add(at(850))
    assert [w.window_end for w in closed] == [W1_END]


def test_closed_window_carries_per_class_counts_and_a_total():
    windows = TumblingWindows()
    for track_id in range(3):
        windows.add(at(track_id, track_id=track_id))
    windows.add(at(4, vehicle_class=VehicleClass.truck, track_id=9))
    closed = windows.flush()
    assert len(closed) == 1
    assert closed[0].counts == {"car": 3, "truck": 1}
    assert closed[0].total == 4


def test_a_jump_closes_every_window_behind_the_watermark_in_order():
    # 19:00:10 opens the second window without closing the first (watermark
    # 18:59:40); the jump to 19:46:30 then closes both at once.
    windows = TumblingWindows()
    windows.add(at(0))
    windows.add(at(820))
    closed = windows.add(at(3600))
    assert [w.window_start for w in closed] == [W1_START, W1_END]


def test_windows_closing_together_come_back_in_a_deterministic_order():
    # Byte-identical replays depend on this ordering, not on dict insertion.
    windows = TumblingWindows()
    windows.add(at(0, direction=TravelDirection.WB))
    windows.add(at(1, direction=TravelDirection.EB))
    windows.add(at(820, direction=TravelDirection.WB))
    closed = windows.add(at(3600))
    assert [(w.window_start, w.direction) for w in closed] == [
        (W1_START, "EB"),
        (W1_START, "WB"),
        (W1_END, "WB"),
    ]


def test_a_silent_camera_does_not_block_another_cameras_windows():
    windows = TumblingWindows()
    windows.add(at(0, camera="tva43"))
    windows.add(at(0, camera="tv516"))
    closed = windows.add(at(1000, camera="tva43"))
    assert [w.camera_id for w in closed] == ["tva43"]


def test_flush_closes_the_trailing_window():
    # The sample capture ends 90s into its second window; with no later event to
    # advance the watermark, only an end-of-stream flush ever emits that tail.
    windows = TumblingWindows()
    windows.add(at(0))
    assert [w.window_start for w in windows.add(at(900))] == [W1_START]
    assert [w.window_start for w in windows.flush()] == [W1_END]


def test_flush_is_idempotent():
    windows = TumblingWindows()
    windows.add(at(0))
    assert len(windows.flush()) == 1
    assert windows.flush() == []


def test_out_of_order_event_still_counts_while_its_window_is_open():
    windows = TumblingWindows()
    windows.add(at(100, track_id=1))
    windows.add(at(50, track_id=2))
    assert windows.flush()[0].total == 2


def test_event_for_a_closed_window_is_dropped_and_counted():
    windows = TumblingWindows()
    windows.add(at(0))
    windows.add(at(850))
    assert windows.add(at(60)) == []
    assert windows.late_dropped == {"tva43": 1}


def test_a_late_drop_never_re_emits_the_closed_window():
    # append_volume_rows only appends and EwmaBaseline.observe has no inverse, so
    # a closed window is final: the drop must not produce a corrected second row.
    windows = TumblingWindows()
    windows.add(at(0))
    windows.add(at(850))
    windows.add(at(60))
    remaining = windows.flush()
    assert [w.window_start for w in remaining] == [W1_END]
    assert remaining[0].total == 1


def test_replay_lateness_injection_exercises_the_late_path():
    # emission_schedule defers events by 120s of event time, so the bound has to
    # sit under that to strand any of them — at 180s nothing is ever late.
    events = [at(i * 10, track_id=i) for i in range(120)]
    schedule = emission_schedule(events, late_fraction=0.5, late_seconds=120.0, seed=0)

    strict = TumblingWindows(lateness_seconds=30)
    lenient = TumblingWindows(lateness_seconds=180)
    for _offset, event in schedule:
        strict.add(event)
        lenient.add(event)
    strict.flush()
    lenient.flush()

    assert sum(strict.late_dropped.values()) > 0
    assert lenient.late_dropped == {}


def test_to_volume_rows_expands_a_window_into_sorted_per_class_rows():
    window = ClosedWindow(
        camera_id="tva43",
        direction="EB",
        window_start=W1_START,
        window_end=W1_END,
        counts={"truck": 2, "car": 5},
    )
    rows = to_volume_rows(window)
    assert [(row.vehicle_class, row.count) for row in rows] == [("car", 5), ("truck", 2)]
    assert all(row.window_start == W1_START and row.camera_id == "tva43" for row in rows)


def test_a_lone_camera_never_looks_stale():
    # stream_time advances with that camera's own events, so its gap is always 0.
    windows = TumblingWindows()
    windows.add(at(0))
    windows.add(at(600))
    last_seen = windows.last_event_at("tva43")
    assert check_camera_staleness("tva43", last_seen, windows.stream_time) is None


def test_a_silent_camera_goes_stale_against_stream_time():
    # Staleness runs on event time, not wall clock: at --speed 60 a real 5-minute
    # outage lasts 5 wall-clock seconds and STALE_AFTER would never be reached.
    windows = TumblingWindows()
    windows.add(at(0, camera="tv516"))
    windows.add(at(600, camera="tva43"))
    alert = check_camera_staleness("tv516", windows.last_event_at("tv516"), windows.stream_time)
    assert alert is not None
    assert alert.alert_type == AlertType.camera_stale
