"""Tests for the replay producer: pacing, lateness injection, ts_publish rewrite."""

import json
from datetime import UTC, datetime, timedelta

from src.replay.producer import emission_schedule, load_events, mirror_events, replay
from src.streaming.contracts import TravelDirection, VehicleClass, VehicleEvent
from src.streaming.producer import DeliveryStats

START = datetime(2026, 8, 9, 18, 46, 30, tzinfo=UTC)


class FakeProducer:
    """Collects published events instead of talking to a broker."""

    def __init__(self):
        self.published = []
        self.flushed = 0

    def publish(self, event):
        self.published.append(event)

    def flush(self, timeout=10.0):
        self.flushed += 1
        return DeliveryStats(delivered=len(self.published))


def make_event(offset_s: float, track_id: int = 1) -> VehicleEvent:
    return VehicleEvent(
        camera_id="tva43",
        ts_event=START + timedelta(seconds=offset_s),
        ts_publish=START + timedelta(seconds=offset_s),
        track_id=track_id,
        vehicle_class=VehicleClass.car,
        direction=TravelDirection.EB,
        confidence=0.9,
    )


def make_events(offsets: list[float]) -> list[VehicleEvent]:
    return [make_event(o, track_id=i) for i, o in enumerate(offsets)]


def test_load_events_parses_recorded_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    events = make_events([0, 30])
    path.write_text("\n".join(json.dumps(e.to_avro_dict()) for e in events) + "\n")
    loaded = load_events(path)
    assert [e.track_id for e in loaded] == [0, 1]
    assert loaded[0].ts_event == START


def test_load_events_honours_limit(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e.to_avro_dict()) for e in make_events([0, 1, 2])))
    assert len(load_events(path, limit=2)) == 2


def test_schedule_offsets_track_event_time():
    schedule = emission_schedule(make_events([0, 30, 90]))
    assert [offset for offset, _event in schedule] == [0.0, 30.0, 90.0]


def test_schedule_without_lateness_preserves_order():
    events = make_events([0, 30, 90])
    schedule = emission_schedule(events, late_fraction=0.0)
    assert [event.track_id for _offset, event in schedule] == [0, 1, 2]


def test_lateness_defers_events_past_their_window():
    events = make_events([0, 30, 60])
    schedule = emission_schedule(events, late_fraction=1.0, late_seconds=120.0)
    assert [offset for offset, _event in schedule] == [120.0, 150.0, 180.0]


def test_late_events_arrive_out_of_event_time_order():
    # Only the first event is held back, so it is emitted after later events.
    events = make_events([0, 30, 60, 90])
    schedule = emission_schedule(events, late_fraction=0.4, late_seconds=120.0, seed=1)
    emitted = [event.ts_event for _offset, event in schedule]
    assert emitted != sorted(emitted), "expected at least one out-of-order arrival"


def test_lateness_draw_is_deterministic_for_a_seed():
    events = make_events([0, 30, 60, 90])
    first = emission_schedule(events, late_fraction=0.5, seed=7)
    second = emission_schedule(events, late_fraction=0.5, seed=7)
    assert [e.track_id for _o, e in first] == [e.track_id for _o, e in second]


def test_empty_input_yields_empty_schedule():
    assert emission_schedule([]) == []


def test_mirror_copies_every_event_onto_the_new_camera():
    mirrored = mirror_events(make_events([0, 30, 60]), "tva43_mirror")
    assert [e.camera_id for e in mirrored] == ["tva43_mirror"] * 3


def test_mirror_keeps_event_time_so_copies_share_windows():
    events = make_events([0, 30, 60])
    mirrored = mirror_events(events, "tva43_mirror")
    assert [e.ts_event for e in mirrored] == [e.ts_event for e in events]


def test_mirror_gives_each_copy_a_fresh_event_id():
    events = make_events([0, 30])
    mirrored = mirror_events(events, "tva43_mirror")
    ids = {e.event_id for e in events} | {e.event_id for e in mirrored}
    assert len(ids) == 4


def test_drop_after_silences_the_mirror_mid_stream():
    mirrored = mirror_events(make_events([0, 30, 60, 90]), "tva43_mirror", drop_after_seconds=60)
    assert [(e.ts_event - START).total_seconds() for e in mirrored] == [0.0, 30.0, 60.0]


def test_drop_after_leaves_the_source_capture_untouched():
    events = make_events([0, 30, 60, 90])
    mirror_events(events, "tva43_mirror", drop_after_seconds=30)
    assert len(events) == 4 and all(e.camera_id == "tva43" for e in events)


def test_mirror_of_nothing_is_nothing():
    assert mirror_events([], "tva43_mirror") == []


def test_mirrored_stream_lets_the_dead_camera_fall_behind():
    # The point of the mirror: the surviving camera keeps advancing stream time,
    # so the silenced one accumulates a gap a lone camera could never show.
    events = make_events([0, 120, 240, 360])
    mirrored = mirror_events(events, "tva43_mirror", drop_after_seconds=60)
    combined = events + mirrored
    newest = max(e.ts_event for e in combined)
    last_mirror = max(e.ts_event for e in combined if e.camera_id == "tva43_mirror")
    assert (newest - last_mirror).total_seconds() == 360.0


def test_replay_publishes_every_event_and_flushes():
    producer = FakeProducer()
    stats = replay(emission_schedule(make_events([0, 30])), producer, speed=0, sleeper=_no_sleep)
    assert len(producer.published) == 2
    assert producer.flushed == 1
    assert stats.delivered == 2


def test_replay_rewrites_ts_publish_but_keeps_ts_event():
    producer = FakeProducer()
    original = make_event(0)
    replay([(0.0, original)], producer, speed=0, sleeper=_no_sleep)
    sent = producer.published[0]
    assert sent.ts_event == original.ts_event
    assert sent.ts_publish > original.ts_publish


def test_speed_compresses_event_time_gaps():
    slept = []
    producer = FakeProducer()
    replay(emission_schedule(make_events([0, 60, 120])), producer, speed=60, sleeper=slept.append)
    assert slept == [1.0, 1.0]


def test_zero_speed_never_sleeps():
    slept = []
    producer = FakeProducer()
    replay(emission_schedule(make_events([0, 600])), producer, speed=0, sleeper=slept.append)
    assert slept == []


def _no_sleep(_seconds: float) -> None:
    pass
