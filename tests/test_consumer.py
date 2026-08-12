"""Tests for the stream processor: decode, bounded loop, and window writing."""

from datetime import UTC, datetime, timedelta

import pytest

from src.streaming.baseline import BaselineStats, EwmaBaseline
from src.streaming.consumer import (
    DEFAULT_ALERTS_TOPIC,
    WindowWriter,
    build_alert_producer,
    consume,
    consumer_config,
    decode_message,
)
from src.streaming.contracts import (
    AlertType,
    TrafficAlert,
    TravelDirection,
    VehicleClass,
    VehicleEvent,
)
from src.streaming.outputs import read_alerts
from src.streaming.windows import TumblingWindows

START = datetime(2026, 8, 9, 18, 46, 30, tzinfo=UTC)

LOCAL_ENV = {"KAFKA_BOOTSTRAP_SERVERS": "localhost:9092"}


class FakeMessage:
    """Stands in for a confluent_kafka Message over an already-decoded event."""

    def __init__(self, event, key=..., error=None, offset=0):
        self._event = event
        self._key = event.camera_id.encode("utf-8") if key is ... else key
        self._error = error
        self._offset = offset

    def value(self):
        return self._event

    def key(self):
        return self._key

    def error(self):
        return self._error

    def topic(self):
        return "vehicle.events"

    def partition(self):
        return 0

    def offset(self):
        return self._offset


class FakeConsumer:
    """Replays a fixed list of messages, then returns None like a quiet broker."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.commits = []
        self.closed = False

    def poll(self, _timeout):
        return self._messages.pop(0) if self._messages else None

    def commit(self, message, asynchronous):
        self.commits.append(message.offset())

    def close(self):
        self.closed = True


class FakeKafkaProducer:
    """Stands in for confluent_kafka.Producer inside a real EventProducer."""

    def __init__(self, sent):
        self._sent = sent

    def produce(self, topic, key, value, on_delivery):
        self._sent.append({"topic": topic, "key": key})
        on_delivery(None, None)

    def poll(self, _timeout):
        pass

    def flush(self, _timeout):
        return 0


class FakeAlertProducer:
    """Collects alerts instead of producing them to traffic.alerts."""

    def __init__(self):
        self.topic = DEFAULT_ALERTS_TOPIC
        self.published = []

    def publish(self, alert):
        self.published.append(alert)


def echo_deserializer(payload, _ctx):
    return payload


def make_event(offset_s=0.0, camera="tva43", direction=TravelDirection.EB, track_id=1):
    return VehicleEvent(
        camera_id=camera,
        ts_event=START + timedelta(seconds=offset_s),
        ts_publish=START + timedelta(seconds=offset_s),
        track_id=track_id,
        vehicle_class=VehicleClass.car,
        direction=direction,
        confidence=0.9,
    )


def test_consumer_config_drops_producer_only_settings():
    config = consumer_config("tfa-test", LOCAL_ENV)
    assert "enable.idempotence" not in config
    assert config["group.id"] == "tfa-test"
    assert config["enable.auto.commit"] is False
    assert config["auto.offset.reset"] == "earliest"


def test_decode_message_returns_the_event():
    event = make_event()
    assert decode_message(FakeMessage(event), echo_deserializer) == event


def test_decode_message_rejects_a_key_that_is_not_the_camera():
    # The key is what puts a camera's events in one partition, in order; a
    # mismatch means the partitioning claim in the report is not true.
    message = FakeMessage(make_event(), key=b"other-camera")
    with pytest.raises(ValueError, match="does not match camera_id"):
        decode_message(message, echo_deserializer)


def test_decode_message_rejects_an_unkeyed_message():
    with pytest.raises(ValueError, match="unkeyed"):
        decode_message(FakeMessage(make_event(), key=None), echo_deserializer)


def test_consume_stops_when_the_stream_goes_quiet():
    consumer = FakeConsumer([FakeMessage(make_event(i), offset=i) for i in range(3)])
    stats = consume(
        consumer,
        echo_deserializer,
        TumblingWindows(),
        lambda _closed: [],
        idle_timeout=0.05,
        poll_timeout=0.0,
    )
    assert stats.consumed == 3
    assert stats.stop_reason == "idle_timeout"


def test_consume_honours_max_messages():
    consumer = FakeConsumer([FakeMessage(make_event(i), offset=i) for i in range(10)])
    stats = consume(
        consumer,
        echo_deserializer,
        TumblingWindows(),
        lambda _closed: [],
        max_messages=4,
        poll_timeout=0.0,
    )
    assert stats.consumed == 4
    assert stats.stop_reason == "max_messages"


def test_consume_commits_every_processed_message():
    consumer = FakeConsumer([FakeMessage(make_event(i), offset=i) for i in range(3)])
    consume(
        consumer,
        echo_deserializer,
        TumblingWindows(),
        lambda _closed: [],
        max_messages=3,
        poll_timeout=0.0,
    )
    assert consumer.commits == [0, 1, 2]


def test_consume_flushes_the_trailing_window_after_the_loop():
    # Nothing pushes a watermark past the last window, so without the flush the
    # run would end having reported no windows at all.
    batches = []
    consumer = FakeConsumer([FakeMessage(make_event(i), offset=i) for i in range(3)])
    stats = consume(
        consumer,
        echo_deserializer,
        TumblingWindows(),
        batches.append,
        idle_timeout=0.05,
        poll_timeout=0.0,
    )
    assert stats.windows_closed == 1
    assert [w.total for batch in batches for w in batch] == [3]


def test_consume_reports_dropped_late_events():
    events = [make_event(0), make_event(850), make_event(60)]
    consumer = FakeConsumer([FakeMessage(e, offset=i) for i, e in enumerate(events)])
    stats = consume(
        consumer,
        echo_deserializer,
        TumblingWindows(),
        lambda _closed: [],
        idle_timeout=0.05,
        poll_timeout=0.0,
    )
    assert stats.late_dropped == 1


def test_window_writer_appends_rows_and_returns_alerts(tmp_path):
    csv_path = tmp_path / "volume.csv"
    writer = WindowWriter(csv_path, tmp_path / "alerts.jsonl")
    windows = TumblingWindows(window_seconds=60)
    closed = []
    for offset in range(0, 300, 60):
        closed.extend(windows.add(make_event(offset)))
    closed.extend(windows.flush())
    writer.handle(closed, windows)
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("window_start")
    assert len(lines) == len(closed) + 1


def test_window_writer_writes_alerts_the_rules_raise(tmp_path):
    alerts_path = tmp_path / "alerts.jsonl"
    writer = WindowWriter(tmp_path / "volume.csv", alerts_path)
    # Pre-warm the baseline past WARMUP_WINDOWS so one loaded window spikes.
    writer.baseline = EwmaBaseline()
    for _ in range(4):
        writer.baseline.observe("tva43", "EB", 10)
    windows = TumblingWindows(window_seconds=60)
    for track_id in range(60):
        windows.add(make_event(0, track_id=track_id))
    alerts = writer.handle(windows.flush(), windows)
    assert [a.alert_type for a in alerts] == [AlertType.volume_spike]
    assert read_alerts(alerts_path)[0].observed_count == 60


def test_window_writer_raises_a_stale_alert_only_once(tmp_path):
    # tv516 falls silent while tva43 keeps producing, so stream_time runs away
    # from it; the alert must not repeat on every later batch.
    writer = WindowWriter(tmp_path / "volume.csv", tmp_path / "alerts.jsonl")
    windows = TumblingWindows(window_seconds=60)
    windows.add(make_event(0, camera="tv516"))
    closed = windows.add(make_event(600, camera="tva43"))
    first = writer.handle(closed, windows)
    second = writer.handle(windows.add(make_event(660, camera="tva43")), windows)
    assert [a.alert_type for a in first] == [AlertType.camera_stale]
    assert second == []


def test_alerts_are_published_to_the_topic_as_well_as_logged(tmp_path):
    producer = FakeAlertProducer()
    writer = WindowWriter(
        tmp_path / "volume.csv", tmp_path / "alerts.jsonl", alert_producer=producer
    )
    for _ in range(4):
        writer.baseline.observe("tva43", "EB", 10)
    windows = TumblingWindows(window_seconds=60)
    for track_id in range(60):
        windows.add(make_event(0, track_id=track_id))
    alerts = writer.handle(windows.flush(), windows)
    assert [a.alert_id for a in producer.published] == [a.alert_id for a in alerts]


def test_quiet_windows_publish_nothing(tmp_path):
    producer = FakeAlertProducer()
    writer = WindowWriter(
        tmp_path / "volume.csv", tmp_path / "alerts.jsonl", alert_producer=producer
    )
    windows = TumblingWindows(window_seconds=60)
    windows.add(make_event(0))
    writer.handle(windows.flush(), windows)
    assert producer.published == []


def test_published_alerts_survive_the_avro_round_trip(tmp_path):
    # The alert contract has to serialize as well as validate: direction and
    # observed_count are nullable unions that camera_stale leaves empty.
    producer = FakeAlertProducer()
    writer = WindowWriter(
        tmp_path / "volume.csv", tmp_path / "alerts.jsonl", alert_producer=producer
    )
    windows = TumblingWindows(window_seconds=60)
    windows.add(make_event(0, camera="tv516"))
    writer.handle(windows.add(make_event(600, camera="tva43")), windows)
    restored = TrafficAlert.from_avro_dict(producer.published[0].to_avro_dict())
    assert restored.alert_type == AlertType.camera_stale
    assert restored.direction is None
    assert restored.camera_id == "tv516"


def test_alert_producer_keys_alerts_by_camera_and_targets_the_alerts_topic():
    # Same key as vehicle.events, so an alert lands on the partition carrying
    # the camera it came from.
    sent = []
    producer = build_alert_producer(
        {**LOCAL_ENV, "TOPIC_TRAFFIC_ALERTS": "traffic.alerts.test"},
        producer_factory=lambda _config: FakeKafkaProducer(sent),
        serializer=lambda alert, _ctx: b"payload",
    )
    alert = TrafficAlert(
        camera_id="tva43",
        alert_type=AlertType.volume_spike,
        ts_alert=START,
        window_start=START,
        window_end=START + timedelta(minutes=1),
        message="spike",
    )
    producer.publish(alert)
    assert producer.topic == "traffic.alerts.test"
    assert sent == [{"topic": "traffic.alerts.test", "key": b"tva43"}]


def test_baseline_sees_window_totals_not_per_class_rows(tmp_path):
    # evaluate_window compares a whole window's traffic against history; feeding
    # it per-class rows would divide the baseline by the class mix.
    writer = WindowWriter(tmp_path / "volume.csv", tmp_path / "alerts.jsonl")
    windows = TumblingWindows(window_seconds=60)
    for track_id in range(5):
        windows.add(make_event(0, track_id=track_id))
    writer.handle(windows.flush(), windows)
    assert writer.baseline.stats_for("tva43", "EB") == BaselineStats(ewma=5.0, windows_seen=1)
