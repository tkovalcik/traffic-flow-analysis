"""Stream processor for `vehicle.events`: event-time windows → volume table + alerts.

Polls the topic, folds each event into clock-aligned tumbling windows, and on
every window close updates the EWMA baseline, evaluates the alert rules, and
appends to the two files the dashboard reads. The run is bounded so the reviewer
demo terminates on its own: it stops once the replay goes quiet and flushes the
trailing window on the way out. Offsets are committed manually, after the batch
they contributed to has been written.

Usage:
    uv run python -m src.streaming.consumer --from-beginning --reset-outputs
    uv run python -m src.streaming.consumer --window-seconds 60 --idle-timeout 5
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.streaming.alerts import check_camera_staleness, evaluate_window
from src.streaming.baseline import EwmaBaseline
from src.streaming.contracts import TrafficAlert, VehicleEvent, load_avro_schema_str
from src.streaming.outputs import VolumeRow, append_alerts, append_volume_rows
from src.streaming.producer import (
    DEFAULT_TOPIC,
    EventProducer,
    build_avro_serializer,
    kafka_config,
    schema_registry_config,
)
from src.streaming.windows import (
    DEFAULT_LATENESS_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    ClosedWindow,
    TumblingWindows,
    to_volume_rows,
)

DEFAULT_GROUP_ID = "tfa-stream-processor"
DEFAULT_ALERTS_TOPIC = "traffic.alerts"
DEFAULT_VOLUME_CSV = Path("outputs/volume_15min.csv")
DEFAULT_ALERTS_JSONL = Path("outputs/alerts.jsonl")
DEFAULT_POLL_TIMEOUT = 0.5
DEFAULT_IDLE_TIMEOUT = 10.0
DEFAULT_RUN_TIMEOUT = 600.0


def consumer_config(group_id: str, env: Mapping[str, str] | None = None) -> dict:
    """Consumer config from the environment, sharing the producer's broker settings."""
    env = os.environ if env is None else env
    config = kafka_config(env)
    # Producer-only settings; the consumer inherits the connection, not the acks.
    for key in ("enable.idempotence", "acks"):
        config.pop(key, None)
    config.update(
        {
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            # Offsets are committed by hand once a batch is on disk, so a crash
            # replays the events whose windows never got written.
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
        }
    )
    return config


def build_avro_deserializer(env: Mapping[str, str] | None = None) -> Callable:
    """AvroDeserializer for vehicle_event; imported lazily so tests need no registry."""
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroDeserializer

    client = SchemaRegistryClient(schema_registry_config(env))
    return AvroDeserializer(
        client,
        load_avro_schema_str("vehicle_event"),
        lambda payload, _ctx: VehicleEvent.from_avro_dict(payload),
    )


def build_alert_producer(
    env: Mapping[str, str] | None = None,
    producer_factory: Callable[[dict], object] | None = None,
    serializer: Callable | None = None,
) -> EventProducer:
    """EventProducer aimed at `traffic.alerts` with the alert contract's schema."""
    env = os.environ if env is None else env
    return EventProducer(
        topic=env.get("TOPIC_TRAFFIC_ALERTS", DEFAULT_ALERTS_TOPIC),
        env=env,
        producer_factory=producer_factory,
        serializer=serializer or build_avro_serializer(env, "traffic_alert"),
    )


def decode_message(message: object, deserializer: Callable) -> VehicleEvent:
    """Deserialize one Kafka message and check its key against the event."""
    from confluent_kafka.serialization import MessageField, SerializationContext

    payload = message.value()
    if payload is None:
        raise ValueError("vehicle.events message has no value")
    event = deserializer(payload, SerializationContext(message.topic(), MessageField.VALUE))
    if not isinstance(event, VehicleEvent):
        event = VehicleEvent.from_avro_dict(event)
    key = message.key()
    if key is None:
        raise ValueError(f"{event.event_id} arrived unkeyed; camera partitioning is broken")
    if key.decode("utf-8") != event.camera_id:
        raise ValueError(f"message key {key!r} does not match camera_id {event.camera_id!r}")
    return event


@dataclass
class ConsumeStats:
    """What one bounded processing run did — the summary the demo prints."""

    consumed: int = 0
    windows_closed: int = 0
    alerts: int = 0
    late_dropped: int = 0
    stop_reason: str = ""


@dataclass
class WindowWriter:
    """Turns closed windows into volume rows and alerts, then appends and publishes.

    alert_producer is optional so the window path is testable without a broker;
    when absent, alerts land in the JSONL only and never reach `traffic.alerts`.
    """

    volume_csv: Path = DEFAULT_VOLUME_CSV
    alerts_jsonl: Path = DEFAULT_ALERTS_JSONL
    baseline: EwmaBaseline = field(default_factory=EwmaBaseline)
    alert_producer: EventProducer | None = None
    _stale_cameras: set[str] = field(default_factory=set)

    def handle(self, closed: list[ClosedWindow], windows: TumblingWindows) -> list[TrafficAlert]:
        """Write one batch of closed windows and return the alerts it raised."""
        rows: list[VolumeRow] = []
        alerts = []
        for window in closed:
            rows.extend(to_volume_rows(window))
            before = self.baseline.observe(window.camera_id, window.direction, window.total)
            alert = evaluate_window(
                window.camera_id,
                window.direction,
                window.window_start,
                window.window_end,
                window.total,
                before,
            )
            if alert is not None:
                alerts.append(alert)
        alerts.extend(self._staleness_alerts(windows))
        append_volume_rows(self.volume_csv, rows)
        if alerts:
            append_alerts(self.alerts_jsonl, alerts)
            if self.alert_producer is not None:
                for alert in alerts:
                    self.alert_producer.publish(alert)
        return alerts

    def _staleness_alerts(self, windows: TumblingWindows) -> list[TrafficAlert]:
        """Liveness on event time, so a 60x replay still reaches STALE_AFTER."""
        now = windows.stream_time
        if now is None:
            return []
        alerts = []
        for camera_id in windows.cameras():
            if camera_id in self._stale_cameras:
                continue
            alert = check_camera_staleness(camera_id, windows.last_event_at(camera_id), now)
            if alert is not None:
                self._stale_cameras.add(camera_id)
                alerts.append(alert)
        return alerts


def consume(
    consumer: object,
    deserializer: Callable,
    windows: TumblingWindows,
    on_closed: Callable[[list[ClosedWindow]], object],
    max_messages: int | None = None,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
) -> ConsumeStats:
    """Poll, window, and hand every closed batch to on_closed until the run ends.

    Bounded three ways — message count, silence, and total runtime — so the
    reviewer demo always exits; whichever fires lands in stats.stop_reason.
    """
    from confluent_kafka import KafkaError, KafkaException

    stats = ConsumeStats()
    started = time.monotonic()
    idle_deadline = started + idle_timeout
    while True:
        now = time.monotonic()
        if now - started > run_timeout:
            stats.stop_reason = "run_timeout"
            break
        if now > idle_deadline:
            stats.stop_reason = "idle_timeout"
            break

        message = consumer.poll(poll_timeout)
        if message is None:
            continue
        error = message.error()
        if error is not None:
            if error.code() == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(error)

        idle_deadline = time.monotonic() + idle_timeout
        closed = windows.add(decode_message(message, deserializer))
        stats.consumed += 1
        if closed:
            stats.windows_closed += len(closed)
            stats.alerts += len(on_closed(closed) or [])
        consumer.commit(message=message, asynchronous=False)

        if max_messages is not None and stats.consumed >= max_messages:
            stats.stop_reason = "max_messages"
            break

    trailing = windows.flush()
    if trailing:
        stats.windows_closed += len(trailing)
        stats.alerts += len(on_closed(trailing) or [])
    stats.late_dropped = sum(windows.late_dropped.values())
    return stats


def _default_consumer_factory(config: dict):
    from confluent_kafka import Consumer

    return Consumer(config)


def _assign_from_beginning(consumer: object, partitions: list) -> None:
    """Rewind every assigned partition so a rerun reprocesses the whole topic."""
    from confluent_kafka import OFFSET_BEGINNING

    for partition in partitions:
        partition.offset = OFFSET_BEGINNING
    consumer.assign(partitions)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--lateness-seconds", type=float, default=DEFAULT_LATENESS_SECONDS)
    parser.add_argument("--volume-csv", type=Path, default=DEFAULT_VOLUME_CSV)
    parser.add_argument("--alerts-jsonl", type=Path, default=DEFAULT_ALERTS_JSONL)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT)
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT)
    parser.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT)
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Rewind to offset 0 on assignment (deterministic rerun)",
    )
    parser.add_argument(
        "--reset-outputs",
        action="store_true",
        help="Delete the volume table and alert log first; the writers only append",
    )
    parser.add_argument(
        "--no-publish-alerts",
        action="store_true",
        help="Write alerts to the log only, without producing to traffic.alerts",
    )
    args = parser.parse_args(argv)

    if args.reset_outputs:
        args.volume_csv.unlink(missing_ok=True)
        args.alerts_jsonl.unlink(missing_ok=True)

    topic = os.environ.get("TOPIC_VEHICLE_EVENTS", DEFAULT_TOPIC)
    consumer = _default_consumer_factory(consumer_config(args.group_id))
    if args.from_beginning:
        consumer.subscribe([topic], on_assign=_assign_from_beginning)
    else:
        consumer.subscribe([topic])
    windows = TumblingWindows(args.window_seconds, args.lateness_seconds)
    alert_producer = None if args.no_publish_alerts else build_alert_producer()
    writer = WindowWriter(args.volume_csv, args.alerts_jsonl, alert_producer=alert_producer)
    print(f"consuming {topic} into {args.window_seconds}s windows (group {args.group_id})")
    try:
        stats = consume(
            consumer,
            build_avro_deserializer(),
            windows,
            lambda closed: writer.handle(closed, windows),
            args.max_messages,
            args.poll_timeout,
            args.idle_timeout,
            args.run_timeout,
        )
    finally:
        consumer.close()
    print(
        f"consumed {stats.consumed}, windows {stats.windows_closed}, alerts {stats.alerts}, "
        f"late dropped {stats.late_dropped} ({stats.stop_reason})"
    )
    if alert_producer is not None:
        delivery = alert_producer.flush()
        print(f"published {delivery.delivered} alerts to {alert_producer.topic}")
    print(f"wrote {args.volume_csv} and {args.alerts_jsonl}")


if __name__ == "__main__":
    main()
