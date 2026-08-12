"""Tests for the vehicle.events Kafka producer (fakes — no broker needed)."""

from datetime import UTC, datetime

from src.streaming.contracts import TravelDirection, VehicleClass, VehicleEvent
from src.streaming.producer import (
    EventProducer,
    kafka_config,
    schema_registry_config,
)

LOCAL_ENV = {
    "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
    "SCHEMA_REGISTRY_URL": "http://localhost:8081",
    "TOPIC_VEHICLE_EVENTS": "vehicle.events",
}

CLOUD_ENV = {
    **LOCAL_ENV,
    "CONFLUENT_BOOTSTRAP_SERVERS": "pkc-abc.us-west1.gcp.confluent.cloud:9092",
    "CONFLUENT_API_KEY": "KEY123",
    "CONFLUENT_API_SECRET": "SECRET456",
    "CONFLUENT_SR_URL": "https://psrc-def.us-west1.gcp.confluent.cloud",
    "CONFLUENT_SR_API_KEY": "SRKEY",
    "CONFLUENT_SR_API_SECRET": "SRSECRET",
}


class FakeProducer:
    """Stands in for confluent_kafka.Producer, invoking delivery callbacks inline."""

    def __init__(self, config, error=None):
        self.config = config
        self.produced = []
        self.polls = 0
        self.flushes = 0
        self._error = error

    def produce(self, topic, key, value, on_delivery):
        self.produced.append({"topic": topic, "key": key, "value": value})
        on_delivery(self._error, None)

    def poll(self, _timeout):
        self.polls += 1

    def flush(self, _timeout):
        self.flushes += 1
        return 0


def make_event(camera_id="tva43", direction=TravelDirection.EB):
    return VehicleEvent(
        camera_id=camera_id,
        ts_event=datetime(2026, 8, 9, 18, 46, 30, tzinfo=UTC),
        ts_publish=datetime(2026, 8, 9, 18, 46, 31, tzinfo=UTC),
        track_id=21,
        vehicle_class=VehicleClass.car,
        direction=direction,
        confidence=0.88,
    )


def build(env=None, error=None, serializer=None):
    """EventProducer wired to a FakeProducer; returns (producer, fake)."""
    made = {}

    def factory(config):
        made["producer"] = FakeProducer(config, error=error)
        return made["producer"]

    producer = EventProducer(
        env=env or LOCAL_ENV,
        producer_factory=factory,
        serializer=serializer or (lambda event, _ctx: b"avro:" + event.event_id.encode()),
    )
    return producer, made["producer"]


def test_local_config_defaults_to_compose_broker():
    config = kafka_config(LOCAL_ENV)
    assert config["bootstrap.servers"] == "localhost:9092"
    assert "security.protocol" not in config


def test_idempotence_is_always_on():
    for env in (LOCAL_ENV, CLOUD_ENV):
        config = kafka_config(env)
        assert config["enable.idempotence"] is True
        assert config["acks"] == "all"


def test_confluent_env_switches_to_sasl_without_code_change():
    config = kafka_config(CLOUD_ENV)
    assert config["bootstrap.servers"] == CLOUD_ENV["CONFLUENT_BOOTSTRAP_SERVERS"]
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.username"] == "KEY123"
    assert config["sasl.password"] == "SECRET456"


def test_schema_registry_config_local_has_no_auth():
    config = schema_registry_config(LOCAL_ENV)
    assert config == {"url": "http://localhost:8081"}


def test_schema_registry_config_cloud_carries_basic_auth():
    config = schema_registry_config(CLOUD_ENV)
    assert config["url"] == CLOUD_ENV["CONFLUENT_SR_URL"]
    assert config["basic.auth.user.info"] == "SRKEY:SRSECRET"


def test_publish_keys_on_camera_id():
    producer, fake = build()
    producer.publish(make_event(camera_id="tv516"))
    assert fake.produced[0]["key"] == b"tv516"
    assert fake.produced[0]["topic"] == "vehicle.events"


def test_publish_sends_serialized_payload():
    event = make_event()
    producer, fake = build()
    producer.publish(event)
    assert fake.produced[0]["value"] == b"avro:" + event.event_id.encode()


def test_serialization_context_targets_the_topic_value():
    seen = {}

    def serializer(event, ctx):
        seen["topic"] = ctx.topic
        seen["field"] = ctx.field
        return b"payload"

    producer, _fake = build(serializer=serializer)
    producer.publish(make_event())
    assert seen["topic"] == "vehicle.events"
    assert seen["field"] == "value"


def test_delivered_events_are_counted():
    producer, _fake = build()
    for _ in range(3):
        producer.publish(make_event())
    assert producer.stats.delivered == 3
    assert producer.stats.failed == 0


def test_delivery_errors_are_counted_and_reported():
    producer, _fake = build(error="broker down")
    producer.publish(make_event())
    assert producer.stats.failed == 1
    assert producer.stats.delivered == 0
    assert "broker down" in producer.stats.last_error


def test_flush_reports_stats_and_drains():
    producer, fake = build()
    producer.publish(make_event())
    stats = producer.flush()
    assert fake.flushes == 1
    assert stats.delivered == 1
    assert stats.pending == 0


def test_topic_comes_from_environment():
    producer, fake = build(env={**LOCAL_ENV, "TOPIC_VEHICLE_EVENTS": "custom.events"})
    producer.publish(make_event())
    assert fake.produced[0]["topic"] == "custom.events"
