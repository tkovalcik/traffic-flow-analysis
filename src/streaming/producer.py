"""Kafka producer for `vehicle.events`: Avro serialization via Schema Registry.

Shared by the perception and replay producers. Events are keyed by `camera_id`,
so each camera's events keep their order within a partition. Broker and registry
settings come from the environment, so moving to Confluent Cloud is a config swap.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from confluent_kafka.serialization import MessageField, SerializationContext

from src.streaming.contracts import VehicleEvent, load_avro_schema_str

DEFAULT_TOPIC = "vehicle.events"
DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_SCHEMA_REGISTRY = "http://localhost:8081"


def kafka_config(env: Mapping[str, str] | None = None) -> dict:
    """Producer config from the environment: Confluent Cloud if set, else local."""
    env = os.environ if env is None else env
    cloud = env.get("CONFLUENT_BOOTSTRAP_SERVERS", "").strip()
    config: dict = {
        "bootstrap.servers": cloud or env.get("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP),
        # The broker de-duplicates retried batches, so a reconnecting producer
        # cannot republish a crossing and double-count a vehicle downstream.
        "enable.idempotence": True,
        "acks": "all",
    }
    if cloud:
        config.update(
            {
                "security.protocol": "SASL_SSL",
                "sasl.mechanisms": "PLAIN",
                "sasl.username": env.get("CONFLUENT_API_KEY", ""),
                "sasl.password": env.get("CONFLUENT_API_SECRET", ""),
            }
        )
    return config


def schema_registry_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Schema Registry config, with Confluent basic auth when credentials exist."""
    env = os.environ if env is None else env
    cloud_url = env.get("CONFLUENT_SR_URL", "").strip()
    config = {"url": cloud_url or env.get("SCHEMA_REGISTRY_URL", DEFAULT_SCHEMA_REGISTRY)}
    key = env.get("CONFLUENT_SR_API_KEY", "").strip()
    secret = env.get("CONFLUENT_SR_API_SECRET", "").strip()
    if cloud_url and key and secret:
        config["basic.auth.user.info"] = f"{key}:{secret}"
    return config


@dataclass
class DeliveryStats:
    """Broker-acknowledged outcomes — the producer's own health signal."""

    delivered: int = 0
    failed: int = 0
    pending: int = 0
    last_error: str = ""


def build_avro_serializer(env: Mapping[str, str] | None = None) -> Callable:
    """AvroSerializer for vehicle_event; imported lazily so tests need no registry."""
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer

    client = SchemaRegistryClient(schema_registry_config(env))
    return AvroSerializer(
        client,
        load_avro_schema_str("vehicle_event"),
        lambda event, _ctx: event.to_avro_dict(),
    )


def _default_producer_factory(config: dict):
    from confluent_kafka import Producer

    return Producer(config)


class EventProducer:
    """Publish VehicleEvents to `vehicle.events`, keyed by camera_id.

    producer_factory and serializer exist for tests, mirroring FrameSource's
    capture_factory: the publish path runs without a broker or a registry.
    """

    def __init__(
        self,
        topic: str | None = None,
        env: Mapping[str, str] | None = None,
        producer_factory: Callable[[dict], object] | None = None,
        serializer: Callable | None = None,
    ):
        env = os.environ if env is None else env
        self.topic = topic or env.get("TOPIC_VEHICLE_EVENTS", DEFAULT_TOPIC)
        self.stats = DeliveryStats()
        self._serializer = build_avro_serializer(env) if serializer is None else serializer
        factory = producer_factory or _default_producer_factory
        self._producer = factory(kafka_config(env))

    def _on_delivery(self, err: object | None, _msg: object) -> None:
        if err is None:
            self.stats.delivered += 1
        else:
            self.stats.failed += 1
            self.stats.last_error = str(err)

    def publish(self, event: VehicleEvent) -> None:
        """Serialize and enqueue one event. Delivery is confirmed on flush()."""
        payload = self._serializer(event, SerializationContext(self.topic, MessageField.VALUE))
        self._producer.produce(
            topic=self.topic,
            key=event.camera_id.encode("utf-8"),
            value=payload,
            on_delivery=self._on_delivery,
        )
        # Serve queued delivery callbacks without blocking the caller.
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> DeliveryStats:
        """Block until the queue drains (or timeout); returns delivery stats."""
        self.stats.pending = int(self._producer.flush(timeout) or 0)
        return self.stats
