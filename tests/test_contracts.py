"""Contract tests: the Avro schema and the Pydantic model must agree."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.streaming.contracts import (
    TravelDirection,
    VehicleClass,
    VehicleEvent,
    avro_and_pydantic_field_names_match,
    load_avro_schema_str,
)


def make_event(**overrides) -> VehicleEvent:
    defaults = dict(
        camera_id="tv516",
        ts_event=datetime(2026, 8, 9, 15, 0, 0, tzinfo=UTC),
        ts_publish=datetime(2026, 8, 9, 15, 0, 1, tzinfo=UTC),
        track_id=42,
        vehicle_class=VehicleClass.car,
        direction=TravelDirection.EB,
        confidence=0.87,
    )
    defaults.update(overrides)
    return VehicleEvent(**defaults)


def test_avro_and_pydantic_fields_match():
    assert avro_and_pydantic_field_names_match()


def test_avro_schema_parses_and_names_topic_record():
    assert '"name": "VehicleEvent"' in load_avro_schema_str()


def test_avro_roundtrip_preserves_event():
    event = make_event()
    restored = VehicleEvent.from_avro_dict(event.to_avro_dict())
    assert restored == event


def test_avro_dict_uses_epoch_millis():
    d = make_event().to_avro_dict()
    assert d["ts_event"] == 1786287600000
    assert d["ts_publish"] - d["ts_event"] == 1000


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        make_event(ts_event=datetime(2026, 8, 9, 15, 0, 0))


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        make_event(confidence=1.5)


def test_unknown_vehicle_class_rejected():
    with pytest.raises(ValidationError):
        make_event(vehicle_class="bicycle")
