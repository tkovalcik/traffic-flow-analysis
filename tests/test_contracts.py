"""Contract tests: the Avro schema and the Pydantic model must agree."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.streaming.contracts import (
    AlertType,
    TrafficAlert,
    TravelDirection,
    VehicleClass,
    VehicleEvent,
    avro_and_pydantic_field_names_match,
    contract_fields_match,
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


def make_alert(**overrides) -> TrafficAlert:
    defaults = dict(
        camera_id="tva43",
        alert_type=AlertType.volume_spike,
        ts_alert=datetime(2026, 8, 9, 15, 15, tzinfo=UTC),
        window_start=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
        window_end=datetime(2026, 8, 9, 15, 15, tzinfo=UTC),
        direction=TravelDirection.EB,
        observed_count=50,
        baseline=20.0,
        message="spike",
    )
    defaults.update(overrides)
    return TrafficAlert(**defaults)


def test_alert_avro_and_pydantic_fields_match():
    assert contract_fields_match("traffic_alert", TrafficAlert)


def test_alert_avro_roundtrip_preserves_alert():
    alert = make_alert()
    assert TrafficAlert.from_avro_dict(alert.to_avro_dict()) == alert


def test_alert_roundtrip_with_nulls():
    alert = make_alert(direction=None, observed_count=None, baseline=None)
    d = alert.to_avro_dict()
    assert d["direction"] is None
    assert TrafficAlert.from_avro_dict(d) == alert


def test_alert_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        make_alert(window_start=datetime(2026, 8, 9, 15, 0))
