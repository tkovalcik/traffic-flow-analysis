"""Event contracts: Pydantic models mirroring the Avro schemas.

The Avro schema (src/streaming/schemas/*.avsc, registered in Schema Registry) is
the wire contract; these models are the in-process validation layer. If you
change one, change both — tests/test_contracts.py checks they stay in sync.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

SCHEMA_DIR = Path(__file__).parent / "schemas"


def load_avro_schema_str(name: str = "vehicle_event") -> str:
    """Raw Avro schema JSON string, as the Schema Registry serializer wants it."""
    return (SCHEMA_DIR / f"{name}.avsc").read_text()


class VehicleClass(StrEnum):
    car = "car"
    truck = "truck"
    bus = "bus"
    motorcycle = "motorcycle"


class TravelDirection(StrEnum):
    NB = "NB"
    SB = "SB"
    EB = "EB"
    WB = "WB"


class VehicleEvent(BaseModel):
    """One vehicle crossing a counting line on one camera (topic: vehicle.events)."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    camera_id: str
    ts_event: datetime
    ts_publish: datetime
    track_id: int
    vehicle_class: VehicleClass
    direction: TravelDirection
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("ts_event", "ts_publish")
    @classmethod
    def require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (use UTC)")
        return v.astimezone(UTC)

    def to_avro_dict(self) -> dict:
        """Dict shaped for the Avro serializer (timestamps → epoch millis)."""
        d = self.model_dump()
        d["ts_event"] = int(self.ts_event.timestamp() * 1000)
        d["ts_publish"] = int(self.ts_publish.timestamp() * 1000)
        d["vehicle_class"] = self.vehicle_class.value
        d["direction"] = self.direction.value
        return d

    @classmethod
    def from_avro_dict(cls, d: dict) -> VehicleEvent:
        """Inverse of to_avro_dict (epoch millis → aware datetimes)."""
        d = dict(d)
        for key in ("ts_event", "ts_publish"):
            if isinstance(d.get(key), (int, float)):
                d[key] = datetime.fromtimestamp(d[key] / 1000, tz=UTC)
        return cls(**d)


def avro_and_pydantic_field_names_match() -> bool:
    """True when the .avsc field set equals VehicleEvent's — used by tests."""
    avro_fields = {f["name"] for f in json.loads(load_avro_schema_str())["fields"]}
    return avro_fields == set(VehicleEvent.model_fields)
