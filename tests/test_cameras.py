"""Tests for the camera registry."""

import json

import pytest

from src.perception.cameras import fetch_registry, get_camera

INVENTORY = {
    "data": [
        {
            "cctv": {
                "index": "1",
                "inService": "true",
                "location": {
                    "locationName": "TV516 -- I-80 : Powell Street",
                    "nearbyPlace": "Emeryville",
                    "longitude": "-122.29",
                    "latitude": "37.84",
                    "direction": "East",
                    "county": "Alameda",
                    "route": "I-80",
                },
                "imageData": {
                    "streamingVideoURL": "https://example.invalid/tv516.m3u8",
                    "static": {"currentImageURL": "https://example.invalid/tv516.jpg"},
                },
            }
        },
        {
            "cctv": {
                "index": "2",
                "inService": "false",
                "location": {"locationName": "TV999 -- I-80 : Nowhere"},
                "imageData": {"streamingVideoURL": ""},
            }
        },
    ]
}


@pytest.fixture
def inventory_path(tmp_path):
    p = tmp_path / "inventory.json"
    p.write_text(json.dumps(INVENTORY))
    return str(p)


def test_registry_indexes_by_lowercase_id(inventory_path):
    registry = fetch_registry(inventory_path)
    assert set(registry) == {"tv516", "tv999"}
    cam = registry["tv516"]
    assert cam.route == "I-80"
    assert cam.place == "Emeryville"
    assert cam.latitude == pytest.approx(37.84)
    assert cam.in_service is True


def test_get_camera_is_case_insensitive_and_keeps_raw_record(inventory_path):
    cam = get_camera("TV516", inventory_path)
    assert cam.stream_url.endswith("tv516.m3u8")
    assert cam.raw["location"]["locationName"].startswith("TV516")


def test_registry_stamps_fetch_provenance(inventory_path):
    cam = get_camera("tv516", inventory_path)
    assert cam.inventory_source == inventory_path
    assert cam.retrieved_utc  # ISO timestamp of when WE fetched the record
    assert cam.retrieved_utc.endswith("+00:00")


def test_get_camera_unknown_id_raises(inventory_path):
    with pytest.raises(KeyError, match="tv000"):
        get_camera("tv000", inventory_path)


def test_get_camera_without_stream_raises(inventory_path):
    with pytest.raises(ValueError, match="static-only"):
        get_camera("tv999", inventory_path)
