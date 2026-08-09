"""Camera registry: resolve camera ids to stream URLs and full source metadata.

Wraps the state-DOT CCTV inventory (same JSON the triage scanner reads). The
inventory URL comes from CCTV_INVENTORY_URL in .env — never hardcoded. Each
camera keeps its complete raw inventory record so downstream artifacts (clip
metadata, events) can carry full provenance.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.triage.scan_cameras import load_inventory


@dataclass
class CameraInfo:
    camera_id: str
    name: str
    route: str
    direction: str
    county: str
    place: str
    latitude: float
    longitude: float
    stream_url: str
    static_url: str
    in_service: bool
    raw: dict = field(repr=False)


def _parse_record(cam: dict) -> CameraInfo:
    loc = cam.get("location", {})
    name = loc.get("locationName", "")
    m = re.match(r"\s*(\S+?)\s*--", name)
    camera_id = (m.group(1) if m else f"idx{cam.get('index', '?')}").lower()
    img = cam.get("imageData", {}) or {}
    return CameraInfo(
        camera_id=camera_id,
        name=name,
        route=loc.get("route", ""),
        direction=loc.get("direction", ""),
        county=loc.get("county", ""),
        place=loc.get("nearbyPlace", ""),
        latitude=float(loc.get("latitude") or 0.0),
        longitude=float(loc.get("longitude") or 0.0),
        stream_url=img.get("streamingVideoURL", "") or "",
        static_url=(img.get("static", {}) or {}).get("currentImageURL", "") or "",
        in_service=str(cam.get("inService", "")).lower() == "true",
        raw=cam,
    )


def fetch_registry(inventory_source: str | None = None) -> dict[str, CameraInfo]:
    """Load the inventory and index cameras by id.

    inventory_source may be a URL or local JSON path; defaults to
    $CCTV_INVENTORY_URL.
    """
    load_dotenv()
    source = inventory_source or os.environ.get("CCTV_INVENTORY_URL", "")
    if not source:
        raise ValueError("no inventory: set CCTV_INVENTORY_URL in .env or pass a source")
    registry: dict[str, CameraInfo] = {}
    for record in load_inventory(source):
        cam = _parse_record(record)
        registry[cam.camera_id] = cam
    return registry


def get_camera(camera_id: str, inventory_source: str | None = None) -> CameraInfo:
    registry = fetch_registry(inventory_source)
    cam = registry.get(camera_id.lower())
    if cam is None:
        raise KeyError(f"camera {camera_id!r} not in inventory ({len(registry)} cameras)")
    if not cam.stream_url:
        raise ValueError(f"camera {camera_id!r} has no streaming URL (static-only)")
    return cam
