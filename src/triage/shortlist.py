"""Rank triage-scan results into corridor candidates for the perception pipeline.

Deterministic follow-up to scan_cameras.py: reads a scan CSV, keeps working
streams, scores each camera, and groups same-route cameras into geographic
clusters ("corridors"). The goal is 2-4 good cameras on ONE corridor.

Scoring favors resolution first (detection quality dominates everything
downstream), then evidence of visible vehicles from the scan's YOLO pass.

Usage:
    uv run python -m src.triage.shortlist                    # latest scan CSV
    uv run python -m src.triage.shortlist --scan path.csv --top 8
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

MIN_PIXELS = 352 * 240  # anything smaller is unusable for detection


@dataclass
class Camera:
    camera_id: str
    route: str
    direction: str
    place: str
    lat: float
    lon: float
    width: int
    height: int
    vehicles: int
    score: float


def vehicle_count(yolo_field: str) -> int:
    """'car:5,truck:1' -> 6."""
    if not yolo_field:
        return 0
    return sum(int(part.split(":")[1]) for part in yolo_field.split(",") if ":" in part)


def score(width: int, height: int, vehicles: int) -> float:
    """Resolution dominates; visible vehicles break ties.

    1080p ≈ 29 + vehicles, 720p ≈ 13, CIF (352x240) ≈ 1.2 — so a single HD
    camera outranks any number of vehicles seen on a CIF stream.
    """
    return (width * height) / MIN_PIXELS + min(vehicles, 10)


def load_cameras(scan_csv: Path) -> list[Camera]:
    cams = []
    with scan_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row["stream_ok"] != "True":
                continue
            w, h = int(row["width"]), int(row["height"])
            if w * h < MIN_PIXELS:
                continue
            n = vehicle_count(row.get("yolo_vehicles", ""))
            cams.append(
                Camera(
                    camera_id=row["camera_id"],
                    route=row["route"],
                    direction=row["direction"],
                    place=row["place"],
                    lat=float(row["latitude"]),
                    lon=float(row["longitude"]),
                    width=w,
                    height=h,
                    vehicles=n,
                    score=score(w, h, n),
                )
            )
    return cams


def km_between(a: Camera, b: Camera) -> float:
    """Equirectangular approximation — plenty for corridor grouping."""
    dx = math.radians(b.lon - a.lon) * math.cos(math.radians((a.lat + b.lat) / 2))
    dy = math.radians(b.lat - a.lat)
    return 6371 * math.hypot(dx, dy)


def cluster_corridors(cams: list[Camera], max_gap_km: float = 12.0) -> list[list[Camera]]:
    """Group same-route cameras into chains where neighbors are < max_gap_km apart."""
    by_route: dict[str, list[Camera]] = {}
    for c in cams:
        by_route.setdefault(c.route, []).append(c)

    corridors = []
    for members in by_route.values():
        members.sort(key=lambda c: (c.lat, c.lon))
        chain = [members[0]]
        for cam in members[1:]:
            if km_between(chain[-1], cam) <= max_gap_km:
                chain.append(cam)
            else:
                corridors.append(chain)
                chain = [cam]
        corridors.append(chain)
    # A corridor needs at least 2 cameras; rank by total score.
    corridors = [c for c in corridors if len(c) >= 2]
    corridors.sort(key=lambda c: sum(cam.score for cam in c), reverse=True)
    return corridors


def latest_scan(results_dir: Path) -> Path:
    scans = sorted(results_dir.glob("scan_*.csv"))
    if not scans:
        raise SystemExit(f"no scan_*.csv in {results_dir} — run scan_cameras first")
    return scans[-1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, help="Scan CSV (default: latest in --results-dir)")
    parser.add_argument("--results-dir", type=Path, default=Path("triage-results"))
    parser.add_argument("--top", type=int, default=5, help="Corridors to print")
    parser.add_argument("--max-gap-km", type=float, default=12.0)
    args = parser.parse_args(argv)

    scan_csv = args.scan or latest_scan(args.results_dir)
    cams = load_cameras(scan_csv)
    corridors = cluster_corridors(cams, args.max_gap_km)
    print(f"{scan_csv}: {len(cams)} usable cameras -> {len(corridors)} corridor candidates\n")

    for rank, corridor in enumerate(corridors[: args.top], 1):
        total = sum(c.score for c in corridor)
        span = max(km_between(corridor[0], c) for c in corridor)
        print(f"#{rank}  {corridor[0].route}  ({len(corridor)} cams, ~{span:.0f} km span, "
              f"score {total:.1f})")
        for c in sorted(corridor, key=lambda c: -c.score):
            print(f"      {c.camera_id:<8} {c.direction:<5} {c.width}x{c.height:<5} "
                  f"vehicles:{c.vehicles:<3} {c.place}")
        print()


if __name__ == "__main__":
    main()
