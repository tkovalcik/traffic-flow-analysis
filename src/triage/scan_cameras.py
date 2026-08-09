"""Scan a state-DOT CCTV inventory and report which video streams are usable.

Reads a camera inventory JSON (Caltrans district CCTV status format), probes each
in-service camera's HLS stream with OpenCV, and writes a CSV report plus one
thumbnail per reachable camera. Use the report to shortlist cameras for the
perception pipeline.

The inventory URL is intentionally not hardcoded: set CCTV_INVENTORY_URL in .env
(see .env.example) or pass --inventory with a URL or a local JSON path.

Usage:
    uv run python -m src.triage.scan_cameras --limit 10          # smoke test
    uv run python -m src.triage.scan_cameras                     # full scan
    uv run python -m src.triage.scan_cameras --route I-880 --with-yolo

Outputs (gitignored): triage-results/scan_<timestamp>.csv and
triage-results/thumbnails/<camera_id>.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Must be set before cv2 is imported so its bundled ffmpeg honors the I/O timeout
# (value in microseconds). Prevents indefinite hangs on dead HTTP streams.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;10000000")

import cv2  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

VEHICLE_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class ProbeResult:
    camera_id: str
    name: str
    route: str
    direction: str
    county: str
    place: str
    latitude: str
    longitude: str
    in_service: bool
    stream_url: str
    static_url: str
    stream_ok: bool = False
    width: int = 0
    height: int = 0
    fps_nominal: float = 0.0
    fps_measured: float = 0.0
    frames_read: int = 0
    static_ok: bool = False
    thumbnail: str = ""
    yolo_vehicles: str = ""
    error: str = ""


def load_inventory(source: str) -> list[dict]:
    """Load camera inventory from a URL or local JSON file path."""
    if source.startswith(("http://", "https://")):
        raw = requests.get(source, timeout=30).json()
    else:
        raw = json.loads(Path(source).read_text())
    return [item["cctv"] for item in raw["data"]]


def parse_camera(cam: dict) -> ProbeResult:
    loc = cam["location"]
    name = loc.get("locationName", "")
    m = re.match(r"\s*(\S+?)\s*--", name)
    camera_id = (m.group(1) if m else f"idx{cam.get('index', '?')}").lower()
    img = cam.get("imageData", {})
    return ProbeResult(
        camera_id=camera_id,
        name=name,
        route=loc.get("route", ""),
        direction=loc.get("direction", ""),
        county=loc.get("county", ""),
        place=loc.get("nearbyPlace", ""),
        latitude=loc.get("latitude", ""),
        longitude=loc.get("longitude", ""),
        in_service=str(cam.get("inService", "")).lower() == "true",
        stream_url=img.get("streamingVideoURL", "") or "",
        static_url=(img.get("static", {}) or {}).get("currentImageURL", "") or "",
    )


def probe_stream(
    result: ProbeResult,
    thumb_dir: Path,
    warmup_frames: int = 8,
    sample_frames: int = 24,
    deadline_s: float = 20.0,
) -> ProbeResult:
    """Open the HLS stream, read frames, estimate FPS, save a thumbnail.

    fps_nominal comes from the stream's own metadata. fps_measured is timed over
    the last few reads and can exceed nominal while the initially buffered HLS
    segments drain at decode speed — treat it as "frames are flowing", not exact
    cadence. (Precise cadence is the capture module's job, not triage's.)
    """
    cap = cv2.VideoCapture(result.stream_url, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            result.error = "stream did not open"
            return result
        nominal = cap.get(cv2.CAP_PROP_FPS)
        if nominal and 0 < nominal < 240:
            result.fps_nominal = round(nominal, 2)
        start = time.monotonic()
        read_times: list[float] = []
        frame = None
        while time.monotonic() - start < deadline_s:
            ok, frame_read = cap.read()
            if not ok:
                break
            frame = frame_read
            read_times.append(time.monotonic())
            result.frames_read += 1
            if result.frames_read == 1:
                result.height, result.width = frame.shape[:2]
            if result.frames_read >= warmup_frames + sample_frames:
                break
        tail = read_times[-10:]
        if len(tail) >= 2 and tail[-1] > tail[0]:
            result.fps_measured = round((len(tail) - 1) / (tail[-1] - tail[0]), 2)
        result.stream_ok = result.frames_read > 0
        if not result.stream_ok:
            result.error = "opened but no frames"
        if frame is not None:
            thumb = thumb_dir / f"{result.camera_id}.jpg"
            cv2.imwrite(str(thumb), frame)
            result.thumbnail = str(thumb)
    except Exception as exc:  # a scan must survive any single bad camera
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
    return result


def fetch_static_fallback(result: ProbeResult, thumb_dir: Path) -> ProbeResult:
    """If the stream gave nothing, try the static snapshot as thumbnail/fallback."""
    if not result.static_url:
        return result
    try:
        resp = requests.get(result.static_url, timeout=10)
        result.static_ok = resp.ok and len(resp.content) > 1000
        if result.static_ok and not result.thumbnail:
            thumb = thumb_dir / f"{result.camera_id}_static.jpg"
            thumb.write_bytes(resp.content)
            result.thumbnail = str(thumb)
    except requests.RequestException:
        result.static_ok = False
    return result


def annotate_with_yolo(results: list[ProbeResult], model_name: str) -> None:
    """Run YOLO on saved thumbnails and record vehicle counts, e.g. 'car:5,truck:1'."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    for r in results:
        if not r.thumbnail:
            continue
        counts: dict[str, int] = {}
        for box in model.predict(r.thumbnail, verbose=False)[0].boxes:
            label = VEHICLE_COCO_CLASSES.get(int(box.cls))
            if label and float(box.conf) >= 0.3:
                counts[label] = counts.get(label, 0) + 1
        r.yolo_vehicles = ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def write_report(results: list[ProbeResult], out_csv: Path) -> None:
    cols = [f.name for f in fields(ProbeResult)]
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def summarize(results: list[ProbeResult]) -> str:
    ok = [r for r in results if r.stream_ok]
    by_route: dict[str, int] = {}
    for r in ok:
        by_route[r.route] = by_route.get(r.route, 0) + 1
    lines = [
        f"probed {len(results)} cameras: {len(ok)} streams OK, "
        f"{sum(1 for r in results if r.static_ok and not r.stream_ok)} static-only, "
        f"{sum(1 for r in results if not r.stream_ok and not r.static_ok)} dead",
        "streams OK by route: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_route.items(), key=lambda kv: -kv[1])),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        default=os.environ.get("CCTV_INVENTORY_URL", ""),
        help="Inventory JSON URL or local path (default: $CCTV_INVENTORY_URL)",
    )
    parser.add_argument("--route", action="append", help="Filter: route substring, repeatable")
    parser.add_argument("--county", help="Filter: county substring")
    parser.add_argument("--place", help="Filter: nearby-place substring")
    parser.add_argument("--limit", type=int, help="Probe at most N cameras")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--deadline", type=float, default=20.0, help="Seconds per camera probe")
    parser.add_argument("--with-yolo", action="store_true", help="Vehicle counts on thumbnails")
    parser.add_argument("--yolo-model", default=os.environ.get("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument("--out-dir", default="triage-results")
    args = parser.parse_args(argv)

    if not args.inventory:
        parser.error("no inventory: set CCTV_INVENTORY_URL in .env or pass --inventory")

    out_dir = Path(args.out_dir)
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    cameras = [parse_camera(c) for c in load_inventory(args.inventory)]
    candidates = [c for c in cameras if c.in_service and c.stream_url]
    for attr, needle_or_list in (("route", args.route), ("county", args.county), ("place", args.place)):
        if needle_or_list:
            needles = needle_or_list if isinstance(needle_or_list, list) else [needle_or_list]
            candidates = [
                c for c in candidates
                if any(n.lower() in getattr(c, attr).lower() for n in needles)
            ]
    skipped = len(cameras) - len(candidates)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"{len(cameras)} cameras in inventory; probing {len(candidates)} "
          f"(skipped {skipped} out-of-service/streamless/filtered)")

    results: list[ProbeResult] = []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending: set[Future] = {
        pool.submit(probe_stream, cam, thumb_dir, deadline_s=args.deadline)
        for cam in candidates
    }
    # Overall deadline: generous per-batch budget; anything unfinished after it is
    # a hung ffmpeg read we abandon (bundled ffmpeg can ignore its I/O timeout).
    overall_deadline = time.monotonic() + args.deadline * (len(candidates) / args.workers + 3)
    while pending and time.monotonic() < overall_deadline:
        done, pending = wait(pending, timeout=10.0, return_when=FIRST_COMPLETED)
        for fut in done:
            res = fetch_static_fallback(fut.result(), thumb_dir)
            results.append(res)
            status = "OK " if res.stream_ok else ("img" if res.static_ok else "DEAD")
            print(f"  [{status}] {res.camera_id:<8} {res.route:<7} {res.direction:<5} "
                  f"{res.width}x{res.height} @ {res.fps_measured} fps  {res.place}",
                  flush=True)
    for fut in pending:  # abandoned probes still get a row
        fut.cancel()

    if args.with_yolo:
        print("running YOLO on thumbnails ...")
        annotate_with_yolo(results, args.yolo_model)

    results.sort(key=lambda r: (not r.stream_ok, r.route, r.camera_id))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"scan_{stamp}.csv"
    write_report(results, out_csv)
    print(summarize(results))
    print(f"report: {out_csv}\nthumbnails: {thumb_dir}/")

    # Hung cv2/ffmpeg reader threads cannot be cancelled and would block normal
    # interpreter exit; all output is already flushed to disk at this point.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
