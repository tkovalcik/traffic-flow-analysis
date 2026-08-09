"""Record timestamped video clips from live cameras, with full provenance metadata.

Every clip gets a sidecar JSON carrying the camera's complete inventory record
(location, route, direction, coordinates, image endpoints — everything the DOT
publishes) plus capture provenance (when, how many frames, effective FPS). Clips
are the raw material for the golden replay dataset and are archived to the GCS
bucket with --upload.

Usage:
    uv run python -m src.replay.record --camera tv516 --seconds 75 --upload
    uv run python -m src.replay.record --camera tva43 --seconds 75 --out data/captures
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

from src.perception.cameras import CameraInfo, get_camera
from src.perception.capture import FrameSource

METADATA_SCHEMA_VERSION = 1
FPS_FALLBACK = 15.0


@dataclass
class RecordedClip:
    video_path: Path
    metadata_path: Path
    frames: int
    fps_writer: float
    duration_wall_s: float


def record_clip(
    camera: CameraInfo,
    seconds: float,
    out_dir: Path,
    fps_fallback: float = FPS_FALLBACK,
    reconnect_attempts: int = 5,
    backoff_s: float = 2.0,
) -> RecordedClip:
    """Capture ~`seconds` of video from the camera's stream to MP4 + JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}_{camera.camera_id}"
    video_path = out_dir / f"{base}.mp4"
    metadata_path = out_dir / f"{base}.json"

    source = FrameSource(
        camera.stream_url, reconnect_attempts=reconnect_attempts, backoff_s=backoff_s
    )
    writer: cv2.VideoWriter | None = None
    fps_writer = fps_fallback
    started_utc = datetime.now(UTC)
    started_mono = time.monotonic()
    frames = 0
    first_ts = last_ts = None
    max_gap_s = 0.0
    prev_read_mono: float | None = None

    # Bound by content frames (fps * seconds) AND wall clock: live HLS delivers
    # its initial buffer faster than realtime, so frame count alone would over-
    # shoot the requested window and wall clock alone would under-fill it.
    max_seconds = seconds * 1.5 + 15
    for captured in source.frames(max_seconds=max_seconds):
        if writer is None:
            if 1.0 < source.fps_nominal < 121.0:
                fps_writer = source.fps_nominal
            h, w = captured.frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_writer, (w, h)
            )
            first_ts = captured.ts_utc
        writer.write(captured.frame)
        last_ts = captured.ts_utc
        now_mono = time.monotonic()
        if prev_read_mono is not None:
            max_gap_s = max(max_gap_s, now_mono - prev_read_mono)
        prev_read_mono = now_mono
        frames += 1
        if frames >= int(fps_writer * seconds):
            break
    if writer is not None:
        writer.release()
    duration_wall = time.monotonic() - started_mono

    if frames == 0:
        raise RuntimeError(f"no frames captured from {camera.camera_id} — stream down?")

    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "camera_id": camera.camera_id,
        "recorded": {
            "started_utc": started_utc.isoformat(),
            "first_frame_utc": first_ts.isoformat() if first_ts else None,
            "last_frame_utc": last_ts.isoformat() if last_ts else None,
            "requested_seconds": seconds,
            "wall_seconds": round(duration_wall, 2),
            "frames": frames,
            "fps_writer": fps_writer,
            "fps_nominal": source.fps_nominal,
            "video_seconds": round(frames / fps_writer, 2),
            "width": source.width,
            "height": source.height,
            "codec": "mp4v",
            "reconnects": source.reconnects,
            "read_failures": source.read_failures,
            "max_interframe_gap_s": round(max_gap_s, 3),
        },
        "source": {
            "stream_url": camera.stream_url,
            "inventory_record": camera.raw,
            "inventory_source": camera.inventory_source,
            "inventory_retrieved_utc": camera.retrieved_utc,
        },
        "recorder": "src.replay.record",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return RecordedClip(video_path, metadata_path, frames, fps_writer, duration_wall)


def upload_clip(clip: RecordedClip, bucket_name: str, camera_id: str) -> list[str]:
    """Upload video + metadata to gs://bucket/clips/<camera_id>/, return GCS URIs."""
    from google.cloud import storage

    client = storage.Client(project=os.environ.get("GCP_PROJECT_ID") or None)
    bucket = client.bucket(bucket_name)
    uris = []
    for path in (clip.video_path, clip.metadata_path):
        blob = bucket.blob(f"clips/{camera_id}/{path.name}")
        blob.upload_from_filename(str(path))
        uris.append(f"gs://{bucket_name}/{blob.name}")
    return uris


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="Camera id, e.g. tv516")
    parser.add_argument("--seconds", type=float, default=75.0)
    parser.add_argument("--out", type=Path, default=Path("data/captures"))
    parser.add_argument("--upload", action="store_true", help="Push to $GCS_BUCKET")
    args = parser.parse_args(argv)

    camera = get_camera(args.camera)
    print(f"recording {args.camera}: {camera.name} ({camera.place}, {camera.direction})")
    clip = record_clip(camera, args.seconds, args.out)
    print(
        f"wrote {clip.video_path} ({clip.frames} frames @ {clip.fps_writer} fps, "
        f"{clip.duration_wall_s:.0f}s wall) + {clip.metadata_path.name}"
    )
    if args.upload:
        bucket = os.environ.get("GCS_BUCKET", "")
        if not bucket:
            raise SystemExit("--upload needs GCS_BUCKET in .env")
        for uri in upload_clip(clip, bucket, camera.camera_id):
            print(f"uploaded {uri}")


if __name__ == "__main__":
    main()
