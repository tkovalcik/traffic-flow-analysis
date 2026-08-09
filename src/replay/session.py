"""Capture session: record all configured cameras simultaneously, one command.

The golden-data workflow: every camera in configs/cameras.json gets recorded
for the same interval (perfect cross-camera comparability), each clip carries
full provenance metadata, everything is archived to the GCS bucket, and a
session manifest ties the set together.

Usage:
    uv run python -m src.replay.session --minutes 15 --upload           # all configured cams
    uv run python -m src.replay.session --minutes 2 --cameras tva43     # subset
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.perception.camera_config import DEFAULT_CONFIG_PATH
from src.perception.cameras import CameraInfo, get_camera
from src.replay.record import RecordedClip, record_clip, upload_clip

MANIFEST_SCHEMA_VERSION = 1


def configured_camera_ids(config_path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    data = json.loads(config_path.read_text())
    return sorted(k for k in data if not k.startswith("_"))


def run_session(
    cameras: list[CameraInfo],
    seconds: float,
    out_dir: Path,
    bucket: str | None = None,
    reconnect_attempts: int = 5,
    backoff_s: float = 2.0,
) -> dict:
    """Record every camera concurrently; return the session manifest dict."""
    session_id = datetime.now(UTC).strftime("session_%Y%m%dT%H%M%SZ")
    session_dir = out_dir / session_id
    started = datetime.now(UTC)

    def capture_one(camera: CameraInfo) -> tuple[str, RecordedClip | None, str]:
        try:
            clip = record_clip(
                camera,
                seconds,
                session_dir,
                reconnect_attempts=reconnect_attempts,
                backoff_s=backoff_s,
            )
            uris = upload_clip(clip, bucket, camera.camera_id) if bucket else []
            return camera.camera_id, clip, ",".join(uris)
        except Exception as exc:  # one dead camera must not sink the session
            return camera.camera_id, None, f"ERROR {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max(1, len(cameras))) as pool:
        results = list(pool.map(capture_one, cameras))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "session_id": session_id,
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(UTC).isoformat(),
        "requested_seconds": seconds,
        "clips": [
            {
                "camera_id": camera_id,
                "ok": clip is not None,
                "video": str(clip.video_path) if clip else None,
                "metadata": str(clip.metadata_path) if clip else None,
                "frames": clip.frames if clip else 0,
                "gcs": note if clip else None,
                "error": None if clip else note,
            }
            for camera_id, clip, note in results
        ],
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if bucket:
        from google.cloud import storage

        client = storage.Client(project=os.environ.get("GCP_PROJECT_ID") or None)
        blob = client.bucket(bucket).blob(f"sessions/{session_id}/manifest.json")
        blob.upload_from_filename(str(manifest_path))

    return manifest


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--cameras", nargs="+", help="Camera ids (default: all configured)")
    parser.add_argument("--out", type=Path, default=Path("data/captures"))
    parser.add_argument("--upload", action="store_true", help="Archive to $GCS_BUCKET")
    args = parser.parse_args(argv)

    camera_ids = args.cameras or configured_camera_ids()
    cameras = [get_camera(cid) for cid in camera_ids]
    bucket = os.environ.get("GCS_BUCKET", "") if args.upload else None
    if args.upload and not bucket:
        raise SystemExit("--upload needs GCS_BUCKET in .env")

    print(f"capture session: {', '.join(camera_ids)} for {args.minutes} min")
    manifest = run_session(cameras, args.minutes * 60, args.out, bucket)
    for clip in manifest["clips"]:
        status = f"{clip['frames']} frames" if clip["ok"] else clip["error"]
        print(f"  {clip['camera_id']}: {status}")
    print(f"manifest: {args.out}/{manifest['session_id']}/manifest.json")


if __name__ == "__main__":
    main()
