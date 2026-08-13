"""Extract still frames from a video at given timestamps (QC for rendered clips).

Usage:
    uv run python scripts/extract_frames.py <video> <seconds>... [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video: Path, seconds: list[float], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    written = []
    try:
        for t in seconds:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok:
                print(f"no frame at {t}s (past end?)")
                continue
            out = out_dir / f"{video.stem}_{t:g}s.png"
            cv2.imwrite(str(out), frame)
            written.append(out)
            print(out)
    finally:
        capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("seconds", type=float, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/frames"))
    args = parser.parse_args()
    extract(args.video, args.seconds, args.out_dir)


if __name__ == "__main__":
    main()
