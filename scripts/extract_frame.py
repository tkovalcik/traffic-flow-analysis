"""Extract still frames from a video at given fractions of its duration.

Used for eyeballing camera views and calibrating counting lines.

Usage:
    uv run python scripts/extract_frame.py <video> --at 0.25 0.5 0.75 --out-dir outputs/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(video: Path, fractions: list[float], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = []
    for fraction in fractions:
        index = max(0, min(total - 1, int(total * fraction)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        out = out_dir / f"{video.stem}_f{index:05d}.jpg"
        cv2.imwrite(str(out), frame)
        written.append(out)
    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=float, nargs="+", default=[0.5])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/frames"))
    args = parser.parse_args()
    for path in extract(args.video, args.at, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
