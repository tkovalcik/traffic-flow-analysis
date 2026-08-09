"""Empty-road extraction: the temporal median of many frames removes traffic.

A moving vehicle occupies any given pixel only briefly; across enough sampled
frames the median value at each pixel is pavement, not car. The resulting
clean road image is the canvas for lane-marking work (dash detection, the
perspective ruler, eventually the ground-plane homography fit).

Usage:
    uv run python -m src.perception.speed.median_frame \
        data/captures/<clip>.mp4 --samples 120 --out outputs/review/<name>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def median_frame(video: Path, samples: int = 120) -> np.ndarray:
    """Median image over `samples` frames spread evenly across the clip."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frame count from {video}")
    indices = np.linspace(0, total - 1, min(samples, total)).astype(int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no readable frames in {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    image = median_frame(args.video, args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    print(f"median of {args.samples} samples -> {args.out}")


if __name__ == "__main__":
    main()
