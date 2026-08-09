"""Crop a region from an image (normalized coords), optionally upscaled.

Used to zoom into review images, e.g. reading lane-dash positions precisely.

Usage:
    uv run python scripts/crop_image.py image.jpg --box 0.4,0.5,0.9,1.0 --scale 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def crop(image_path: Path, box: tuple[float, float, float, float], scale: float) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"cannot read {image_path}")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    region = image[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
    if scale != 1.0:
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = image_path.with_name(f"{image_path.stem}_crop_{x1:.2f}_{y1:.2f}{image_path.suffix}")
    cv2.imwrite(str(out), region)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--box", required=True, help="x1,y1,x2,y2 normalized")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    box = tuple(float(v) for v in args.box.split(","))
    print(crop(args.image, box, args.scale))


if __name__ == "__main__":
    main()
