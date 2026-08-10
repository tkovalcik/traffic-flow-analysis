"""Camera onboarding: one-time semantic scene segmentation → road-surface mask.

Runs a semantic segmentation model (SegFormer fine-tuned on ADE20K, which has a
first-class `road` label) over the camera's clean median frame, and saves:

- a binary road mask (configs/scene_masks/<camera>.png) that downstream tools
  use to ignore signs, bridge railings, and vegetation, and
- a color-coded overlay + legend for human review.

This is deliberately the expensive-but-rare step: run once when onboarding a
camera (and again if the camera is re-aimed). Lane-marking detection then only
looks where pavement actually is.

Usage:
    uv run python -m src.perception.speed.scene_mask \
        outputs/review/<median>.jpg --camera tva43
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"
MASK_DIR = Path("configs/scene_masks")
# ADE20K labels counted as drivable pavement. `road` does the heavy lifting;
# highways under overpass shadow sometimes classify as `path`/`sidewalk`.
ROAD_LABELS = {"road", "route", "path", "sidewalk", "pavement"}

# Fixed BGR colors for the review overlay's most relevant classes.
OVERLAY_COLORS = {
    "road": (80, 200, 80),
    "tree": (40, 90, 40),
    "sky": (200, 160, 60),
    "bridge": (60, 120, 220),
    "signboard": (60, 60, 230),
    "building": (150, 120, 120),
    "grass": (90, 170, 90),
    "car": (200, 80, 200),
}

Segmenter = Callable[[np.ndarray], tuple[np.ndarray, dict[int, str]]]


def _hf_segmenter(image: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """Default segmenter: SegFormer/ADE20K via transformers (downloads once)."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        outputs = model(**processor(images=rgb, return_tensors="pt"))
    upsampled = torch.nn.functional.interpolate(
        outputs.logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    label_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return label_map, id2label


def road_mask_from_labels(
    label_map: np.ndarray,
    id2label: dict[int, str],
    dilate_px: int = 6,
) -> np.ndarray:
    """Binary mask (255=drivable) from a semantic label map, slightly dilated
    so lane paint on the mask boundary isn't clipped."""
    road_ids = {i for i, name in id2label.items() if name in ROAD_LABELS}
    mask = np.isin(label_map, list(road_ids)).astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel)
    return mask


def render_overlay(
    image: np.ndarray, label_map: np.ndarray, id2label: dict[int, str]
) -> np.ndarray:
    """Half-transparent class coloring + legend of the classes present."""
    out = image.copy()
    color_layer = np.zeros_like(image)
    present: list[tuple[str, tuple[int, int, int], float]] = []
    total = label_map.size
    for class_id in np.unique(label_map):
        name = id2label.get(int(class_id), f"class{class_id}")
        color = OVERLAY_COLORS.get(name)
        share = float((label_map == class_id).sum()) / total
        if color is None and share >= 0.02:
            color = (127, 127, 127)
        if color is None:
            continue
        color_layer[label_map == class_id] = color
        if share >= 0.01:
            present.append((name, color, share))
    cv2.addWeighted(color_layer, 0.45, out, 0.55, 0, out)
    present.sort(key=lambda item: -item[2])
    for i, (name, color, share) in enumerate(present[:8]):
        y = 24 + 22 * i
        cv2.rectangle(out, (10, y - 12), (26, y + 4), color, -1)
        cv2.putText(
            out,
            f"{name} {share:.0%}",
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"{name} {share:.0%}",
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def onboard_camera(
    median_image: np.ndarray,
    camera_id: str,
    segmenter: Segmenter | None = None,
    mask_dir: Path = MASK_DIR,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Segment the scene, save the road mask, return (mask_path, mask, overlay)."""
    label_map, id2label = (segmenter or _hf_segmenter)(median_image)
    mask = road_mask_from_labels(label_map, id2label)
    overlay = render_overlay(median_image, label_map, id2label)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{camera_id}.png"
    cv2.imwrite(str(mask_path), mask)
    return mask_path, mask, overlay


def main() -> None:
    from src.perception.calibrate_line import review_stamp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Clean median-frame image")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--review-dir", type=Path, default=Path("outputs/review"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    mask_path, mask, overlay = onboard_camera(image, args.camera)
    road_share = float((mask > 0).sum()) / mask.size

    stamp = review_stamp()
    out_dir = args.review_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{args.camera}_scene_overlay_{stamp}.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"road mask: {mask_path} ({road_share:.0%} of frame is drivable)")
    print(f"overlay:   {overlay_path}")


if __name__ == "__main__":
    main()
