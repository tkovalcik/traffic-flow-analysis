"""Tests for scene onboarding (fake segmenter — no model download in CI)."""

import numpy as np
import pytest

from src.perception.speed.scene_mask import (
    onboard_camera,
    render_overlay,
    road_mask_from_labels,
)

cv2 = pytest.importorskip("cv2")

ID2LABEL = {0: "sky", 1: "tree", 2: "road", 3: "signboard"}


def fake_label_map(h=120, w=160):
    """Sky on top, trees left, road bottom-right, a sign patch inside the trees."""
    labels = np.zeros((h, w), dtype=np.int32)
    labels[40:, :] = 1
    labels[40:, 60:] = 2
    labels[50:60, 20:30] = 3
    return labels


def test_road_mask_covers_road_and_excludes_trees_and_signs():
    mask = road_mask_from_labels(fake_label_map(), ID2LABEL, dilate_px=0)
    assert mask[100, 120] == 255  # road
    assert mask[100, 20] == 0  # trees
    assert mask[55, 25] == 0  # sign
    assert mask[10, 80] == 0  # sky


def test_dilation_grows_mask_at_boundary():
    tight = road_mask_from_labels(fake_label_map(), ID2LABEL, dilate_px=0)
    grown = road_mask_from_labels(fake_label_map(), ID2LABEL, dilate_px=8)
    assert int((grown > 0).sum()) > int((tight > 0).sum())
    assert grown[100, 57] == 255  # just left of the road boundary, now included


def test_onboard_camera_writes_mask_and_overlay(tmp_path):
    image = np.full((120, 160, 3), 120, dtype=np.uint8)

    def segmenter(_image):
        return fake_label_map(), ID2LABEL

    mask_path, mask, overlay = onboard_camera(
        image, "camtest", segmenter=segmenter, mask_dir=tmp_path
    )
    assert mask_path == tmp_path / "camtest.png"
    assert mask_path.exists()
    reloaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(reloaded, mask)
    assert overlay.shape == image.shape


def test_overlay_renders_legend_without_crashing():
    image = np.full((120, 160, 3), 120, dtype=np.uint8)
    overlay = render_overlay(image, fake_label_map(), ID2LABEL)
    assert not np.array_equal(overlay, image)  # something was drawn
