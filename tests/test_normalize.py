"""Tests for normalize.py."""

import numpy as np
import pytest

from meterocr.normalize import (
    center_and_resize,
    find_main_digit_bbox,
    normalize_digit,
    remove_small_components,
    threshold_digit,
    to_gray,
)
from meterocr.types import NormalizationConfig


def _cfg(**kwargs) -> NormalizationConfig:
    defaults = dict(
        blur_kernel=3,
        min_component_area=8,
        bbox_margin_px=2,
        output_width=40,
        output_height=64,
        foreground="white",
    )
    defaults.update(kwargs)
    return NormalizationConfig(**defaults)


def test_normalize_digit_output_size(norm_cfg):
    cell = np.zeros((120, 92, 3), dtype=np.uint8)
    cell[20:100, 20:70] = 200  # bright rectangle simulating digit
    result = normalize_digit(cell, "otsu", invert_binary=True, cfg=norm_cfg)
    assert result.normalized.shape == (norm_cfg.output_height, norm_cfg.output_width)


def test_normalize_digit_failure_on_blank_image(norm_cfg):
    # All-black image with invert_binary=False: Otsu threshold on constant image
    # keeps everything black → no foreground blob → success=False.
    cell = np.zeros((120, 92, 3), dtype=np.uint8)
    result = normalize_digit(cell, "otsu", invert_binary=False, cfg=norm_cfg)
    assert result.success is False
    assert result.reason != ""


def test_to_gray_on_bgr():
    bgr = np.zeros((64, 40, 3), dtype=np.uint8)
    gray = to_gray(bgr)
    assert gray.ndim == 2


def test_to_gray_passthrough_on_grayscale():
    gray = np.zeros((64, 40), dtype=np.uint8)
    result = to_gray(gray)
    assert result.ndim == 2


def test_remove_small_components_removes_noise():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[5:7, 5:7] = 255   # tiny 2x2 region
    img[20:80, 20:80] = 255  # large region
    result = remove_small_components(img, min_area=10)
    assert result[5, 5] == 0   # small component removed
    assert result[50, 50] == 255  # large component kept


def test_find_main_digit_bbox_returns_largest():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10:20, 10:20] = 255   # 100 px
    img[50:90, 50:90] = 255   # 1600 px
    bbox = find_main_digit_bbox(img, min_area=1)
    assert bbox is not None
    x, y, w, h = bbox
    assert w >= 38 and h >= 38  # the larger region


def test_find_main_digit_bbox_none_when_empty():
    img = np.zeros((100, 100), dtype=np.uint8)
    bbox = find_main_digit_bbox(img, min_area=1)
    assert bbox is None


def test_center_and_resize_output_size():
    digit = np.ones((50, 20), dtype=np.uint8) * 255
    result = center_and_resize(digit, (40, 64))
    assert result.shape == (64, 40)


def test_center_and_resize_preserves_aspect_ratio():
    # Tall digit should fit vertically, leaving horizontal padding
    digit = np.ones((100, 10), dtype=np.uint8) * 255
    result = center_and_resize(digit, (40, 64))
    assert result.shape == (64, 40)
    # Should not be all zeros
    assert result.max() > 0
