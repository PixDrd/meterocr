"""Tests for align.py."""

import numpy as np
import pytest

from meterocr.align import (
    align_meter,
    apply_translation,
    crop_meter,
    estimate_translation,
)
from meterocr.types import DigitBox, MeterConfig


def _make_config(crop_box=None, perspective_pts=None, max_translation=0):
    return MeterConfig(
        meter_id="T",
        aligned_width=500,
        aligned_height=120,
        digit_boxes=[DigitBox(x=0, y=0, w=92, h=120)],
        threshold_mode="otsu",
        invert_binary=True,
        crop_source_box=crop_box,
        perspective_src_points=perspective_pts,
        max_translation_px=max_translation,
    )


def test_crop_meter_returns_expected_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = _make_config(crop_box=DigitBox(x=0, y=0, w=640, h=480))
    result = crop_meter(frame, cfg)
    assert result.shape == (120, 500, 3)


def test_crop_meter_no_crop_box_resizes_whole_frame():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cfg = _make_config()  # no crop_source_box
    result = crop_meter(frame, cfg)
    assert result.shape == (120, 500, 3)


def test_align_meter_uses_crop_when_no_perspective():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = _make_config(crop_box=DigitBox(x=10, y=10, w=500, h=120))
    result = align_meter(frame, cfg)
    assert result.shape == (120, 500, 3)


def test_apply_translation_zero_shift():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = apply_translation(img, 0, 0)
    np.testing.assert_array_equal(result, img)


def test_apply_translation_preserves_shape():
    img = np.zeros((120, 500, 3), dtype=np.uint8)
    result = apply_translation(img, 3, -2)
    assert result.shape == img.shape


def test_estimate_translation_returns_integers():
    ref = np.zeros((120, 500), dtype=np.uint8)
    cand = np.zeros((120, 500), dtype=np.uint8)
    dx, dy = estimate_translation(ref, cand, max_shift_px=4)
    assert isinstance(dx, int)
    assert isinstance(dy, int)
    assert abs(dx) <= 4
    assert abs(dy) <= 4
