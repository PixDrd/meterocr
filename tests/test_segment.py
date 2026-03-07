"""Tests for segment.py."""

import numpy as np
import pytest

from meterocr.segment import crop_digit_cell, extract_digit_cells
from meterocr.types import FrameMeta
from datetime import datetime
from pathlib import Path


def test_extract_digit_cells_count(meter_config, blank_meter_image):
    frame_meta = FrameMeta(
        frame_id="f1",
        meter_id="TEST",
        timestamp=datetime.now(),
        source_path=Path("test.png"),
    )
    cells = extract_digit_cells(blank_meter_image, meter_config, frame_meta)
    assert len(cells) == len(meter_config.digit_boxes)


def test_extract_digit_cells_positions_are_zero_based(meter_config, blank_meter_image):
    frame_meta = FrameMeta(
        frame_id="f1",
        meter_id="TEST",
        timestamp=datetime.now(),
        source_path=Path("test.png"),
    )
    cells = extract_digit_cells(blank_meter_image, meter_config, frame_meta)
    positions = [c.position for c in cells]
    assert positions == list(range(len(meter_config.digit_boxes)))


def test_extract_digit_cells_images_are_bgr(meter_config, blank_meter_image):
    frame_meta = FrameMeta(
        frame_id="f1",
        meter_id="TEST",
        timestamp=datetime.now(),
        source_path=Path("test.png"),
    )
    cells = extract_digit_cells(blank_meter_image, meter_config, frame_meta)
    for cell in cells:
        assert cell.image_bgr.ndim == 3
        assert cell.image_bgr.shape[2] == 3


def test_crop_digit_cell_clips_to_image_bounds():
    img = np.zeros((120, 500, 3), dtype=np.uint8)
    # Request a region that extends past the right edge
    result = crop_digit_cell(img, x=490, y=0, w=50, h=120)
    assert result.shape[1] <= 10  # clipped to 10 pixels wide


def test_crop_digit_cell_correct_size():
    img = np.ones((120, 500, 3), dtype=np.uint8) * 200
    result = crop_digit_cell(img, x=10, y=5, w=80, h=100)
    assert result.shape == (100, 80, 3)
