"""Shared test fixtures."""

import numpy as np
import pytest

from meterocr.types import (
    DigitBox,
    HOGConfig,
    MeterConfig,
    NormalizationConfig,
    TrainingConfig,
)


@pytest.fixture
def meter_config() -> MeterConfig:
    """A simple meter config for testing."""
    return MeterConfig(
        meter_id="TEST",
        aligned_width=500,
        aligned_height=120,
        digit_boxes=[
            DigitBox(x=0, y=0, w=92, h=120),
            DigitBox(x=96, y=0, w=92, h=120),
            DigitBox(x=192, y=0, w=92, h=120),
            DigitBox(x=288, y=0, w=92, h=120),
            DigitBox(x=384, y=0, w=92, h=120),
        ],
        threshold_mode="otsu",
        invert_binary=True,
        crop_source_box=DigitBox(x=0, y=0, w=500, h=120),
        inner_pad_x=2,
        inner_pad_y=2,
    )


@pytest.fixture
def hog_cfg() -> HOGConfig:
    return HOGConfig(
        image_width=40,
        image_height=64,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
    )


@pytest.fixture
def norm_cfg() -> NormalizationConfig:
    return NormalizationConfig(
        blur_kernel=3,
        min_component_area=8,
        bbox_margin_px=2,
        output_width=40,
        output_height=64,
        foreground="white",
    )


@pytest.fixture
def training_cfg() -> TrainingConfig:
    return TrainingConfig(
        model_type="knn",
        test_size=0.2,
        group_by="frame",
        random_state=42,
        knn_neighbors=1,
        knn_weights="distance",
        svc_c=1.0,
        use_standard_scaler=False,
    )


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A blank BGR frame of 640x480."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def blank_meter_image() -> np.ndarray:
    """A blank aligned meter image of 500x120."""
    return np.zeros((120, 500, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_digit_image() -> np.ndarray:
    """A synthetic 40x64 grayscale image with a white rectangle (fake digit)."""
    img = np.zeros((64, 40), dtype=np.uint8)
    img[10:54, 8:32] = 255  # white block in the middle
    return img
