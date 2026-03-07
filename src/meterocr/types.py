"""Core data types for the water meter OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ImageU8 = NDArray[np.uint8]
FeatureVectorF32 = NDArray[np.float32]


@dataclass(frozen=True)
class DigitBox:
    """Bounding box for a digit cell within an aligned meter image."""

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class MeterConfig:
    """Per-meter geometry and processing configuration."""

    meter_id: str
    aligned_width: int
    aligned_height: int
    digit_boxes: list[DigitBox]
    threshold_mode: Literal["otsu", "adaptive"]
    invert_binary: bool
    crop_source_box: DigitBox | None = None
    perspective_src_points: list[tuple[float, float]] | None = None
    alignment_reference_path: Path | None = None
    max_translation_px: int = 0
    inner_pad_x: int = 0
    inner_pad_y: int = 0


@dataclass(frozen=True)
class HOGConfig:
    """Configuration for HOG feature extraction."""

    image_width: int = 40
    image_height: int = 64
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (4, 4)
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = False


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for digit image normalization."""

    blur_kernel: int = 3
    min_component_area: int = 8
    bbox_margin_px: int = 2
    output_width: int = 40
    output_height: int = 64
    foreground: Literal["white", "black"] = "white"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""

    model_type: Literal["knn", "linear_svc"]
    test_size: float = 0.2
    group_by: Literal["frame", "date", "meter"] = "frame"
    random_state: int = 42
    knn_neighbors: int = 3
    knn_weights: Literal["uniform", "distance"] = "distance"
    svc_c: float = 1.0
    use_standard_scaler: bool = True


@dataclass(frozen=True)
class FrameMeta:
    """Metadata for a captured frame."""

    frame_id: str
    meter_id: str
    timestamp: datetime
    source_path: Path


@dataclass(frozen=True)
class DigitCell:
    """A single extracted digit cell from an aligned meter image."""

    frame_id: str
    meter_id: str
    position: int
    bbox: DigitBox
    image_bgr: ImageU8


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalizing a digit cell image."""

    normalized: ImageU8
    gray: ImageU8
    binary: ImageU8
    digit_bbox_xywh: tuple[int, int, int, int] | None
    success: bool
    reason: str


@dataclass(frozen=True)
class DigitSample:
    """A single labeled digit sample for training."""

    frame_id: str
    meter_id: str
    timestamp: datetime
    position: int
    digit_label: int
    full_reading: str
    raw_cell_path: Path
    normalized_path: Path
    quality: Literal["ok", "uncertain", "rejected"] = "ok"


@dataclass(frozen=True)
class DigitPrediction:
    """Prediction result for a single digit position."""

    position: int
    digit: int
    confidence: float
    margin: float
    top2_digits: tuple[int, int] | None = None
    top2_scores: tuple[float, float] | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Full prediction result for one meter frame."""

    frame_id: str
    meter_id: str
    timestamp: datetime
    raw_reading: str
    digits: list[DigitPrediction]
    min_confidence: float
    mean_confidence: float


@dataclass(frozen=True)
class StableReadingResult:
    """Temporally filtered reading result."""

    frame_id: str
    meter_id: str
    timestamp: datetime
    accepted: bool
    raw_reading: str
    stable_reading: str
    reason: str


@dataclass
class MeterState:
    """Mutable per-meter temporal state."""

    meter_id: str
    last_stable_reading: str | None = None
    last_stable_timestamp: datetime | None = None
    recent_predictions: list[PredictionResult] = field(default_factory=list)
