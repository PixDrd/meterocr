"""Per-frame inference: align, segment, normalize, classify, combine."""

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from meterocr.align import align_meter
from meterocr.features import extract_hog_features
from meterocr.normalize import normalize_digit
from meterocr.segment import extract_digit_cells
from meterocr.types import (
    DigitPrediction,
    FrameMeta,
    HOGConfig,
    MeterConfig,
    NormalizationConfig,
    PredictionResult,
)
import uuid


def predict_meter_reading(
    image_path: Path,
    meter_config: MeterConfig,
    model_bundle: dict[str, Any],
    timestamp: datetime | None = None,
) -> PredictionResult:
    """Run inference on a single frame image file.

    Args:
        image_path: Path to the frame image.
        meter_config: Meter configuration.
        model_bundle: Loaded model bundle dict.
        timestamp: Optional timestamp; defaults to now.

    Returns:
        PredictionResult with per-digit predictions and combined reading.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    ts = timestamp or datetime.now()
    frame_id = f"{meter_config.meter_id}_{image_path.stem}_{ts.strftime('%Y%m%d_%H%M%S')}"
    return predict_meter_reading_from_array(frame_bgr, frame_id, ts, meter_config, model_bundle)


def predict_meter_reading_from_array(
    frame_bgr: np.ndarray,
    frame_id: str,
    timestamp: datetime,
    meter_config: MeterConfig,
    model_bundle: dict[str, Any],
) -> PredictionResult:
    """Run inference on a BGR image array.

    Args:
        frame_bgr: Full camera frame in BGR.
        frame_id: Unique identifier for this frame.
        timestamp: Frame capture timestamp.
        meter_config: Meter configuration.
        model_bundle: Loaded model bundle dict.

    Returns:
        PredictionResult with per-digit predictions and combined reading.
    """
    hog_cfg: HOGConfig = model_bundle["hog_cfg"]
    norm_cfg: NormalizationConfig = model_bundle["normalization_cfg"]
    classifier = model_bundle["classifier"]
    scaler = model_bundle.get("scaler")

    frame_meta = FrameMeta(
        frame_id=frame_id,
        meter_id=meter_config.meter_id,
        timestamp=timestamp,
        source_path=Path(frame_id),
    )

    aligned = align_meter(frame_bgr, meter_config)
    cells = extract_digit_cells(aligned, meter_config, frame_meta)

    digit_predictions: list[DigitPrediction] = []
    for cell in cells:
        norm_result = normalize_digit(
            cell.image_bgr,
            meter_config.threshold_mode,
            meter_config.invert_binary,
            norm_cfg,
        )
        img = cv2.resize(
            norm_result.normalized,
            (hog_cfg.image_width, hog_cfg.image_height),
        )
        digit, confidence, margin = classify_normalized_digit(img, classifier, scaler, hog_cfg)
        digit_predictions.append(
            DigitPrediction(
                position=cell.position,
                digit=digit,
                confidence=confidence,
                margin=margin,
            )
        )

    confidences = [d.confidence for d in digit_predictions]
    raw_reading = combine_digits_to_reading([d.digit for d in digit_predictions])

    return PredictionResult(
        frame_id=frame_id,
        meter_id=meter_config.meter_id,
        timestamp=timestamp,
        raw_reading=raw_reading,
        digits=digit_predictions,
        min_confidence=float(min(confidences)) if confidences else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else 0.0,
    )


def classify_normalized_digit(
    normalized_img: np.ndarray,
    classifier: Any,
    scaler: Any | None,
    hog_cfg: HOGConfig,
) -> tuple[int, float, float]:
    """Classify a single normalized digit image.

    Args:
        normalized_img: Grayscale normalized digit image.
        classifier: Fitted scikit-learn classifier.
        scaler: Fitted scaler or None.
        hog_cfg: HOG configuration.

    Returns:
        (digit, confidence, margin) where for LinearSVC, confidence == margin
        (the decision function score for the winning class).
    """
    features = extract_hog_features(normalized_img, hog_cfg).reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)

    predicted = int(classifier.predict(features)[0])

    # Use decision_function margin as confidence for LinearSVC / kNN-compatible
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(features)[0]
        margin = float(np.max(scores))
        confidence = margin
    elif hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(features)[0]
        confidence = float(np.max(proba))
        margin = confidence
    else:
        confidence = 1.0
        margin = 1.0

    return predicted, confidence, margin


def combine_digits_to_reading(digits: list[int]) -> str:
    """Combine a list of digit integers into a reading string.

    Args:
        digits: Ordered list of digit values (0-9).

    Returns:
        Concatenated digit string, e.g. [0, 3, 5, 0, 6] -> '03506'.
    """
    return "".join(str(d) for d in digits)
