"""Tests for predict.py."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from meterocr.predict import (
    classify_normalized_digit,
    combine_digits_to_reading,
    predict_meter_reading_from_array,
)
from meterocr.types import HOGConfig


def _make_mock_bundle(hog_cfg: HOGConfig, norm_cfg, digit: int = 3):
    """Build a minimal mock model bundle that always predicts digit."""
    clf = MagicMock()
    clf.predict.return_value = [digit]
    clf.decision_function.return_value = np.array([[0.1] * 10])
    clf.decision_function.return_value[0][digit] = 2.5

    return {
        "classifier": clf,
        "scaler": None,
        "hog_cfg": hog_cfg,
        "normalization_cfg": norm_cfg,
        "class_labels": list(range(10)),
        "version": "0.1.0",
    }


def test_predict_returns_prediction_result(meter_config, hog_cfg, norm_cfg):
    bundle = _make_mock_bundle(hog_cfg, norm_cfg, digit=5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = predict_meter_reading_from_array(
        frame_bgr=frame,
        frame_id="test_frame",
        timestamp=datetime.now(),
        meter_config=meter_config,
        model_bundle=bundle,
    )
    assert result.meter_id == "TEST"
    assert len(result.digits) == len(meter_config.digit_boxes)
    assert len(result.raw_reading) == len(meter_config.digit_boxes)


def test_predict_reading_contains_predicted_digit(meter_config, hog_cfg, norm_cfg):
    bundle = _make_mock_bundle(hog_cfg, norm_cfg, digit=7)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = predict_meter_reading_from_array(
        frame_bgr=frame,
        frame_id="f",
        timestamp=datetime.now(),
        meter_config=meter_config,
        model_bundle=bundle,
    )
    assert all(d.digit == 7 for d in result.digits)
    assert result.raw_reading == "77777"


def test_combine_digits_to_reading():
    assert combine_digits_to_reading([0, 3, 5, 0, 6]) == "03506"
    assert combine_digits_to_reading([]) == ""
    assert combine_digits_to_reading([9]) == "9"


def test_classify_normalized_digit_returns_tuple(hog_cfg):
    clf = MagicMock()
    clf.predict.return_value = [4]
    clf.decision_function.return_value = np.array([[0.1] * 10])
    clf.decision_function.return_value[0][4] = 3.0

    img = np.zeros((64, 40), dtype=np.uint8)
    digit, confidence, margin = classify_normalized_digit(img, clf, None, hog_cfg)
    assert isinstance(digit, int)
    assert isinstance(confidence, float)
    assert isinstance(margin, float)
