"""Tests for train.py and model_io.py."""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from meterocr.model_io import load_model_bundle, save_model_bundle
from meterocr.train import build_training_dataframe, train_and_save_model
from meterocr.types import HOGConfig, NormalizationConfig, TrainingConfig


def _write_synthetic_samples(tmp_path: Path, hog_cfg: HOGConfig) -> Path:
    """Create a minimal samples.csv with 20 synthetic digit images."""
    norm_dir = tmp_path / "normalized"
    norm_dir.mkdir()
    samples_csv = tmp_path / "samples.csv"

    rows = []
    for i in range(20):
        digit = i % 10
        img = np.zeros((hog_cfg.image_height, hog_cfg.image_width), dtype=np.uint8)
        # Draw a unique rectangle per digit class
        img[5 + digit : 55 + digit, 5:35] = 255
        img_path = norm_dir / f"frame{i}_pos0_norm.png"
        cv2.imwrite(str(img_path), img)
        rows.append({
            "frame_id": f"frame{i}",
            "timestamp": "2024-01-01T12:00:00",
            "meter_id": "M1",
            "position": 0,
            "digit_label": digit,
            "full_reading": str(digit) * 5,
            "raw_cell_path": str(img_path),
            "normalized_path": str(img_path),
            "quality": "ok",
        })

    fieldnames = [
        "frame_id", "timestamp", "meter_id", "position", "digit_label",
        "full_reading", "raw_cell_path", "normalized_path", "quality",
    ]
    with samples_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return samples_csv


def test_train_and_save_bundle(tmp_path, hog_cfg, norm_cfg, training_cfg):
    samples_csv = _write_synthetic_samples(tmp_path, hog_cfg)
    model_path = tmp_path / "model.joblib"

    result = train_and_save_model(
        samples_csv=samples_csv,
        model_path=model_path,
        hog_cfg=hog_cfg,
        normalization_cfg=norm_cfg,
        training_cfg=training_cfg,
    )

    assert model_path.exists()
    assert result["train_count"] > 0
    assert result["feature_dim"] > 0


def test_load_model_bundle(tmp_path, hog_cfg, norm_cfg, training_cfg):
    samples_csv = _write_synthetic_samples(tmp_path, hog_cfg)
    model_path = tmp_path / "model.joblib"

    train_and_save_model(
        samples_csv=samples_csv,
        model_path=model_path,
        hog_cfg=hog_cfg,
        normalization_cfg=norm_cfg,
        training_cfg=training_cfg,
    )

    bundle = load_model_bundle(model_path)
    assert "classifier" in bundle
    assert "hog_cfg" in bundle
    assert "normalization_cfg" in bundle
    assert "class_labels" in bundle


def test_load_model_bundle_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_bundle(tmp_path / "does_not_exist.joblib")
