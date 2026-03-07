"""Evaluate a trained digit classifier and save reports."""

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from meterocr.features import extract_feature_matrix
from meterocr.types import HOGConfig


def evaluate_digit_classifier(
    val_df: pd.DataFrame,
    classifier: Any,
    scaler: Any | None,
    hog_cfg: HOGConfig,
    reports_dir: Path,
) -> dict[str, Any]:
    """Evaluate a classifier on a validation DataFrame.

    Args:
        val_df: Validation samples DataFrame.
        classifier: Fitted scikit-learn classifier.
        scaler: Fitted scaler or None.
        hog_cfg: HOG configuration.
        reports_dir: Directory to save report files.

    Returns:
        Dict with keys: digit_accuracy, per_meter, per_position, y_true, y_pred.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    images = []
    y_true = []
    valid_rows = []
    for _, row in val_df.iterrows():
        img = cv2.imread(str(row["normalized_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (hog_cfg.image_width, hog_cfg.image_height))
        images.append(img)
        y_true.append(int(row["digit_label"]))
        valid_rows.append(row)

    if not images:
        return {"digit_accuracy": 0.0, "per_meter": {}, "per_position": {}, "y_true": [], "y_pred": []}

    X = extract_feature_matrix(images, hog_cfg)
    if scaler is not None:
        X = scaler.transform(X)

    y_pred = list(classifier.predict(X))
    digit_accuracy = float(accuracy_score(y_true, y_pred))

    save_confusion_report(y_true, y_pred, reports_dir, "overall")

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    valid_df["y_pred"] = y_pred

    per_meter: dict[str, float] = {}
    for meter_id, group in valid_df.groupby("meter_id"):
        per_meter[str(meter_id)] = float(
            accuracy_score(group["digit_label"].tolist(), group["y_pred"].tolist())
        )

    per_position: dict[str, float] = {}
    for pos, group in valid_df.groupby("position"):
        per_position[str(pos)] = float(
            accuracy_score(group["digit_label"].tolist(), group["y_pred"].tolist())
        )

    _save_per_meter_report(per_meter, reports_dir)
    _save_per_position_report(per_position, reports_dir)
    save_misclassified_examples(valid_df, y_pred, reports_dir)

    return {
        "digit_accuracy": digit_accuracy,
        "per_meter": per_meter,
        "per_position": per_position,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_report(
    y_true: list[int],
    y_pred: list[int],
    reports_dir: Path,
    label: str,
) -> None:
    """Save a confusion matrix CSV to reports_dir.

    Args:
        y_true: Ground-truth digit labels.
        y_pred: Predicted digit labels.
        reports_dir: Directory to save the report.
        label: Suffix for the filename.
    """
    all_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    out_path = reports_dir / f"confusion_{label}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [str(c) for c in all_labels])
        for i, row in enumerate(cm):
            writer.writerow([str(all_labels[i])] + [str(v) for v in row])


def save_misclassified_examples(
    val_df: pd.DataFrame,
    y_pred: list[int],
    reports_dir: Path,
    limit: int = 100,
) -> None:
    """Save metadata for misclassified samples to a CSV.

    Args:
        val_df: Validation DataFrame (must have digit_label, normalized_path, etc.).
        y_pred: Predicted labels aligned with val_df rows.
        reports_dir: Directory to save the report.
        limit: Maximum number of misclassified samples to save.
    """
    df = val_df.copy()
    df["y_pred"] = y_pred
    df["digit_label"] = df["digit_label"].astype(int)
    misclassified = df[df["digit_label"] != df["y_pred"]].head(limit)
    out_path = reports_dir / "misclassified.csv"
    misclassified.to_csv(out_path, index=False)


def _save_per_meter_report(per_meter: dict[str, float], reports_dir: Path) -> None:
    out_path = reports_dir / "accuracy_per_meter.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["meter_id", "accuracy"])
        for k, v in sorted(per_meter.items()):
            writer.writerow([k, f"{v:.4f}"])


def _save_per_position_report(per_position: dict[str, float], reports_dir: Path) -> None:
    out_path = reports_dir / "accuracy_per_position.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "accuracy"])
        for k, v in sorted(per_position.items(), key=lambda x: int(x[0])):
            writer.writerow([k, f"{v:.4f}"])
