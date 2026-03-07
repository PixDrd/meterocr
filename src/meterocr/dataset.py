"""Manage sample storage, CSV persistence, and dataset loading."""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from meterocr.types import DigitCell, DigitSample, FrameMeta, ImageU8, NormalizationResult

_FRAMES_COLUMNS = [
    "frame_id",
    "timestamp",
    "meter_id",
    "source_path",
    "full_reading",
]

_SAMPLES_COLUMNS = [
    "frame_id",
    "timestamp",
    "meter_id",
    "position",
    "digit_label",
    "full_reading",
    "raw_cell_path",
    "normalized_path",
    "quality",
]


def append_frame_label(
    labels_csv: Path,
    frame_id: str,
    meter_id: str,
    timestamp: datetime,
    source_path: Path,
    full_reading: str,
) -> None:
    """Append one frame label row to the frames CSV.

    Creates the CSV with headers if it does not exist.

    Args:
        labels_csv: Path to frames.csv.
        frame_id: Unique frame identifier.
        meter_id: Meter identifier.
        timestamp: Capture timestamp.
        source_path: Path to the source image.
        full_reading: Full meter reading string (e.g. '03506').
    """
    _ensure_csv(labels_csv, _FRAMES_COLUMNS)
    row = {
        "frame_id": frame_id,
        "timestamp": timestamp.isoformat(),
        "meter_id": meter_id,
        "source_path": str(source_path),
        "full_reading": full_reading,
    }
    _append_row(labels_csv, row, _FRAMES_COLUMNS)


def derive_digit_samples(
    frame_meta: FrameMeta,
    full_reading: str,
    cells: list[DigitCell],
    normalized_results: list[NormalizationResult],
    raw_cell_dir: Path,
    normalized_dir: Path,
) -> list[DigitSample]:
    """Derive per-digit samples from a labeled frame and save images.

    Args:
        frame_meta: Frame metadata.
        full_reading: Full reading string; length must equal len(cells).
        cells: Extracted digit cells in order.
        normalized_results: Normalization result for each cell.
        raw_cell_dir: Directory to save raw cell images.
        normalized_dir: Directory to save normalized images.

    Returns:
        List of DigitSample, one per digit.

    Raises:
        ValueError: If full_reading length does not match cells length.
    """
    if len(full_reading) != len(cells):
        raise ValueError(
            f"Reading '{full_reading}' has {len(full_reading)} digits "
            f"but found {len(cells)} cells"
        )

    raw_cell_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    samples: list[DigitSample] = []
    for cell, norm_result, digit_char in zip(cells, normalized_results, full_reading):
        stem = f"{frame_meta.frame_id}_{frame_meta.meter_id}_pos{cell.position}"
        raw_path = raw_cell_dir / f"{stem}_raw.png"
        norm_path = normalized_dir / f"{stem}_norm.png"

        cv2.imwrite(str(raw_path), cell.image_bgr)
        cv2.imwrite(str(norm_path), norm_result.normalized)

        quality: str
        if not norm_result.success:
            quality = "uncertain"
        else:
            quality = "ok"

        samples.append(
            DigitSample(
                frame_id=frame_meta.frame_id,
                meter_id=frame_meta.meter_id,
                timestamp=frame_meta.timestamp,
                position=cell.position,
                digit_label=int(digit_char),
                full_reading=full_reading,
                raw_cell_path=raw_path,
                normalized_path=norm_path,
                quality=quality,
            )
        )
    return samples


def append_digit_samples(samples_csv: Path, samples: list[DigitSample]) -> None:
    """Append digit sample rows to samples.csv.

    Creates the CSV with headers if it does not exist.

    Args:
        samples_csv: Path to samples.csv.
        samples: Digit samples to append.
    """
    _ensure_csv(samples_csv, _SAMPLES_COLUMNS)
    for s in samples:
        row = {
            "frame_id": s.frame_id,
            "timestamp": s.timestamp.isoformat(),
            "meter_id": s.meter_id,
            "position": s.position,
            "digit_label": s.digit_label,
            "full_reading": s.full_reading,
            "raw_cell_path": str(s.raw_cell_path),
            "normalized_path": str(s.normalized_path),
            "quality": s.quality,
        }
        _append_row(samples_csv, row, _SAMPLES_COLUMNS)


def load_samples(samples_csv: Path) -> pd.DataFrame:
    """Load the samples CSV into a DataFrame.

    Args:
        samples_csv: Path to samples.csv.

    Returns:
        DataFrame with one row per digit sample.
    """
    if not samples_csv.exists():
        return pd.DataFrame(columns=_SAMPLES_COLUMNS)
    df = pd.read_csv(samples_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_frame_labels(labels_csv: Path) -> pd.DataFrame:
    """Load the frames CSV into a DataFrame.

    Args:
        labels_csv: Path to frames.csv.

    Returns:
        DataFrame with one row per labeled frame.
    """
    if not labels_csv.exists():
        return pd.DataFrame(columns=_FRAMES_COLUMNS)
    df = pd.read_csv(labels_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _ensure_csv(path: Path, columns: list[str]) -> None:
    """Create a CSV file with headers if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def _append_row(path: Path, row: dict, columns: list[str]) -> None:
    """Append one row to an existing CSV."""
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)
