"""Convenience wrapper: label a frame and produce per-digit samples in one step."""

from datetime import datetime
from pathlib import Path

import cv2

from meterocr import dataset
from meterocr.align import align_meter
from meterocr.normalize import normalize_digit
from meterocr.segment import extract_digit_cells
from meterocr.types import FrameMeta, MeterConfig, NormalizationConfig


def label_frame(
    image_path: Path,
    meter_config: MeterConfig,
    normalization_cfg: NormalizationConfig,
    full_reading: str,
    frames_csv: Path,
    samples_csv: Path,
    raw_cell_dir: Path,
    normalized_dir: Path,
) -> None:
    """Label a frame image and persist all derived samples.

    Loads the image, aligns it, extracts digit cells, normalizes each cell,
    writes cell images, and appends rows to frames.csv and samples.csv.

    Args:
        image_path: Path to the source frame image.
        meter_config: Meter configuration.
        normalization_cfg: Normalization configuration.
        full_reading: Full meter reading string (digits only, e.g. '03506').
        frames_csv: Path to frames.csv.
        samples_csv: Path to samples.csv.
        raw_cell_dir: Directory to save raw cell images.
        normalized_dir: Directory to save normalized cell images.

    Raises:
        ValueError: If full_reading contains non-digit characters.
        FileNotFoundError: If the image cannot be loaded.
    """
    if not full_reading.isdigit():
        raise ValueError(
            f"full_reading must contain only digits, got '{full_reading}'"
        )

    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    timestamp = datetime.now()
    frame_id = _make_frame_id(image_path, meter_config.meter_id, timestamp)

    frame_meta = FrameMeta(
        frame_id=frame_id,
        meter_id=meter_config.meter_id,
        timestamp=timestamp,
        source_path=image_path,
    )

    aligned = align_meter(frame_bgr, meter_config)
    cells = extract_digit_cells(aligned, meter_config, frame_meta)

    normalized_results = [
        normalize_digit(
            cell.image_bgr,
            meter_config.threshold_mode,
            meter_config.invert_binary,
            normalization_cfg,
        )
        for cell in cells
    ]

    dataset.append_frame_label(
        labels_csv=frames_csv,
        frame_id=frame_id,
        meter_id=meter_config.meter_id,
        timestamp=timestamp,
        source_path=image_path,
        full_reading=full_reading,
    )

    samples = dataset.derive_digit_samples(
        frame_meta=frame_meta,
        full_reading=full_reading,
        cells=cells,
        normalized_results=normalized_results,
        raw_cell_dir=raw_cell_dir,
        normalized_dir=normalized_dir,
    )
    dataset.append_digit_samples(samples_csv, samples)


def _make_frame_id(image_path: Path, meter_id: str, timestamp: datetime) -> str:
    """Generate a unique frame ID from image filename stem and meter ID."""
    stem = image_path.stem
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{meter_id}_{stem}_{ts}"
