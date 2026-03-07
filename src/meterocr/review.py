"""Collect uncertain or rejected predictions for manual review."""

import csv
from pathlib import Path

from meterocr.types import PredictionResult, StableReadingResult

_REVIEW_COLUMNS = [
    "frame_id",
    "timestamp",
    "meter_id",
    "raw_reading",
    "stable_reading",
    "accepted",
    "reason",
    "min_confidence",
    "mean_confidence",
    "image_path",
]


def append_review_item(
    review_csv: Path,
    prediction: PredictionResult,
    stable: StableReadingResult,
    image_path: Path | None,
) -> None:
    """Append one uncertain or rejected frame to the review queue CSV.

    Creates the CSV with headers if it does not exist.

    Args:
        review_csv: Path to review_queue.csv.
        prediction: The raw prediction for the frame.
        stable: The temporal filter result.
        image_path: Optional path to the original frame image.
    """
    if not review_csv.exists():
        review_csv.parent.mkdir(parents=True, exist_ok=True)
        with review_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_REVIEW_COLUMNS)
            writer.writeheader()

    row = {
        "frame_id": prediction.frame_id,
        "timestamp": prediction.timestamp.isoformat(),
        "meter_id": prediction.meter_id,
        "raw_reading": prediction.raw_reading,
        "stable_reading": stable.stable_reading,
        "accepted": str(stable.accepted),
        "reason": stable.reason,
        "min_confidence": f"{prediction.min_confidence:.4f}",
        "mean_confidence": f"{prediction.mean_confidence:.4f}",
        "image_path": str(image_path) if image_path else "",
    }
    with review_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REVIEW_COLUMNS)
        writer.writerow(row)
