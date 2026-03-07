"""Miscellaneous utilities."""

import uuid
from datetime import datetime
from pathlib import Path


def make_frame_id(meter_id: str, timestamp: datetime | None = None) -> str:
    """Generate a unique frame ID.

    Args:
        meter_id: The meter identifier.
        timestamp: Optional timestamp; defaults to now.

    Returns:
        String frame ID.
    """
    ts = timestamp or datetime.now()
    uid = uuid.uuid4().hex[:8]
    return f"{meter_id}_{ts.strftime('%Y%m%d_%H%M%S')}_{uid}"


def save_debug_image(image, path: Path) -> None:
    """Save an image to disk, creating parent directories as needed.

    Args:
        image: NumPy image array (BGR or grayscale).
        path: Destination path.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
