"""Image acquisition from USB webcams or test image files.

Each of the three meters has a dedicated USB webcam. This module provides
a unified interface that works both for live capture and offline testing
using pre-recorded images from a directory.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from meterocr.types import ImageU8


class CaptureError(RuntimeError):
    """Raised when a frame cannot be acquired."""


# ---------------------------------------------------------------------------
# Live USB webcam capture
# ---------------------------------------------------------------------------


class WebcamCapture:
    """Capture frames from a USB webcam.

    Resolution defaults to 1920x1080 with MJPG encoding, which is the maximum
    supported by most USB webcams and avoids the bandwidth limit of raw YUYV.
    Pass explicit width/height to override.

    The device can be specified as a full device path (e.g. '/dev/video0') or
    as an integer index (e.g. 0).  Prefer the full path so the correct camera
    is always used regardless of enumeration order on boot.

    Args:
        device: Device path string (e.g. '/dev/video0') or integer index.
        width: Desired capture width in pixels, or None to use camera default.
        height: Desired capture height in pixels, or None to use camera default.
    """

    def __init__(
        self,
        device: int | str,
        width: int = 1920,
        height: int = 1080,
        focus: int | None = None,
        warmup_grabs: int = 3,
        focus_settle_s: float = 4.0,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._focus = focus
        self._warmup_grabs = warmup_grabs
        self._focus_settle_s = focus_settle_s
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam. Raises CaptureError if unavailable."""
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise CaptureError(
                f"Cannot open webcam at device '{self._device}'"
            )
        # Request MJPG encoding before setting resolution; this is required for
        # USB webcams to deliver full 1920x1080 without hitting the USB 2.0
        # bandwidth ceiling that raw YUYV would hit at high resolutions.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._focus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, self._focus)
            if self._focus_settle_s > 0:
                time.sleep(self._focus_settle_s)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(self._warmup_grabs):
            cap.grab()
        self._cap = cap

    def close(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab_frame(self) -> ImageU8:
        """Grab one frame from the webcam.

        Returns:
            BGR image array.

        Raises:
            CaptureError: If the webcam is not open or the frame cannot be read.
        """
        if self._cap is None:
            raise CaptureError("Webcam is not open; call open() first")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CaptureError(
                f"Failed to read frame from webcam '{self._device}'"
            )
        return frame

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def list_available_webcams(max_index: int = 10) -> list[int]:
    """Probe device indices and return those that open successfully.

    Args:
        max_index: Upper bound (exclusive) for device indices to probe.

    Returns:
        Sorted list of working device indices.
    """
    available: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


# ---------------------------------------------------------------------------
# Offline / test image source
# ---------------------------------------------------------------------------


class TestImageCapture:
    """Serve frames from a directory of static test images.

    Images are sorted by filename. When the end of the list is reached the
    capture stops iterating (or loops if loop=True).

    Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff, .tif

    Args:
        image_dir: Directory containing test images for one meter.
        loop: If True, loop back to the first image after the last one.
    """

    _EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, image_dir: Path, loop: bool = False) -> None:
        self._image_dir = image_dir
        self._loop = loop
        self._paths: list[Path] = []
        self._index: int = 0

    def open(self) -> None:
        """Scan the directory and prepare the image list.

        Raises:
            CaptureError: If directory does not exist or is empty.
        """
        if not self._image_dir.is_dir():
            raise CaptureError(
                f"Test image directory does not exist: {self._image_dir}"
            )
        paths = sorted(
            p for p in self._image_dir.iterdir()
            if p.suffix.lower() in self._EXTENSIONS
        )
        if not paths:
            raise CaptureError(
                f"No images found in {self._image_dir}"
            )
        self._paths = paths
        self._index = 0

    def close(self) -> None:
        """No-op; nothing to release for file-based sources."""

    def grab_frame(self) -> ImageU8:
        """Return the next image in the sorted list.

        Raises:
            CaptureError: If all images have been consumed and loop=False,
                or if an image file cannot be read.
        """
        if self._index >= len(self._paths):
            if self._loop:
                self._index = 0
            else:
                raise CaptureError("All test images have been consumed")

        path = self._paths[self._index]
        self._index += 1

        frame = cv2.imread(str(path))
        if frame is None:
            raise CaptureError(f"Cannot read test image: {path}")
        return frame

    def current_path(self) -> Path | None:
        """Return the path of the image that will be returned by the next grab_frame call."""
        if self._index < len(self._paths):
            return self._paths[self._index]
        return None

    def reset(self) -> None:
        """Reset to the first image."""
        self._index = 0

    def __len__(self) -> int:
        return len(self._paths)

    def __enter__(self) -> "TestImageCapture":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Multi-meter capture manager
# ---------------------------------------------------------------------------


class MeterCaptureSet:
    """Manage capture sources for all three meters.

    Each meter can independently use either a webcam or a test image directory.
    Meter IDs are mapped to capture sources in the constructor.

    Args:
        sources: Mapping from meter_id to a WebcamCapture or TestImageCapture.

    Example::

        sources = {
            "M1": WebcamCapture(device_index=0),
            "M2": WebcamCapture(device_index=1),
            "M3": TestImageCapture(Path("data/test_images/M3")),
        }
        with MeterCaptureSet(sources) as capture_set:
            frame = capture_set.grab_frame("M1")
    """

    def __init__(
        self,
        sources: dict[str, "WebcamCapture | TestImageCapture"],
    ) -> None:
        self._sources = sources

    def open(self) -> None:
        """Open all capture sources."""
        for source in self._sources.values():
            source.open()

    def close(self) -> None:
        """Close all capture sources."""
        for source in self._sources.values():
            source.close()

    def grab_frame(self, meter_id: str) -> ImageU8:
        """Grab one frame for the specified meter.

        Args:
            meter_id: The meter identifier.

        Returns:
            BGR image array.

        Raises:
            KeyError: If meter_id is not registered.
            CaptureError: If the frame cannot be acquired.
        """
        if meter_id not in self._sources:
            raise KeyError(f"No capture source registered for meter '{meter_id}'")
        return self._sources[meter_id].grab_frame()

    def meter_ids(self) -> list[str]:
        """Return sorted list of registered meter IDs."""
        return sorted(self._sources.keys())

    def __enter__(self) -> "MeterCaptureSet":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_capture_set(
    meter_ids: list[str],
    webcam_devices: dict[str, str | int] | None = None,
    test_image_dirs: dict[str, Path] | None = None,
    offline: bool = False,
) -> MeterCaptureSet:
    """Build a MeterCaptureSet from config parameters.

    When offline=True, all meters use TestImageCapture from test_image_dirs.
    When offline=False, meters listed in webcam_devices use WebcamCapture;
    any remaining meters fall back to TestImageCapture if available in
    test_image_dirs.

    Args:
        meter_ids: List of meter IDs to set up (e.g. ['M1', 'M2', 'M3']).
        webcam_devices: Mapping from meter_id to device path (e.g. '/dev/video0')
            or integer index.  Typically populated from meters.yaml.
        test_image_dirs: Mapping from meter_id to test image directory.
        offline: Force all meters to use test images.

    Returns:
        Configured MeterCaptureSet (not yet opened).

    Raises:
        ValueError: If a meter has no viable capture source.
    """
    webcam_devices = webcam_devices or {}
    test_image_dirs = test_image_dirs or {}

    sources: dict[str, WebcamCapture | TestImageCapture] = {}
    for mid in meter_ids:
        if not offline and mid in webcam_devices:
            sources[mid] = WebcamCapture(webcam_devices[mid])
        elif mid in test_image_dirs:
            sources[mid] = TestImageCapture(test_image_dirs[mid])
        else:
            raise ValueError(
                f"No capture source available for meter '{mid}'. "
                "Set video_device in meters.yaml or provide a test image directory."
            )
    return MeterCaptureSet(sources)
