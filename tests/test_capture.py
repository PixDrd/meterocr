"""Tests for capture.py."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from meterocr.capture import (
    CaptureError,
    MeterCaptureSet,
    TestImageCapture,
    build_capture_set,
)


def _write_test_images(directory: Path, count: int = 3) -> None:
    """Write synthetic PNG images to a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = np.full((480, 640, 3), i * 40, dtype=np.uint8)
        cv2.imwrite(str(directory / f"frame_{i:04d}.png"), img)


def test_test_image_capture_returns_frames(tmp_path):
    img_dir = tmp_path / "images"
    _write_test_images(img_dir, count=3)

    capture = TestImageCapture(img_dir)
    capture.open()
    frame = capture.grab_frame()
    assert frame.shape == (480, 640, 3)
    capture.close()


def test_test_image_capture_correct_frame_count(tmp_path):
    img_dir = tmp_path / "images"
    _write_test_images(img_dir, count=3)

    with TestImageCapture(img_dir) as capture:
        assert len(capture) == 3
        frames = []
        for _ in range(3):
            frames.append(capture.grab_frame())
        assert len(frames) == 3


def test_test_image_capture_raises_after_exhausted(tmp_path):
    img_dir = tmp_path / "images"
    _write_test_images(img_dir, count=2)

    with TestImageCapture(img_dir) as capture:
        capture.grab_frame()
        capture.grab_frame()
        with pytest.raises(CaptureError):
            capture.grab_frame()


def test_test_image_capture_loop(tmp_path):
    img_dir = tmp_path / "images"
    _write_test_images(img_dir, count=2)

    with TestImageCapture(img_dir, loop=True) as capture:
        for _ in range(6):  # 3x through 2-image set
            frame = capture.grab_frame()
            assert frame is not None


def test_test_image_capture_missing_dir():
    capture = TestImageCapture(Path("/nonexistent/path"))
    with pytest.raises(CaptureError):
        capture.open()


def test_test_image_capture_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    capture = TestImageCapture(empty)
    with pytest.raises(CaptureError):
        capture.open()


def test_meter_capture_set_multiple_meters(tmp_path):
    for mid in ["M1", "M2", "M3"]:
        _write_test_images(tmp_path / mid, count=2)

    sources = {
        mid: TestImageCapture(tmp_path / mid) for mid in ["M1", "M2", "M3"]
    }
    with MeterCaptureSet(sources) as cs:
        for mid in ["M1", "M2", "M3"]:
            frame = cs.grab_frame(mid)
            assert frame.shape == (480, 640, 3)


def test_meter_capture_set_unknown_meter(tmp_path):
    _write_test_images(tmp_path / "M1", count=1)
    sources = {"M1": TestImageCapture(tmp_path / "M1")}
    with MeterCaptureSet(sources) as cs:
        with pytest.raises(KeyError):
            cs.grab_frame("M_UNKNOWN")


def test_build_capture_set_offline(tmp_path):
    for mid in ["M1", "M2", "M3"]:
        _write_test_images(tmp_path / mid, count=1)

    test_dirs = {mid: tmp_path / mid for mid in ["M1", "M2", "M3"]}
    cs = build_capture_set(
        meter_ids=["M1", "M2", "M3"],
        test_image_dirs=test_dirs,
        offline=True,
    )
    assert cs.meter_ids() == ["M1", "M2", "M3"]


def test_build_capture_set_missing_source_raises():
    with pytest.raises(ValueError, match="No capture source"):
        build_capture_set(
            meter_ids=["M1"],
            webcam_devices={},
            test_image_dirs={},
            offline=False,
        )
