"""Capture a single frame from a V4L2 webcam with retries.

Usage:
    python capture_frame_stable.py <video_device> <image_output>

Examples:
    python capture_frame_stable.py /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_12345-video-index0 out.jpg
    python capture_frame_stable.py /dev/video0 out.png
    python capture_frame_stable.py 0 out.jpg

Notes:
- Prefer /dev/v4l/by-id/... instead of numeric indices. Those paths are much more stable.
- The script explicitly uses the V4L2 backend, requests MJPG at 1920x1080, and retries on transient failures.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

OPEN_RETRIES = 4
READ_RETRIES = 5
RETRY_DELAY_SECONDS = 1.0
WARMUP_GRABS = 3
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080
REQUESTED_FOURCC = "MJPG"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_device_arg(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def validate_output_path(output_name: str) -> Path:
    output_path = Path(output_name)
    if output_path.suffix.lower() not in VALID_EXTENSIONS:
        valid = ", ".join(sorted(VALID_EXTENSIONS))
        raise ValueError(f"<image_output> must end with one of: {valid}")
    return output_path


def describe_device(device: str | int) -> str:
    return str(device)


def open_capture(device: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open video capture device {describe_device(device)}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*REQUESTED_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    return cap


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, object]:
    for _ in range(WARMUP_GRABS):
        cap.grab()
    return cap.read()


def capture_with_retries(device: str | int):
    last_error: str | None = None

    for open_attempt in range(1, OPEN_RETRIES + 1):
        cap: cv2.VideoCapture | None = None
        try:
            cap = open_capture(device)

            for read_attempt in range(1, READ_RETRIES + 1):
                ok, frame = read_frame(cap)
                if ok and frame is not None:
                    return frame

                last_error = (
                    f"read failed on attempt {read_attempt}/{READ_RETRIES} "
                    f"after opening {describe_device(device)}"
                )
                time.sleep(0.2)

            last_error = (
                f"opened {describe_device(device)} but could not read a frame after "
                f"{READ_RETRIES} attempts"
            )
        except Exception as exc:
            last_error = str(exc)
        finally:
            if cap is not None:
                cap.release()

        if open_attempt < OPEN_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(last_error or f"failed to capture from {describe_device(device)}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python capture_frame_stable.py <video_device> <image_output>")
        print("Tip: use /dev/v4l/by-id/... instead of numeric indices like 0, 2, 4.")
        return 1

    device = parse_device_arg(sys.argv[1])

    try:
        output_path = validate_output_path(sys.argv[2])
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frame = capture_with_retries(device)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    if not cv2.imwrite(str(output_path), frame):
        print(f"Error: could not write image file '{output_path}'.")
        return 1

    print(f"Saved frame from device {describe_device(device)} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
