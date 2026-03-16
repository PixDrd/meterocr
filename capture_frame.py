"""Capture a single frame from an OpenCV video device and save it as a PNG."""

from __future__ import annotations

import sys

import cv2


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python capture_frame.py <video_capture_index> <png_image_name>")
        return 1

    try:
        capture_index = int(sys.argv[1])
    except ValueError:
        print("Error: <video_capture_index> must be an integer.")
        return 1

    output_name = sys.argv[2]

    if not output_name.lower().endswith(".png"):
        print("Error: <png_image_name> must end with .png")
        return 1

    cap = cv2.VideoCapture(capture_index)
    if not cap.isOpened():
        print(f"Error: could not open video capture device {capture_index}.")
        return 1

    # Request MJPG and 1920x1080; the camera will honour the highest resolution
    # it supports (MJPG avoids the USB 2.0 bandwidth limit of raw YUYV).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: could not read a frame from device {capture_index}.")
            return 1

        if not cv2.imwrite(output_name, frame):
            print(f"Error: could not write PNG file '{output_name}'.")
            return 1
    finally:
        cap.release()

    print(f"Saved frame from device {capture_index} to {output_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
