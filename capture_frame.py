"""Capture a single frame from an OpenCV video device and save it as a JPEG."""

from __future__ import annotations

import sys

import cv2


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python capture_frame.py <video_capture_index> <jpg_image_name>")
        return 1

    try:
        capture_index = int(sys.argv[1])
    except ValueError:
        print("Error: <video_capture_index> must be an integer.")
        return 1

    output_name = sys.argv[2]

    if not output_name.lower().endswith((".jpg", ".jpeg")):
        print("Error: <jpg_image_name> must end with .jpg or .jpeg")
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
    # CAP_PROP_FORMAT = -1 makes retrieve() return the raw compressed bytes
    # instead of decoding to BGR, so we get the camera's JPEG directly.
    cap.set(cv2.CAP_PROP_FORMAT, -1)

    try:
        if not cap.grab():
            print(f"Error: could not grab a frame from device {capture_index}.")
            return 1
        ret, buf = cap.retrieve()
        if not ret or buf is None:
            print(f"Error: could not retrieve frame from device {capture_index}.")
            return 1

        with open(output_name, "wb") as f:
            f.write(buf.tobytes())
    finally:
        cap.release()

    print(f"Saved frame from device {capture_index} to {output_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
