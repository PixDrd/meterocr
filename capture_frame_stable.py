"""Capture a single frame from a V4L2 webcam with retries.

Usage:
    python capture_frame_stable.py <video_device> <image_output>
    python capture_frame_stable.py <video_device> --focus-test <start> <end>

Examples:
    python capture_frame_stable.py /dev/video0 out.jpg
    python capture_frame_stable.py /dev/video0 --focus-test 0 255

Focus test mode:
    Disables autofocus and captures one image per focus value from <start> to
    <end> in steps of 5, saving each to focustest/f<value>.jpg.

Notes:
- Prefer /dev/v4l/by-id/... instead of numeric indices. Those paths are much more stable.
- The script explicitly uses the V4L2 backend, requests MJPG at 1920x1080, and retries on transient failures.
- Frames are decoded from MJPG and re-encoded as JPEG at quality 95.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

OPEN_RETRIES = 4
READ_RETRIES = 5
RETRY_DELAY_SECONDS = 4.0
FOCUS_SLEEP_SECONDS = 0.5
WARMUP_GRABS = 3
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080
REQUESTED_FOURCC = "MJPG"
VALID_EXTENSIONS = {".jpg", ".jpeg"}


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


def open_capture(device: str | int, focus: int | None = None) -> cv2.VideoCapture:
    # V4L2 backend cannot open cameras by symlink path; resolve to the real
    # device node (e.g. /dev/v4l/by-id/... -> /dev/video0) before opening.
    if isinstance(device, str):
        device = str(Path(device).resolve())
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open video capture device {describe_device(device)}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*REQUESTED_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    if focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        time.sleep(FOCUS_SLEEP_SECONDS)
    return cap


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, object]:
    for _ in range(WARMUP_GRABS):
        cap.grab()
    return cap.read()


def write_frame(frame, output_path: Path) -> bool:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    try:
        output_path.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False


def capture_with_retries(device: str | int, focus: int | None = None):
    last_error: str | None = None

    for open_attempt in range(1, OPEN_RETRIES + 1):
        cap: cv2.VideoCapture | None = None
        try:
            cap = open_capture(device, focus=focus)

            for read_attempt in range(1, READ_RETRIES + 1):
                ok, buf = read_frame(cap)
                if ok and buf is not None:
                    return buf

                last_error = (
                    f"read failed on attempt {read_attempt}/{READ_RETRIES} "
                    f"after opening {describe_device(device)}"
                )
                time.sleep(RETRY_DELAY_SECONDS)

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


def focus_test(device: str | int, focus_start: int, focus_end: int) -> int:
    out_dir = Path("focustest")
    out_dir.mkdir(exist_ok=True)

    step = 5 if focus_end >= focus_start else -5
    values = range(focus_start, focus_end + step, step)

    print(f"Focus test: {focus_start} → {focus_end}, step 5 — saving to {out_dir}/")
    errors = 0
    for focus in values:
        print(f"  focus={focus} ... ", end="", flush=True)
        try:
            buf = capture_with_retries(device, focus=focus)
        except Exception as exc:
            print(f"FAILED ({exc})")
            errors += 1
            continue

        out_path = out_dir / f"f{focus}.jpg"
        if not write_frame(buf, out_path):
            print(f"FAILED (could not write {out_path})")
            errors += 1
            continue

        print(f"saved {out_path}")

    total = len(list(values))
    print(f"\nDone: {total - errors}/{total} images saved to {out_dir}/")
    return 0 if errors == 0 else 1


def main() -> int:
    args = sys.argv[1:]

    if len(args) == 4 and args[1] == "--focus-test":
        device = parse_device_arg(args[0])
        try:
            focus_start = int(args[2])
            focus_end = int(args[3])
        except ValueError:
            print("Error: focus start and end must be integers")
            return 1
        return focus_test(device, focus_start, focus_end)

    focus: int | None = None
    if "--focus" in args:
        idx = args.index("--focus")
        if idx + 1 >= len(args):
            print("Error: --focus requires a value")
            return 1
        try:
            focus = int(args[idx + 1])
        except ValueError:
            print("Error: --focus value must be an integer")
            return 1
        args = args[:idx] + args[idx + 2:]

    if len(args) != 2:
        print("Usage: python capture_frame_stable.py <video_device> <image_output> [--focus <value>]")
        print("       python capture_frame_stable.py <video_device> --focus-test <start> <end>")
        print("Tip: use /dev/v4l/by-id/... instead of numeric indices like 0, 2, 4.")
        return 1

    device = parse_device_arg(args[0])

    try:
        output_path = validate_output_path(args[1])
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        buf = capture_with_retries(device, focus=focus)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    if not write_frame(buf, output_path):
        print(f"Error: could not write image file '{output_path}'.")
        return 1

    print(f"Saved frame from device {describe_device(device)} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
