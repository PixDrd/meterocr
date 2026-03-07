"""Align a raw camera frame to a canonical meter crop."""

import cv2
import numpy as np

from meterocr.types import ImageU8, MeterConfig


def align_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    """Align a raw frame to a canonical meter image.

    Chooses crop or perspective warp based on config, then optionally
    applies a bounded translation correction against a reference.

    Args:
        frame_bgr: Full camera frame in BGR.
        config: Meter configuration.

    Returns:
        Aligned meter image of size (aligned_height, aligned_width).
    """
    if config.perspective_src_points is not None:
        aligned = warp_meter(frame_bgr, config)
    else:
        aligned = crop_meter(frame_bgr, config)

    if config.alignment_reference_path is not None and config.max_translation_px > 0:
        ref_bgr = cv2.imread(str(config.alignment_reference_path))
        if ref_bgr is not None:
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
            cand_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            dx, dy = estimate_translation(ref_gray, cand_gray, config.max_translation_px)
            aligned = apply_translation(aligned, dx, dy)

    return aligned


def crop_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    """Crop and resize a meter region from the frame using crop_source_box.

    Args:
        frame_bgr: Full camera frame in BGR.
        config: Meter configuration with crop_source_box defined.

    Returns:
        Cropped meter image resized to (aligned_height, aligned_width).
    """
    if config.crop_source_box is not None:
        box = config.crop_source_box
        h_frame, w_frame = frame_bgr.shape[:2]
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w_frame, box.x + box.w)
        y2 = min(h_frame, box.y + box.h)
        cropped = frame_bgr[y1:y2, x1:x2]
    else:
        cropped = frame_bgr

    return cv2.resize(
        cropped,
        (config.aligned_width, config.aligned_height),
        interpolation=cv2.INTER_LINEAR,
    )


def warp_meter(frame_bgr: ImageU8, config: MeterConfig) -> ImageU8:
    """Apply a perspective warp to extract the meter region.

    Args:
        frame_bgr: Full camera frame in BGR.
        config: Meter configuration with perspective_src_points defined.

    Returns:
        Perspective-corrected meter image of size (aligned_height, aligned_width).
    """
    if config.perspective_src_points is None:
        raise ValueError(f"Meter {config.meter_id} has no perspective_src_points")

    src_pts = np.array(config.perspective_src_points, dtype=np.float32)
    dst_pts = np.array(
        [
            [0, 0],
            [config.aligned_width - 1, 0],
            [config.aligned_width - 1, config.aligned_height - 1],
            [0, config.aligned_height - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        frame_bgr,
        M,
        (config.aligned_width, config.aligned_height),
    )


def estimate_translation(
    reference_gray: ImageU8,
    candidate_gray: ImageU8,
    max_shift_px: int,
) -> tuple[int, int]:
    """Estimate the (dx, dy) shift between a reference and candidate image.

    Uses normalized cross-correlation (template matching) within a bounded
    search window.

    Args:
        reference_gray: Reference grayscale image.
        candidate_gray: Candidate grayscale image to compare against.
        max_shift_px: Maximum allowed shift in pixels in each direction.

    Returns:
        (dx, dy) integer pixel shift to move candidate onto reference.
    """
    pad = max_shift_px
    padded = cv2.copyMakeBorder(
        candidate_gray,
        pad, pad, pad, pad,
        cv2.BORDER_REPLICATE,
    )
    result = cv2.matchTemplate(padded, reference_gray, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    dx = pad - max_loc[0]
    dy = pad - max_loc[1]
    dx = int(np.clip(dx, -max_shift_px, max_shift_px))
    dy = int(np.clip(dy, -max_shift_px, max_shift_px))
    return dx, dy


def apply_translation(image: ImageU8, dx: int, dy: int) -> ImageU8:
    """Shift an image by (dx, dy) pixels, filling borders with replication.

    Args:
        image: Input image.
        dx: Horizontal shift (positive = shift right).
        dy: Vertical shift (positive = shift down).

    Returns:
        Shifted image of the same size.
    """
    if dx == 0 and dy == 0:
        return image
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = image.shape[:2]
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return shifted
