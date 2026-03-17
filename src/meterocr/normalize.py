"""Normalize a raw digit cell into a canonical image for HOG classification."""

import cv2
import numpy as np

from meterocr.types import ImageU8, NormalizationConfig, NormalizationResult


def normalize_digit(
    cell_bgr: ImageU8,
    threshold_mode: str,
    invert_binary: bool,
    cfg: NormalizationConfig,
) -> NormalizationResult:
    """Transform a raw digit cell into a normalized binary image.

    The same normalization code must be used for both training and inference.

    Args:
        cell_bgr: Raw digit cell in BGR.
        threshold_mode: 'otsu' or 'adaptive'.
        invert_binary: If True, invert the binary image after thresholding.
        cfg: Normalization configuration.

    Returns:
        NormalizationResult with normalized image and diagnostic fields.
    """
    gray = to_gray(cell_bgr)
    binary = threshold_digit(gray, threshold_mode, invert_binary)
    binary = remove_small_components(binary, cfg.min_component_area)

    digit_bbox = find_main_digit_bbox(binary, cfg.min_component_area)
    if digit_bbox is None:
        blank = np.zeros((cfg.output_height, cfg.output_width), dtype=np.uint8)
        return NormalizationResult(
            normalized=blank,
            gray=gray,
            binary=binary,
            digit_bbox_xywh=None,
            success=False,
            reason="no foreground blob found",
        )

    bx, by, bw, bh = digit_bbox
    margin = cfg.bbox_margin_px
    h_img, w_img = binary.shape
    x1 = max(0, bx - margin)
    y1 = max(0, by - margin)
    x2 = min(w_img, bx + bw + margin)
    y2 = min(h_img, by + bh + margin)
    digit_crop = binary[y1:y2, x1:x2]

    normalized = center_and_resize(digit_crop, (cfg.output_width, cfg.output_height))

    return NormalizationResult(
        normalized=normalized,
        gray=gray,
        binary=binary,
        digit_bbox_xywh=(bx, by, bw, bh),
        success=True,
        reason="ok",
    )


def to_gray(cell_bgr: ImageU8) -> ImageU8:
    """Convert a BGR image to grayscale.

    Args:
        cell_bgr: BGR image.

    Returns:
        Grayscale image.
    """
    if cell_bgr.ndim == 2:
        return cell_bgr.copy()
    return cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)


def threshold_digit(gray: ImageU8, mode: str, invert_binary: bool) -> ImageU8:
    """Apply thresholding to a grayscale digit image.

    Args:
        gray: Grayscale image.
        mode: 'otsu' or 'adaptive'.
        invert_binary: If True, invert the result so digit is foreground.

    Returns:
        Binary image (0 or 255).
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if mode == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    else:
        # Default: Otsu
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    if invert_binary:
        binary = cv2.bitwise_not(binary)
    return binary


def remove_small_components(binary_img: ImageU8, min_area: int) -> ImageU8:
    """Remove connected components smaller than min_area pixels.

    Args:
        binary_img: Binary image (0 or 255).
        min_area: Minimum component area in pixels to retain.

    Returns:
        Cleaned binary image.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    result = np.zeros_like(binary_img)
    for label in range(1, n_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == label] = 255
    return result


def find_main_digit_bbox(
    binary_img: ImageU8,
    min_area: int,
) -> tuple[int, int, int, int] | None:
    """Find the bounding box of the largest foreground connected component.

    Args:
        binary_img: Binary image (0 or 255, foreground is 255).
        min_area: Minimum component area to consider.

    Returns:
        (x, y, w, h) bounding box or None if no valid component exists.
    """
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    best_area = -1
    best_box = None
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_box = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
    return best_box


def center_and_resize(
    binary_digit: ImageU8,
    output_size: tuple[int, int],
) -> ImageU8:
    """Center a digit crop into a fixed canvas, preserving aspect ratio.

    Args:
        binary_digit: Tightly cropped binary digit image.
        output_size: (width, height) of the output canvas.

    Returns:
        Binary image of exactly output_size.
    """
    out_w, out_h = output_size
    h, w = binary_digit.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(binary_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    x_off = (out_w - new_w) // 2
    y_off = (out_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas
