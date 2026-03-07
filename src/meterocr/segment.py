"""Extract fixed-position digit cells from an aligned meter image."""

import cv2
import numpy as np

from meterocr.types import DigitBox, DigitCell, FrameMeta, ImageU8, MeterConfig


def extract_digit_cells(
    aligned_meter_bgr: ImageU8,
    config: MeterConfig,
    frame_meta: FrameMeta,
) -> list[DigitCell]:
    """Extract all digit cells from an aligned meter image.

    Positions are zero-based and ordered left-to-right. Inner padding is
    applied to avoid capturing separator borders.

    Args:
        aligned_meter_bgr: Aligned meter image in BGR.
        config: Meter configuration with digit_boxes.
        frame_meta: Frame metadata for labeling cells.

    Returns:
        List of DigitCell, one per configured digit box.
    """
    cells: list[DigitCell] = []
    for position, box in enumerate(config.digit_boxes):
        padded_box = DigitBox(
            x=box.x + config.inner_pad_x,
            y=box.y + config.inner_pad_y,
            w=max(1, box.w - 2 * config.inner_pad_x),
            h=max(1, box.h - 2 * config.inner_pad_y),
        )
        cell_img = crop_digit_cell(
            aligned_meter_bgr,
            padded_box.x,
            padded_box.y,
            padded_box.w,
            padded_box.h,
        )
        cells.append(
            DigitCell(
                frame_id=frame_meta.frame_id,
                meter_id=frame_meta.meter_id,
                position=position,
                bbox=padded_box,
                image_bgr=cell_img,
            )
        )
    return cells


def crop_digit_cell(
    aligned_meter_bgr: ImageU8,
    x: int,
    y: int,
    w: int,
    h: int,
) -> ImageU8:
    """Crop a rectangular region from the aligned meter image.

    Args:
        aligned_meter_bgr: Aligned meter image in BGR.
        x: Left pixel (inclusive).
        y: Top pixel (inclusive).
        w: Width in pixels.
        h: Height in pixels.

    Returns:
        Cropped image in BGR, same dtype as input.
    """
    img_h, img_w = aligned_meter_bgr.shape[:2]
    x1 = int(np.clip(x, 0, img_w - 1))
    y1 = int(np.clip(y, 0, img_h - 1))
    x2 = int(np.clip(x + w, 1, img_w))
    y2 = int(np.clip(y + h, 1, img_h))
    return aligned_meter_bgr[y1:y2, x1:x2].copy()
