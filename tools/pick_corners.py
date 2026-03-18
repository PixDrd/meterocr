#!/usr/bin/env python3
"""Click the four corners of a meter display to generate perspective_src_points config.

Usage:
    python tools/pick_corners.py path/to/frame.png

Click order: top-left → top-right → bottom-right → bottom-left

Press 'r' to reset and start over, 'q' or Escape to quit without output.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
COLORS = [
    (0, 255, 0),    # green  - top-left
    (0, 200, 255),  # yellow - top-right
    (0, 100, 255),  # orange - bottom-right
    (255, 80, 80),  # blue   - bottom-left
]
RADIUS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_state(base_img, points):
    img = base_img.copy()
    n = len(points)

    # Draw lines between collected points
    for i in range(1, n):
        cv2.line(img, points[i - 1], points[i], (200, 200, 200), 1, cv2.LINE_AA)
    if n == 4:
        cv2.line(img, points[3], points[0], (200, 200, 200), 1, cv2.LINE_AA)
        # Fill polygon lightly
        poly = np.array(points, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (255, 255, 255))
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # Draw dots and labels
    for i, pt in enumerate(points):
        cv2.circle(img, pt, RADIUS, COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(img, pt, RADIUS + 1, (0, 0, 0), 1, cv2.LINE_AA)
        label = f"{i + 1}. {CORNER_LABELS[i]}"
        cv2.putText(img, label, (pt[0] + 12, pt[1] + 5), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (pt[0] + 12, pt[1] + 5), FONT, 0.5, COLORS[i], 1, cv2.LINE_AA)

    # Status bar at top
    if n < 4:
        next_label = CORNER_LABELS[n]
        status = f"Click {n + 1}/4: {next_label}   |   r = reset   |   q / Esc = quit"
    else:
        status = "All 4 corners set. Press Enter to confirm, r to reset."
    cv2.putText(img, status, (10, 22), FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, status, (10, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    base = cv2.imread(str(image_path))
    if base is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(1)

    # Scale down for display if image is very large, keep mapping back to original coords
    h, w = base.shape[:2]
    max_display_w = 1600
    scale = min(1.0, max_display_w / w)
    if scale < 1.0:
        display_base = cv2.resize(base, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        display_base = base
        scale = 1.0

    points_display = []  # coords in display space
    window = "pick_corners — click corners in order (TL TR BR BL)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, draw_state(display_base, points_display))

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_display) < 4:
            points_display.append((x, y))
            cv2.imshow(window, draw_state(display_base, points_display))

    cv2.setMouseCallback(window, on_click)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):  # q or Escape
            print("Quit without output.")
            break
        elif key == ord('r'):
            points_display.clear()
            cv2.imshow(window, draw_state(display_base, points_display))
        elif key == 13 and len(points_display) == 4:  # Enter
            # Map display coords back to original image coords
            pts = [(round(x / scale), round(y / scale)) for x, y in points_display]
            print("\n# Copy-paste into configs/meters.yaml under the relevant meter:\n")
            print("    perspective_src_points:")
            for pt, label in zip(pts, CORNER_LABELS):
                print(f"      - [{pt[0]}, {pt[1]}]             # {label}")
            print()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
