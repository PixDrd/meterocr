#!/usr/bin/env python3
"""Click the four corners of a meter display to generate perspective_src_points config.

Usage:
    python tools/pick_corners.py path/to/frame.png

Click order: top-left → top-right → bottom-right → bottom-left

Controls:
  Left-click          Place next corner (when fewer than 4 placed)
  Left-click + drag   Move an existing corner
  + / -               Zoom in / out centred on cursor
  Middle-click drag   Pan  (or right-click drag)
  Enter               Print config and exit
  r                   Reset all corners
  q / Escape          Quit without output
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
GRAB_RADIUS_PX = 14   # screen pixels within which a click grabs a corner
FONT = cv2.FONT_HERSHEY_SIMPLEX
ZOOM_FACTOR = 1.15
MAX_ZOOM = 32.0
MIN_ZOOM = 0.05
STATUS_BAR_H = 32     # pixels reserved at bottom for the status bar
MIN_WIN_W = 600
MIN_WIN_H = 450

# ── coordinate helpers ────────────────────────────────────────────────────────

def i2s(ix, iy, view_x, view_y, scale):
    """Image coords → screen coords."""
    return (round((ix - view_x) * scale), round((iy - view_y) * scale))

def s2i(sx, sy, view_x, view_y, scale):
    """Screen coords → image coords (float)."""
    return (sx / scale + view_x, sy / scale + view_y)

# ── rendering ─────────────────────────────────────────────────────────────────

def render(img, win_w, win_h, view_x, view_y, scale, points, dragging, loupe_pos):
    H, W = img.shape[:2]

    # Visible image region
    vis_w = win_w / scale
    vis_h = win_h / scale
    ix0 = max(0.0, view_x)
    iy0 = max(0.0, view_y)
    ix1 = min(float(W), view_x + vis_w)
    iy1 = min(float(H), view_y + vis_h)

    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

    if ix1 > ix0 and iy1 > iy0:
        crop = img[int(iy0):int(iy1), int(ix0):int(ix1)]
        out_w = max(1, round((ix1 - ix0) * scale))
        out_h = max(1, round((iy1 - iy0) * scale))
        interp = cv2.INTER_NEAREST if scale > 3 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        sx0 = max(0, round(-view_x * scale))
        sy0 = max(0, round(-view_y * scale))
        canvas[sy0:sy0 + out_h, sx0:sx0 + out_w] = resized

    n = len(points)

    # Outline polygon
    if n >= 2:
        screen_pts = [i2s(px, py, view_x, view_y, scale) for px, py in points]
        for i in range(1, n):
            cv2.line(canvas, screen_pts[i - 1], screen_pts[i], (180, 180, 180), 1, cv2.LINE_AA)
        if n == 4:
            cv2.line(canvas, screen_pts[3], screen_pts[0], (180, 180, 180), 1, cv2.LINE_AA)
            poly = np.array(screen_pts, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], (255, 255, 255))
            cv2.addWeighted(overlay, 0.07, canvas, 0.93, 0, canvas)

    # Corner dots
    for i, (px, py) in enumerate(points):
        sp = i2s(px, py, view_x, view_y, scale)
        r = 9 if i == dragging else 7
        cv2.circle(canvas, sp, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, sp, r, COLORS[i], -1, cv2.LINE_AA)
        label = f"{i + 1}. {CORNER_LABELS[i]}  ({round(px)}, {round(py)})"
        cv2.putText(canvas, label, (sp[0] + 11, sp[1] + 5), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (sp[0] + 11, sp[1] + 5), FONT, 0.45, COLORS[i], 1, cv2.LINE_AA)

    # Loupe (magnifier) when placing or dragging
    if loupe_pos is not None:
        _draw_loupe(canvas, img, loupe_pos, view_x, view_y, scale, win_w, win_h)

    # Status bar at bottom
    if n < 4:
        status = f"Click {n + 1}/4: {CORNER_LABELS[n]}   |   +/-=zoom   |   r=reset   |   q=quit"
    else:
        status = "Drag corners to adjust.  Enter = confirm & print config.  r = reset."
    bar_y = win_h - STATUS_BAR_H
    cv2.rectangle(canvas, (0, bar_y), (win_w, win_h), (30, 30, 30), -1)
    text_y = win_h - (STATUS_BAR_H - 20) // 2 - 4
    cv2.putText(canvas, status, (8, text_y), FONT, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Zoom level badge (right-aligned in status bar)
    zlabel = f"zoom {scale:.2f}x"
    (tw, _th), _ = cv2.getTextSize(zlabel, FONT, 0.45, 1)
    zx = max(8, win_w - tw - 10)
    cv2.putText(canvas, zlabel, (zx, text_y), FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return canvas


def _draw_loupe(canvas, img, screen_pos, view_x, view_y, scale, win_w, win_h):
    """Draw a small magnifier centred near the cursor."""
    LOUPE_SIZE = 160    # pixels wide/tall in screen space
    LOUPE_ZOOM = 6.0    # magnification relative to current view
    HALF = LOUPE_SIZE // 2

    sx, sy = screen_pos
    ix, iy = s2i(sx, sy, view_x, view_y, scale)

    H, W = img.shape[:2]
    loupe_scale = scale * LOUPE_ZOOM
    half_img = HALF / loupe_scale
    lx0 = max(0, ix - half_img)
    ly0 = max(0, iy - half_img)
    lx1 = min(W, ix + half_img)
    ly1 = min(H, iy + half_img)

    if lx1 <= lx0 or ly1 <= ly0:
        return

    crop = img[int(ly0):int(ly1), int(lx0):int(lx1)]
    zoomed = cv2.resize(crop, (LOUPE_SIZE, LOUPE_SIZE), interpolation=cv2.INTER_NEAREST)

    # Crosshair
    mid = LOUPE_SIZE // 2
    cv2.line(zoomed, (mid, 0), (mid, LOUPE_SIZE), (0, 200, 255), 1)
    cv2.line(zoomed, (0, mid), (LOUPE_SIZE, mid), (0, 200, 255), 1)

    # Position loupe away from cursor (top-right by default, flip if near edge)
    ox = sx + 20
    oy = sy - 20 - LOUPE_SIZE
    if ox + LOUPE_SIZE > win_w:
        ox = sx - 20 - LOUPE_SIZE
    if oy < 0:
        oy = sy + 20
    # Keep loupe above the status bar
    oy = min(oy, win_h - STATUS_BAR_H - LOUPE_SIZE)
    ox = max(0, min(ox, win_w - LOUPE_SIZE))
    oy = max(0, oy)

    if ox + LOUPE_SIZE <= win_w and oy + LOUPE_SIZE <= win_h:
        canvas[oy:oy + LOUPE_SIZE, ox:ox + LOUPE_SIZE] = zoomed
        cv2.rectangle(canvas, (ox, oy), (ox + LOUPE_SIZE, oy + LOUPE_SIZE), (0, 200, 255), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(1)

    H, W = img.shape[:2]

    # Window size — fit image but cap at a sensible max; enforce a minimum
    WIN_W = max(MIN_WIN_W, min(1400, W))
    WIN_H = max(MIN_WIN_H, min(900, H + STATUS_BAR_H))

    # Initial zoom: fit image in the image area (canvas minus status bar)
    avail_h = WIN_H - STATUS_BAR_H
    scale = min(WIN_W / W, avail_h / H)
    view_x = -(WIN_W / scale - W) / 2  # centre image horizontally
    view_y = -(avail_h / scale - H) / 2  # centre image vertically

    points = []        # corners in image coords (floats)
    dragging = None    # index of corner currently being dragged
    panning = False
    pan_start = None   # (screen_x, screen_y, view_x_at_start, view_y_at_start)
    loupe_pos = None   # screen pos to show loupe at, or None
    cursor_sx = WIN_W // 2  # last known cursor screen position (for keyboard zoom)
    cursor_sy = WIN_H // 2
    dirty = True

    window = "pick_corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, WIN_W, WIN_H)

    def redraw():
        frame = render(img, WIN_W, WIN_H, view_x, view_y, scale, points, dragging, loupe_pos)
        cv2.imshow(window, frame)

    def find_corner_near(sx, sy):
        """Return index of corner within GRAB_RADIUS of screen pos, or None."""
        for i, (px, py) in enumerate(points):
            sp = i2s(px, py, view_x, view_y, scale)
            if abs(sp[0] - sx) <= GRAB_RADIUS_PX and abs(sp[1] - sy) <= GRAB_RADIUS_PX:
                return i
        return None

    def on_mouse(event, sx, sy, flags, _param):
        nonlocal dragging, panning, pan_start, view_x, view_y, scale, loupe_pos, dirty

        nonlocal cursor_sx, cursor_sy
        cursor_sx, cursor_sy = sx, sy

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = find_corner_near(sx, sy)
            if hit is not None:
                dragging = hit
                loupe_pos = (sx, sy)
                dirty = True
            elif len(points) < 4:
                ix, iy = s2i(sx, sy, view_x, view_y, scale)
                ix = max(0.0, min(float(W - 1), ix))
                iy = max(0.0, min(float(H - 1), iy))
                points.append((ix, iy))
                loupe_pos = None
                dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging is not None:
                ix, iy = s2i(sx, sy, view_x, view_y, scale)
                ix = max(0.0, min(float(W - 1), ix))
                iy = max(0.0, min(float(H - 1), iy))
                points[dragging] = (ix, iy)
                loupe_pos = (sx, sy)
                dirty = True
            elif panning:
                ox, oy, vx0, vy0 = pan_start
                view_x = vx0 - (sx - ox) / scale
                view_y = vy0 - (sy - oy) / scale
                dirty = True
            else:
                # Show loupe when near a corner or still placing
                hit = find_corner_near(sx, sy)
                new_loupe = (sx, sy) if (hit is not None or len(points) < 4) else None
                if new_loupe != loupe_pos:
                    loupe_pos = new_loupe
                    dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging is not None:
                dragging = None
                loupe_pos = None
                dirty = True

        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            panning = True
            pan_start = (sx, sy, view_x, view_y)

        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            panning = False
            pan_start = None


    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        # Detect window resize and update render dimensions
        try:
            rect = cv2.getWindowImageRect(window)
            if rect[2] >= MIN_WIN_W and rect[3] >= MIN_WIN_H:
                if rect[2] != WIN_W or rect[3] != WIN_H:
                    WIN_W, WIN_H = rect[2], rect[3]
                    dirty = True
        except Exception:
            pass

        if dirty:
            redraw()
            dirty = False

        key = cv2.waitKey(16) & 0xFF  # ~60 fps poll

        if key in (ord('q'), 27):
            print("Quit without output.")
            break

        elif key in (ord('+'), ord('='), ord('-')):
            factor = ZOOM_FACTOR if key in (ord('+'), ord('=')) else 1.0 / ZOOM_FACTOR
            new_scale = max(MIN_ZOOM, min(MAX_ZOOM, scale * factor))
            ix, iy = s2i(cursor_sx, cursor_sy, view_x, view_y, scale)
            view_x = ix - cursor_sx / new_scale
            view_y = iy - cursor_sy / new_scale
            scale = new_scale
            dirty = True

        elif key == ord('r'):
            points.clear()
            dragging = None
            loupe_pos = None
            dirty = True

        elif key == 13 and len(points) == 4:  # Enter
            pts = [(round(px), round(py)) for px, py in points]
            print("\n# Copy-paste into configs/meters.yaml under the relevant meter:\n")
            print("    perspective_src_points:")
            for pt, label in zip(pts, CORNER_LABELS):
                print(f"      - [{pt[0]}, {pt[1]}]             # {label}")
            print()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
