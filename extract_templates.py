"""
Auto-extract accessible tile crops from a game screenshot.

Accessible (top-layer) tiles have cream/light backgrounds; blocked tiles have
grey/dark backgrounds. We threshold on the cream color in HSV, find rectangular
contours of tile-like size, and save numbered crops.

Usage:
    python extract_templates.py <screenshot_path> <output_dir>
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np


def find_accessible_tiles(image_bgr: np.ndarray) -> list:
    """Returns list of (x, y, w, h) for cream-colored tile rectangles."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Cream / light beige: low saturation, high value, warm hue
    lower = np.array([5, 10, 200])
    upper = np.array([40, 110, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Close small gaps so tile body is one blob
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image_bgr.shape[:2]
    img_area = h_img * w_img

    tiles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < img_area * 0.001 or area > img_area * 0.05:
            continue
        aspect = w / h if h > 0 else 0
        if not (0.6 < aspect < 1.5):
            continue
        # Reject contours that don't fill their bbox (e.g. ring shapes)
        contour_area = cv2.contourArea(c)
        if contour_area / area < 0.6:
            continue
        tiles.append((x, y, w, h))

    return tiles


def main(screenshot_path: str, output_dir: str) -> None:
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"ERROR: cannot read {screenshot_path}", file=sys.stderr)
        sys.exit(1)

    tiles = find_accessible_tiles(img)
    print(f"{screenshot_path}: found {len(tiles)} candidate tile crops")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(screenshot_path).stem
    for i, (x, y, w, h) in enumerate(tiles):
        crop = img[y:y + h, x:x + w]
        crop_path = out / f"{stem}_{i:03d}_x{x}_y{y}_w{w}_h{h}.png"
        cv2.imwrite(str(crop_path), crop)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
