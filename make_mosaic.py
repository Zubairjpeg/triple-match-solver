"""Build a labeled mosaic of all crops so they can be identified in one pass."""
import sys
from pathlib import Path
import cv2
import numpy as np


def main(staging_dir: str, output: str, tile_size: int = 130, cols: int = 8):
    paths = sorted(Path(staging_dir).glob("*.png"))
    if not paths:
        print("No crops found.")
        sys.exit(1)

    rows = (len(paths) + cols - 1) // cols
    cell_w = tile_size + 12
    cell_h = tile_size + 32  # extra space for label

    mosaic = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y = r * cell_h + 4
        x = c * cell_w + 6
        mosaic[y:y + tile_size, x:x + tile_size] = img
        label = f"{i:03d}"
        cv2.putText(mosaic, label, (x + 2, y + tile_size + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output, mosaic)
    print(f"Wrote {output} with {len(paths)} crops")
    # Also print the index->filename map
    for i, p in enumerate(paths):
        print(f"{i:03d}  {p.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
