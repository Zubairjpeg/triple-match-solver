"""
Template-matching tile detector.

Pipeline:
  1. Load PNG templates from a directory (filename stem = tile type).
  2. Multi-scale cv2.matchTemplate against the screenshot.
  3. Non-max suppression to dedupe overlapping matches.
  4. 1D clustering of x/y centers to snap matches onto a grid.
  5. Return a GameState ready for the solver.
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from board import Tile, Board, GameState


MATCH_THRESHOLD = 0.55
NMS_OVERLAP = 0.4
SCALE_RANGE = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.2, 1.4, 1.7, 2.0)


def load_templates(templates_dir: Path) -> dict:
    """Load all PNG templates from a directory. Returns {tile_type: grayscale_array}."""
    templates = {}
    for path in sorted(Path(templates_dir).glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[path.stem.lower()] = img
    return templates


def _nms(matches: list, overlap_thresh: float = NMS_OVERLAP) -> list:
    """Non-max suppression. matches: list of (score, x, y, w, h, name)."""
    if not matches:
        return []
    matches = sorted(matches, key=lambda m: m[0], reverse=True)
    kept = []
    for m in matches:
        _, x, y, w, h, _ = m
        overlap = False
        for k in kept:
            _, kx, ky, kw, kh, _ = k
            xi1, yi1 = max(x, kx), max(y, ky)
            xi2, yi2 = min(x + w, kx + kw), min(y + h, ky + kh)
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            inter = (xi2 - xi1) * (yi2 - yi1)
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > overlap_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(m)
    return kept


def _match_one_template(gray: np.ndarray, template: np.ndarray, name: str) -> list:
    """Run matchTemplate across all scales. Returns raw matches before NMS."""
    matches = []
    th, tw = template.shape
    for scale in SCALE_RANGE:
        sw, sh = int(tw * scale), int(th * scale)
        if sw < 10 or sh < 10:
            continue
        if sw > gray.shape[1] or sh > gray.shape[0]:
            continue
        scaled = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= MATCH_THRESHOLD)
        for x, y in zip(xs, ys):
            matches.append((float(result[y, x]), int(x), int(y), sw, sh, name))
    return matches


def detect_tile_boxes(image_bgr: np.ndarray, templates: dict) -> list:
    """
    Run template matching against an image. Returns deduped list of:
        [(score, x, y, w, h, tile_type), ...]
    """
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    all_matches = []
    for name, template in templates.items():
        all_matches.extend(_match_one_template(gray, template, name))

    return _nms(all_matches)


def _cluster_1d(values: list, gap: float) -> list:
    """1D clustering: group values that are within `gap` of any neighbor. Returns cluster means sorted."""
    if not values:
        return []
    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _assign_grid(matches: list) -> list:
    """Snap each match to a row/col by clustering centers. Returns dicts with row/col/type/score."""
    if not matches:
        return []
    avg_w = sum(m[3] for m in matches) / len(matches)
    avg_h = sum(m[4] for m in matches) / len(matches)
    gap_x = avg_w * 0.55
    gap_y = avg_h * 0.55

    centers_x = [m[1] + m[3] / 2 for m in matches]
    centers_y = [m[2] + m[4] / 2 for m in matches]
    col_centers = _cluster_1d(centers_x, gap_x)
    row_centers = _cluster_1d(centers_y, gap_y)

    tiles = []
    for tile_id, m in enumerate(matches):
        score, x, y, w, h, name = m
        cx, cy = x + w / 2, y + h / 2
        col = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - cx))
        row = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - cy))
        tiles.append({
            "id": tile_id,
            "tile_type": name,
            "row": row,
            "col": col,
            "score": score,
            "bbox": (x, y, w, h),
        })
    return tiles


def detect_board(image_bgr: np.ndarray, templates_dir: Path, overlap_radius: float = 0.9):
    """
    Detect a Triple Match board from a screenshot.

    Returns (GameState, debug_info) where debug_info contains:
        - "boxes": detected bounding boxes for overlay rendering
        - "raw_matches": pre-NMS matches (for debugging)
    """
    templates = load_templates(templates_dir)
    if not templates:
        raise ValueError(f"No templates found in {templates_dir}")

    matches = detect_tile_boxes(image_bgr, templates)
    grid_tiles = _assign_grid(matches)

    tile_objs = [
        Tile(
            tile_id=t["id"],
            tile_type=t["tile_type"],
            layer=0,  # flat detection — solver treats all as accessible
            row=float(t["row"]),
            col=float(t["col"]),
        )
        for t in grid_tiles
    ]
    board = Board(tile_objs, overlap_radius=overlap_radius)
    state = GameState(board=board, tray=(), moves=[])
    debug = {"boxes": grid_tiles, "match_count": len(matches)}
    return state, debug


def annotate_image(image_bgr: np.ndarray, debug_info: dict, move_order: dict = None) -> np.ndarray:
    """
    Draw detected tile boxes onto the image.
    move_order: optional {tile_id: move_number} to overlay numbered circles.
    """
    out = image_bgr.copy()
    move_order = move_order or {}
    for box in debug_info.get("boxes", []):
        x, y, w, h = box["bbox"]
        name = box["tile_type"]
        score = box["score"]
        color = (0, 255, 0) if score > 0.75 else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{name}"
        cv2.putText(out, label, (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if box["id"] in move_order:
            n = move_order[box["id"]]
            center = (x + w // 2, y + h // 2)
            cv2.circle(out, center, 18, (0, 0, 255), 3)
            cv2.putText(out, str(n), (center[0] - 8, center[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out
