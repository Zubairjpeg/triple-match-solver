"""
Triple Match 3D Solver — Phase 1 (manual input)

Usage:
    python main.py board.txt
    python main.py board.txt --cv        # Phase 2: CV detection (not yet implemented)

Input formats
-------------
Format A — grid (LAYER keyword detected):

    # overlap_radius: 0.9
    LAYER 1
    tomato  carrot  salmon  chicken  beef
    carrot  onion   .       corn     shrimp
    LAYER 2
    .  salmon  chicken  .  .

Format B — list (type, layer, row, col):

    tomato, 1, 0, 0
    carrot, 1, 0, 1
    salmon, 2, 0.5, 0.5
"""

from __future__ import annotations
import sys
import re
from board import Tile, Board, GameState
from solver import Solver


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_config(lines_raw: list) -> dict:
    config = {}
    for line in lines_raw:
        line = line.strip()
        if not line.startswith('#'):
            continue
        m = re.match(r'#\s*overlap_radius\s*:\s*([0-9.]+)', line)
        if m:
            config['overlap_radius'] = float(m.group(1))
    return config


def _parse_grid_format(content_lines: list, overlap_radius: float) -> list:
    tiles = []
    tile_id = 0
    current_layer = None
    current_row = 0
    for line in content_lines:
        if line.upper().startswith('LAYER'):
            current_layer = int(line.split()[1]) - 1  # 0-indexed
            current_row = 0
            continue
        cells = line.split()
        for col_idx, cell in enumerate(cells):
            if cell != '.':
                tiles.append(Tile(
                    tile_id=tile_id,
                    tile_type=cell.lower().replace('-', '_').replace(' ', '_'),
                    layer=current_layer,
                    row=float(current_row),
                    col=float(col_idx),
                ))
                tile_id += 1
        current_row += 1
    return tiles


def _parse_list_format(content_lines: list, overlap_radius: float) -> list:
    tiles = []
    for tile_id, line in enumerate(content_lines):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 4:
            raise ValueError(f"Expected 4 comma-separated values, got: {line!r}")
        tile_type = parts[0].lower().replace('-', '_').replace(' ', '_')
        layer = int(parts[1]) - 1  # 0-indexed
        row = float(parts[2])
        col = float(parts[3])
        tiles.append(Tile(tile_id=tile_id, tile_type=tile_type, layer=layer, row=row, col=col))
    return tiles


def parse_input(filepath: str) -> GameState:
    with open(filepath) as f:
        raw = f.readlines()

    config = _parse_config(raw)
    overlap_radius = config.get('overlap_radius', 0.9)

    content_lines = [l.strip() for l in raw if l.strip() and not l.strip().startswith('#')]

    if content_lines and content_lines[0].upper().startswith('LAYER'):
        tiles = _parse_grid_format(content_lines, overlap_radius)
    else:
        tiles = _parse_list_format(content_lines, overlap_radius)

    board = Board(tiles, overlap_radius=overlap_radius)
    return GameState(board=board, tray=(), moves=[])


# ---------------------------------------------------------------------------
# CV seam — Phase 2 hook
# ---------------------------------------------------------------------------

def load_game_state(source: str, use_cv: bool = False) -> GameState:
    if use_cv:
        from cv_detector import CVBoardDetector  # noqa: F401 — Phase 2
        return CVBoardDetector(template_dir='templates/').detect(source)
    return parse_input(source)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    use_cv = '--cv' in sys.argv

    state = load_game_state(source, use_cv=use_cv)
    total = len(state.board.tiles)
    accessible = len(state.board.accessible_tiles())
    tile_counts = state.board.type_counts()
    print(f"Board loaded: {total} tiles total, {accessible} accessible")
    print(f"Tile types ({len(tile_counts)}): {dict(sorted(tile_counts.items()))}")
    print()

    solver = Solver(state)
    print("Solving...", flush=True)
    solution = solver.solve()

    if solution is None:
        print("No solution found. Check your board input for accuracy.")
    else:
        print(f"Solution found in {len(solution)} moves:\n")
        for move in solution:
            print(move)


if __name__ == '__main__':
    main()
