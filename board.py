from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import List


@dataclass(frozen=True)
class Tile:
    tile_id: int
    tile_type: str   # normalized lowercase, e.g. "carrot"
    layer: int       # 0 = bottom layer
    row: float       # float to support half-offset Mahjong-style stacking
    col: float

    def __hash__(self):
        return hash(self.tile_id)

    def __eq__(self, other):
        return isinstance(other, Tile) and self.tile_id == other.tile_id


class Board:
    def __init__(self, tiles: List[Tile], overlap_radius: float = 0.9):
        # overlap_radius: tile centers must be closer than this on BOTH axes to block
        # 0.9 handles direct stacking (dr=0) and half-offset (dr=0.5), excludes adjacent (dr=1.0)
        self.tiles: frozenset[Tile] = frozenset(tiles)
        self.overlap_radius = overlap_radius

    def is_blocked(self, tile: Tile) -> bool:
        for other in self.tiles:
            if other.layer <= tile.layer:
                continue
            if (abs(other.row - tile.row) < self.overlap_radius and
                    abs(other.col - tile.col) < self.overlap_radius):
                return True
        return False

    def accessible_tiles(self) -> List[Tile]:
        return [t for t in self.tiles if not self.is_blocked(t)]

    def remove(self, tile: Tile) -> Board:
        return Board([t for t in self.tiles if t.tile_id != tile.tile_id], self.overlap_radius)

    def type_counts(self) -> Counter:
        return Counter(t.tile_type for t in self.tiles)


@dataclass
class GameState:
    board: Board
    tray: tuple          # tuple[str, ...], max TRAY_MAX items
    moves: list          # list[str], human-readable log

    TRAY_MAX: int = 7

    def tray_counts(self) -> Counter:
        return Counter(self.tray)

    def is_won(self) -> bool:
        return len(self.board.tiles) == 0 and len(self.tray) == 0

    def is_lost(self) -> bool:
        return len(self.tray) >= self.TRAY_MAX and not any(
            v >= 3 for v in self.tray_counts().values()
        )

    def canonical_state(self) -> tuple:
        board_key = tuple(sorted(t.tile_id for t in self.board.tiles))
        tray_key = tuple(sorted(self.tray_counts().items()))
        return (board_key, tray_key)
