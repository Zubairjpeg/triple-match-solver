from __future__ import annotations
from collections import Counter
from typing import Optional, List
from board import Tile, Board, GameState


def _clear_triplets(tray: list) -> list:
    counts = Counter(tray)
    to_clear = {t for t, c in counts.items() if c >= 3}
    result = []
    removed = Counter()
    for item in tray:
        if item in to_clear and removed[item] < 3:
            removed[item] += 1
        else:
            result.append(item)
    return result


class Solver:
    def __init__(self, initial_state: GameState):
        self.initial_state = initial_state
        self.visited: set = set()

    def solve(self) -> Optional[List[str]]:
        return self._dfs(self.initial_state)

    def _dfs(self, state: GameState) -> Optional[List[str]]:
        if state.is_won():
            return state.moves

        canonical = state.canonical_state()
        if canonical in self.visited:
            return None
        self.visited.add(canonical)

        if state.is_lost():
            return None

        if self._has_orphan(state):
            return None

        candidates = self._ordered_candidates(state)
        if not candidates:
            # tray not empty but no accessible tiles and no win → stuck
            return None

        for tile in candidates:
            next_state = self._apply_move(state, tile)
            result = self._dfs(next_state)
            if result is not None:
                return result

        return None

    def _has_orphan(self, state: GameState) -> bool:
        tray_counts = state.tray_counts()
        board_counts = state.board.type_counts()
        for tile_type, tray_k in tray_counts.items():
            needed = 3 - tray_k
            if board_counts.get(tile_type, 0) < needed:
                return True
        return False

    def _ordered_candidates(self, state: GameState) -> List[Tile]:
        accessible = state.board.accessible_tiles()
        tray_counts = state.tray_counts()

        def priority(tile: Tile) -> tuple:
            k = tray_counts.get(tile.tile_type, 0)
            if k == 2:
                p = 0   # completes a triplet
            elif k == 1:
                p = 1   # extends a pair
            else:
                p = 2   # new type
            return (p, -tile.layer)  # higher layer tiles unblock more when removed

        return sorted(accessible, key=priority)

    def _apply_move(self, state: GameState, tile: Tile) -> GameState:
        new_board = state.board.remove(tile)
        new_tray_list = list(state.tray) + [tile.tile_type]
        cleared_types = {t for t, c in Counter(new_tray_list).items() if c >= 3}
        final_tray = _clear_triplets(new_tray_list)

        n = len(state.moves) + 1
        move = (
            f"Move {n}: Tap {tile.tile_type.upper()} "
            f"(Layer {tile.layer + 1}, Row {tile.row}, Col {tile.col})"
            f" -> Tray: {final_tray}"
        )
        if cleared_types:
            move += f"  [CLEARED: {', '.join(t.upper() for t in sorted(cleared_types))}]"

        return GameState(
            board=new_board,
            tray=tuple(final_tray),
            moves=state.moves + [move],
        )
