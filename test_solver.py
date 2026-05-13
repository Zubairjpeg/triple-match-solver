"""Quick verification tests for the solver."""
import sys
sys.path.insert(0, '.')

from board import Tile, Board, GameState
from solver import Solver, _clear_triplets


def test_accessibility():
    # Tile at layer 0 is blocked by tile at layer 1 at same position
    tiles = [
        Tile(0, 'carrot', 0, 0.0, 0.0),
        Tile(1, 'tomato', 1, 0.0, 0.0),  # directly above
        Tile(2, 'salmon', 0, 2.0, 0.0),  # separate, not blocked
    ]
    board = Board(tiles)
    accessible = board.accessible_tiles()
    accessible_ids = {t.tile_id for t in accessible}
    assert 0 not in accessible_ids, "carrot should be blocked"
    assert 1 in accessible_ids, "tomato (top) should be accessible"
    assert 2 in accessible_ids, "salmon should be accessible"
    print("PASS: accessibility")


def test_half_offset_blocking():
    # Upper tile at half-offset should still block lower tile
    lower = Tile(0, 'beef', 0, 1.0, 1.0)
    upper = Tile(1, 'onion', 1, 1.5, 1.5)  # offset by 0.5 on both axes
    board = Board([lower, upper])
    assert board.is_blocked(lower), "beef should be blocked by half-offset onion"
    assert not board.is_blocked(upper), "onion (top) should be accessible"
    print("PASS: half-offset blocking")


def test_orphan_detection():
    # Tray has 2 carrots, 0 carrots on board → orphan
    tiles = [Tile(0, 'tomato', 0, 0.0, 0.0), Tile(1, 'tomato', 0, 1.0, 0.0)]
    state = GameState(
        board=Board(tiles),
        tray=('carrot', 'carrot'),
        moves=[]
    )
    solver = Solver(state)
    assert solver._has_orphan(state), "should detect carrot orphan"

    # Tray has 2 carrots, 1 carrot on board → fine (can complete)
    tiles2 = tiles + [Tile(2, 'carrot', 0, 2.0, 0.0)]
    state2 = GameState(board=Board(tiles2), tray=('carrot', 'carrot'), moves=[])
    assert not solver._has_orphan(state2), "one carrot on board should satisfy the triplet"
    print("PASS: orphan detection")


def test_clear_triplets():
    result = _clear_triplets(['a', 'b', 'a', 'a', 'c'])
    assert result == ['b', 'c'], f"Expected ['b', 'c'], got {result}"
    result2 = _clear_triplets(['a', 'a', 'b'])
    assert result2 == ['a', 'a', 'b'], "no triplet, should be unchanged"
    print("PASS: clear triplets")


def test_tray_triplet_clears_on_move():
    # Tray [carrot, carrot], tap carrot → should clear and win
    tiles = [Tile(0, 'carrot', 0, 0.0, 0.0)]
    state = GameState(board=Board(tiles), tray=('carrot', 'carrot'), moves=[])
    solver = Solver(state)
    solution = solver.solve()
    assert solution is not None, "should find 1-move solution"
    assert len(solution) == 1
    assert 'CLEARED: CARROT' in solution[0]
    print("PASS: tray triplet clears on move")


def test_solve_toy_board():
    # 9 tiles, 3 types × 3, single layer — trivially solvable
    tiles = []
    tid = 0
    for t in ['carrot', 'tomato', 'salmon']:
        for i in range(3):
            tiles.append(Tile(tid, t, 0, float(tid), 0.0))
            tid += 1
    state = GameState(board=Board(tiles), tray=(), moves=[])
    solver = Solver(state)
    solution = solver.solve()
    assert solution is not None, "toy board should be solvable"
    assert len(solution) == 9
    print(f"PASS: toy board solved in {len(solution)} moves")


def test_tray_overflow_loses():
    # 7 different tile types in tray — lost state
    tray = tuple(f'type{i}' for i in range(7))
    state = GameState(board=Board([]), tray=tray, moves=[])
    assert state.is_lost()
    print("PASS: tray overflow is lost state")


if __name__ == '__main__':
    test_accessibility()
    test_half_offset_blocking()
    test_orphan_detection()
    test_clear_triplets()
    test_tray_triplet_clears_on_move()
    test_solve_toy_board()
    test_tray_overflow_loses()
    print("\nAll tests passed.")
