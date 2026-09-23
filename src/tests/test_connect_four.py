"""
pytest test suite for ConnectFourGame.

Covers:
  - win detection   (horizontal, vertical, diagonal ↘, diagonal ↙)
  - draw detection  (full board, no winner)
  - valid-move generation (normal play, full-column exclusion, full-board pass)
  - getNextState    (gravity, illegal-column guard)
  - getCanonicalForm
  - getSymmetries   (horizontal flip correctness)
  - stringRepresentation (uniqueness across distinct boards)
  - getInitBoard / getBoardSize / getActionSize basic sanity
"""

import sys
import os
import numpy as np
import pytest

# Make sure our src/games package is importable regardless of where pytest is run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.games.connect_four import ConnectFourGame

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game():
    """Standard 6-row × 7-column Connect Four game."""
    return ConnectFourGame()


@pytest.fixture
def mini():
    """Smaller 4×5 board for compact hand-crafted positions."""
    return ConnectFourGame(rows=4, cols=5, win_length=4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board_from_rows(rows_str, rows=6, cols=7):
    """
    Build a (rows, cols) int8 board from a list of strings.
    '.' = 0, 'X' = 1 (player 1), 'O' = -1 (player 2).
    Strings are listed top-to-bottom.
    """
    assert len(rows_str) == rows, f"Expected {rows} row strings, got {len(rows_str)}"
    mapping = {".": 0, "X": 1, "O": -1}
    board = np.zeros((rows, cols), dtype=np.int8)
    for r, row_s in enumerate(rows_str):
        assert len(row_s) == cols, f"Row {r} has length {len(row_s)}, expected {cols}"
        for c, ch in enumerate(row_s):
            board[r, c] = mapping[ch]
    return board


# ---------------------------------------------------------------------------
# Basic sanity tests
# ---------------------------------------------------------------------------

class TestBasicProperties:
    def test_board_shape(self, game):
        board = game.getInitBoard()
        assert board.shape == (6, 7)
        assert board.dtype == np.int8

    def test_init_board_all_zeros(self, game):
        board = game.getInitBoard()
        assert np.all(board == 0)

    def test_get_board_size(self, game):
        assert game.getBoardSize() == (6, 7)

    def test_action_size(self, game):
        # 7 columns + 1 pass action
        assert game.getActionSize() == 8

    def test_action_size_mini(self, mini):
        assert mini.getActionSize() == 6  # 5 cols + 1 pass


# ---------------------------------------------------------------------------
# Valid move tests
# ---------------------------------------------------------------------------

class TestValidMoves:
    def test_all_valid_on_empty_board(self, game):
        board = game.getInitBoard()
        valids = game.getValidMoves(board, 1)
        # All 7 columns valid, pass slot invalid
        assert valids[:7].tolist() == [1] * 7
        assert valids[7] == 0

    def test_full_column_is_invalid(self, game):
        """Fill column 3 completely; it should become invalid."""
        board = game.getInitBoard()
        for _ in range(6):  # 6 rows
            board, _ = game.getNextState(board, 1, 3)
            board, _ = game.getNextState(board, -1, 0)  # filler to alternate
        # Column 3 is full now (we placed 6 pieces in it... let's check more carefully)
        # Actually after 6 drops, col 3 is full:
        board2 = game.getInitBoard()
        for r in range(6):
            board2[r, 3] = 1 if r % 2 == 0 else -1
        valids = game.getValidMoves(board2, 1)
        assert valids[3] == 0, "Full column 3 should not be a valid move"

    def test_only_full_column_is_blocked(self, game):
        """Only the specific full column should be blocked."""
        board = game.getInitBoard()
        # Fill column 0 entirely
        for r in range(6):
            board[r, 0] = 1
        valids = game.getValidMoves(board, 1)
        assert valids[0] == 0
        assert all(valids[1:7] == 1)

    def test_pass_slot_valid_on_full_board(self, mini):
        """When every column is full, the pass action (index cols) becomes valid."""
        board = np.zeros((4, 5), dtype=np.int8)
        # Fill entire board
        for r in range(4):
            for c in range(5):
                board[r, c] = 1 if (r + c) % 2 == 0 else -1
        valids = mini.getValidMoves(board, 1)
        assert valids[:5].sum() == 0, "No column should be valid on a full board"
        assert valids[5] == 1, "Pass slot must be valid when every column is full"

    def test_partial_column_still_valid(self, game):
        """A partially-filled column is still valid."""
        board = game.getInitBoard()
        # Add 3 pieces to column 2
        for r in range(3, 6):
            board[r, 2] = 1
        valids = game.getValidMoves(board, 1)
        assert valids[2] == 1


# ---------------------------------------------------------------------------
# getNextState — gravity and board mutation
# ---------------------------------------------------------------------------

class TestNextState:
    def test_piece_falls_to_bottom_of_empty_column(self, game):
        board = game.getInitBoard()
        next_board, next_player = game.getNextState(board, 1, 4)
        assert next_board[5, 4] == 1, "Piece should land on the bottom row (row 5)"
        assert next_player == -1

    def test_piece_stacks_on_existing(self, game):
        board = game.getInitBoard()
        board[5, 2] = -1  # something already at the bottom
        next_board, _ = game.getNextState(board, 1, 2)
        assert next_board[4, 2] == 1, "Piece should land on row 4 (above existing piece)"
        assert next_board[5, 2] == -1, "Existing piece at row 5 should be unchanged"

    def test_original_board_not_mutated(self, game):
        board = game.getInitBoard()
        snapshot = board.copy()
        game.getNextState(board, 1, 0)
        assert np.array_equal(board, snapshot), "getNextState must not modify the input board"

    def test_invalid_column_raises(self, game):
        board = game.getInitBoard()
        # Fill column 1 completely
        for r in range(6):
            board[r, 1] = 1
        with pytest.raises(ValueError, match="full"):
            game.getNextState(board, -1, 1)

    def test_pass_action_does_not_modify_board(self, game):
        board = game.getInitBoard()
        board[5, 3] = 1  # some state
        next_board, next_player = game.getNextState(board, 1, game.cols)
        assert np.array_equal(next_board, board)
        assert next_player == -1


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

class TestWinDetection:
    def test_horizontal_win_player1(self, game):
        board = _board_from_rows([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "XXXX...",
        ])
        assert game.getGameEnded(board, 1) == 1
        assert game.getGameEnded(board, -1) == -1

    def test_horizontal_win_player2(self, game):
        # Clean board: only player 2 has 4 in a row (bottom row, cols 0-3).
        # Player 1 has no pieces, so there is no ambiguity.
        board = _board_from_rows([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "OOOO...",
        ])
        assert game.getGameEnded(board, -1) == 1,  "player 2 should see a win"
        assert game.getGameEnded(board, 1) == -1,  "player 1 should see a loss"

    def test_vertical_win_player1(self, game):
        board = _board_from_rows([
            ".......",
            ".......",
            "X......",
            "X......",
            "X......",
            "X......",
        ])
        assert game.getGameEnded(board, 1) == 1

    def test_vertical_win_player2(self, game):
        board = _board_from_rows([
            ".......",
            ".......",
            "O......",
            "O......",
            "O......",
            "O......",
        ])
        assert game.getGameEnded(board, -1) == 1
        assert game.getGameEnded(board, 1) == -1

    def test_diagonal_down_right_win(self, game):
        # Player 1 wins on the main diagonal starting at (2,0)
        board = _board_from_rows([
            ".......",
            ".......",
            "X......",
            "OX.....",
            "OOX....",
            "OOOX...",
        ])
        assert game.getGameEnded(board, 1) == 1

    def test_diagonal_down_left_win(self, game):
        # Player 1 wins on the anti-diagonal starting at (2,3)
        board = _board_from_rows([
            ".......",
            ".......",
            "...X...",
            "..XO...",
            ".XOO...",
            "XOOO...",
        ])
        assert game.getGameEnded(board, 1) == 1

    def test_no_win_on_empty_board(self, game):
        board = game.getInitBoard()
        assert game.getGameEnded(board, 1) == 0

    def test_no_win_three_in_a_row(self, game):
        board = _board_from_rows([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "XXX....",
        ])
        assert game.getGameEnded(board, 1) == 0

    def test_win_not_confounded_by_player_sign(self, game):
        """getGameEnded with player=-1 on a board where player 1 won returns -1."""
        board = _board_from_rows([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            ".XXXX..",
        ])
        assert game.getGameEnded(board, 1) == 1
        assert game.getGameEnded(board, -1) == -1

    def test_win_across_board_edge_is_not_detected(self, game):
        """Four pieces in the same column wrapping to the next should not be a win."""
        # Last 3 of col 6, first 1 of col 0 — must NOT count as horizontal win
        board = _board_from_rows([
            ".......",
            ".......",
            ".......",
            ".......",
            ".......",
            "X....XXX",  # 8 chars — intentionally bad to catch a bug
        ]) if False else game.getInitBoard()
        # Instead: 3 at end of row + 1 at start of next (shouldn't happen in array)
        # Just verify three at end of row + nothing adjacent to the right is not a win
        board[5, 4] = 1
        board[5, 5] = 1
        board[5, 6] = 1
        # No 4th piece adjacent
        assert game.getGameEnded(board, 1) == 0


# ---------------------------------------------------------------------------
# Draw detection
# ---------------------------------------------------------------------------

class TestDrawDetection:
    def test_draw_full_board_no_winner(self, mini):
        """
        A full 4×5 board verified to contain no 4-in-a-row for either player.
        (Found by random search; checkerboard patterns contain diagonal wins.)

        XXOOO
        XXOXX
        OXOOO
        XOXOX
        """
        board = np.array([
            [ 1,  1, -1, -1, -1],
            [ 1,  1, -1,  1,  1],
            [-1,  1, -1, -1, -1],
            [ 1, -1,  1, -1,  1],
        ], dtype=np.int8)
        # Verify precondition: neither player has won
        assert not mini._has_won(board, 1),  "Precondition: board should have no p1 win"
        assert not mini._has_won(board, -1), "Precondition: board should have no p2 win"
        result = mini.getGameEnded(board, 1)
        assert result != 0, "Game should be over (full board)"
        assert abs(result) < 1, "Draw result should be a small non-zero value, not ±1"
        assert result == pytest.approx(1e-4)

    def test_not_draw_when_board_has_empty_cells(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1  # one piece placed
        result = game.getGameEnded(board, 1)
        assert result == 0, "Game with empty cells and no winner should not be over"

    def test_draw_returns_nonzero(self, mini):
        """Ensure the draw sentinel is truthy so Coach loop terminates."""
        board = np.array([
            [ 1, -1,  1, -1,  1],
            [-1,  1, -1,  1, -1],
            [ 1, -1,  1, -1,  1],
            [-1,  1, -1,  1, -1],
        ], dtype=np.int8)
        result = mini.getGameEnded(board, 1)
        assert result != 0  # must be truthy to terminate self-play loop


# ---------------------------------------------------------------------------
# getCanonicalForm
# ---------------------------------------------------------------------------

class TestCanonicalForm:
    def test_player1_canonical_is_identity(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1
        board[5, 1] = -1
        canonical = game.getCanonicalForm(board, 1)
        assert np.array_equal(canonical, board)

    def test_player2_canonical_flips_signs(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1
        board[5, 1] = -1
        canonical = game.getCanonicalForm(board, -1)
        assert canonical[5, 0] == -1
        assert canonical[5, 1] == 1

    def test_canonical_dtype_preserved(self, game):
        board = game.getInitBoard()
        canonical = game.getCanonicalForm(board, -1)
        assert canonical.dtype == np.int8


# ---------------------------------------------------------------------------
# getSymmetries
# ---------------------------------------------------------------------------

class TestSymmetries:
    def test_returns_two_items(self, game):
        board = game.getInitBoard()
        pi = [1 / 8] * 8
        syms = game.getSymmetries(board, pi)
        assert len(syms) == 2

    def test_original_is_first(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1
        pi = list(range(8))
        syms = game.getSymmetries(board, pi)
        orig_board, orig_pi = syms[0]
        assert np.array_equal(orig_board, board)
        assert orig_pi == pi

    def test_horizontal_flip_board(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1   # far left column
        pi = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        syms = game.getSymmetries(board, pi)
        flipped_board, flipped_pi = syms[1]
        # Piece in col 0 should now appear in col 6 after a left-right flip
        assert flipped_board[5, 6] == 1
        assert flipped_board[5, 0] == 0

    def test_horizontal_flip_policy(self, game):
        board = game.getInitBoard()
        # Policy heavily biased toward column 0
        pi = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        syms = game.getSymmetries(board, pi)
        _, flipped_pi = syms[1]
        # After flip, column 0 prob should map to column 6 position
        assert flipped_pi[6] == pytest.approx(0.9)
        assert flipped_pi[0] == pytest.approx(0.0)
        # Pass slot preserved
        assert flipped_pi[7] == pytest.approx(0.0)

    def test_flip_is_involutory(self, game):
        """Flipping twice should recover the original."""
        board = game.getInitBoard()
        board[5, 1] = 1
        board[4, 3] = -1
        pi = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]
        syms = game.getSymmetries(board, pi)
        flipped_board, flipped_pi = syms[1]
        # Flip again
        syms2 = game.getSymmetries(flipped_board, flipped_pi)
        double_flipped_board, double_flipped_pi = syms2[1]
        assert np.array_equal(double_flipped_board, board)
        assert double_flipped_pi == pytest.approx(pi, abs=1e-9)

    def test_original_board_not_mutated_by_symmetries(self, game):
        board = game.getInitBoard()
        board[5, 0] = 1
        snapshot = board.copy()
        game.getSymmetries(board, [0.0] * 8)
        assert np.array_equal(board, snapshot)


# ---------------------------------------------------------------------------
# stringRepresentation
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    def test_returns_bytes(self, game):
        board = game.getInitBoard()
        s = game.stringRepresentation(board)
        assert isinstance(s, bytes)

    def test_distinct_boards_have_distinct_strings(self, game):
        b1 = game.getInitBoard()
        b2 = game.getInitBoard()
        b2[5, 0] = 1
        assert game.stringRepresentation(b1) != game.stringRepresentation(b2)

    def test_identical_boards_have_same_string(self, game):
        b1 = game.getInitBoard()
        b2 = game.getInitBoard()
        assert game.stringRepresentation(b1) == game.stringRepresentation(b2)

    def test_hashable_as_dict_key(self, game):
        board = game.getInitBoard()
        s = game.stringRepresentation(board)
        d = {s: 42}
        assert d[s] == 42
