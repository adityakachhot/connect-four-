"""
tests/test_tictactoe_game.py
=============================
Unit tests for TicTacToeGame class:
  - Win detection in all directions (horizontal rows 0..2, vertical cols 0..2, main & anti diagonals)
  - Draw detection on full board
  - Invalid move handling and pass action behavior
  - Canonical form transformations for both players
  - Dihedral group D4 symmetries (8 transforms) and policy preservation
"""

import os
import sys
import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.games.tic_tac_toe import TicTacToeGame


@pytest.fixture
def game():
    return TicTacToeGame()


# ─── Win Detection Tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("row", [0, 1, 2])
def test_win_horizontal(game, row):
    # Player 1 wins on row
    b1 = game.getInitBoard()
    b1[row, :] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins on row
    b2 = game.getInitBoard()
    b2[row, :] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


@pytest.mark.parametrize("col", [0, 1, 2])
def test_win_vertical(game, col):
    # Player 1 wins on col
    b1 = game.getInitBoard()
    b1[:, col] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins on col
    b2 = game.getInitBoard()
    b2[:, col] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


def test_win_main_diagonal(game):
    # Player 1 wins on main diagonal (↘)
    b1 = game.getInitBoard()
    np.fill_diagonal(b1, 1)
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins on main diagonal (↘)
    b2 = game.getInitBoard()
    np.fill_diagonal(b2, -1)
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


def test_win_anti_diagonal(game):
    # Player 1 wins on anti-diagonal (↙)
    b1 = game.getInitBoard()
    b1[0, 2] = 1
    b1[1, 1] = 1
    b1[2, 0] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins on anti-diagonal (↙)
    b2 = game.getInitBoard()
    b2[0, 2] = -1
    b2[1, 1] = -1
    b2[2, 0] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


# ─── Draw Detection Tests ────────────────────────────────────────────────────

def test_draw_cat_game(game):
    # Classic draw board:
    #  X  O  X
    #  X  O  O
    #  O  X  X
    draw_board = np.array([
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1]
    ], dtype=np.int8)

    assert not game._has_won(draw_board, 1)
    assert not game._has_won(draw_board, -1)
    res = game.getGameEnded(draw_board, 1)
    assert 0 < abs(res) < 1e-3  # 1e-4 draw sentinel


# ─── Invalid Moves Tests ─────────────────────────────────────────────────────

def test_initial_valid_moves(game):
    b = game.getInitBoard()
    valids = game.getValidMoves(b, 1)
    assert len(valids) == 10  # 9 cells + 1 pass
    assert np.all(valids[:9] == 1)
    assert valids[9] == 0  # Pass is invalid on empty board


def test_occupied_cell_becomes_invalid(game):
    b = game.getInitBoard()
    b[1, 1] = 1  # Center occupied
    valids = game.getValidMoves(b, -1)
    assert valids[4] == 0  # Center action 4 is invalid
    assert np.sum(valids[:9]) == 8


def test_occupied_cell_raises_value_error(game):
    b = game.getInitBoard()
    b[0, 0] = 1
    # Attempting to play on an already occupied (invalid) cell must raise ValueError
    with pytest.raises(ValueError, match="already occupied"):
        game.getNextState(b, -1, 0)


def test_full_board_pass_action_valid(game):
    full_board = np.ones((3, 3), dtype=np.int8)
    valids = game.getValidMoves(full_board, 1)
    assert np.all(valids[:9] == 0)
    assert valids[9] == 1  # Pass action is valid


# ─── Canonical Form Tests ────────────────────────────────────────────────────

def test_canonical_form(game):
    b = np.array([
        [1, -1, 0],
        [0,  1, 0],
        [0,  0, 0]
    ], dtype=np.int8)

    canon_p1 = game.getCanonicalForm(b, 1)
    np.testing.assert_array_equal(canon_p1, b)

    canon_p2 = game.getCanonicalForm(b, -1)
    np.testing.assert_array_equal(canon_p2, -b)

    # In canonical form, current player's pieces are always +1
    assert canon_p1[0, 0] == 1
    assert canon_p2[0, 1] == 1


# ─── Symmetry Tests ──────────────────────────────────────────────────────────

def test_dihedral_symmetries(game):
    b = np.zeros((3, 3), dtype=np.int8)
    b[0, 0] = 1  # Top-left corner
    pi = np.zeros(10, dtype=np.float32)
    pi[0] = 0.7  # Top-left corner
    pi[4] = 0.3  # Center

    syms = game.getSymmetries(b, pi)
    assert len(syms) == 8  # 4 rotations x 2 reflections

    for s_b, s_pi in syms:
        assert s_b.shape == (3, 3)
        assert len(s_pi) == 10
        # Center piece / policy is invariant under D4
        assert s_b[1, 1] == 0
        assert pytest.approx(s_pi[4]) == 0.3
        # Total policy distribution sums to 1.0
        assert pytest.approx(np.sum(s_pi)) == 1.0
