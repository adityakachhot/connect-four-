"""
tests/test_connect_four_game.py
================================
Unit tests for ConnectFourGame class:
  - Win detection in all 4 directions (horizontal, vertical, diagonal, anti-diagonal)
  - Draw detection on full board
  - Invalid move handling and pass action behavior
  - Canonical form transformations for both players
  - Symmetry generation and policy preservation
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

from src.games.connect_four import ConnectFourGame


@pytest.fixture
def game():
    return ConnectFourGame()


# ─── Win Detection Tests ─────────────────────────────────────────────────────

def test_win_horizontal(game):
    # Player 1 wins horizontally along row 5 (bottom row)
    b1 = game.getInitBoard()
    for col in range(4):
        b1[5, col] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins horizontally along row 3
    b2 = game.getInitBoard()
    for col in range(2, 6):
        b2[3, col] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


def test_win_vertical(game):
    # Player 1 wins vertically in column 2
    b1 = game.getInitBoard()
    for row in range(2, 6):
        b1[row, 2] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins vertically in column 6
    b2 = game.getInitBoard()
    for row in range(0, 4):
        b2[row, 6] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


def test_win_diagonal_main(game):
    # Player 1 wins along main diagonal (↘)
    # (row, col): (2, 1), (3, 2), (4, 3), (5, 4)
    b1 = game.getInitBoard()
    for offset in range(4):
        b1[2 + offset, 1 + offset] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins along another main diagonal (↘)
    # (row, col): (0, 3), (1, 4), (2, 5), (3, 6)
    b2 = game.getInitBoard()
    for offset in range(4):
        b2[0 + offset, 3 + offset] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


def test_win_diagonal_anti(game):
    # Player 1 wins along anti-diagonal (↙)
    # (row, col): (2, 4), (3, 3), (4, 2), (5, 1)
    b1 = game.getInitBoard()
    for offset in range(4):
        b1[2 + offset, 4 - offset] = 1
    assert game.getGameEnded(b1, 1) == 1.0
    assert game.getGameEnded(b1, -1) == -1.0

    # Player -1 wins along anti-diagonal (↙)
    # (row, col): (1, 6), (2, 5), (3, 4), (4, 3)
    b2 = game.getInitBoard()
    for offset in range(4):
        b2[1 + offset, 6 - offset] = -1
    assert game.getGameEnded(b2, -1) == 1.0
    assert game.getGameEnded(b2, 1) == -1.0


# ─── Draw Detection Tests ────────────────────────────────────────────────────

def test_draw_full_board(game):
    # Construct a full board without 4-in-a-row
    # Repeated pattern:
    # 1 -1  1 -1  1 -1  1
    # 1 -1  1 -1  1 -1  1
    # -1 1 -1  1 -1  1 -1
    # -1 1 -1  1 -1  1 -1
    # 1 -1  1 -1  1 -1  1
    # 1 -1  1 -1  1 -1  1
    pattern = np.array([
        [ 1, -1,  1, -1,  1, -1,  1],
        [ 1, -1,  1, -1,  1, -1,  1],
        [-1,  1, -1,  1, -1,  1, -1],
        [-1,  1, -1,  1, -1,  1, -1],
        [ 1, -1,  1, -1,  1, -1,  1],
        [ 1, -1,  1, -1,  1, -1,  1],
    ], dtype=np.int8)

    assert not game._has_won(pattern, 1)
    assert not game._has_won(pattern, -1)
    res = game.getGameEnded(pattern, 1)
    assert 0 < abs(res) < 1e-3  # 1e-4 draw sentinel


# ─── Invalid Moves Tests ─────────────────────────────────────────────────────

def test_initial_valid_moves(game):
    b = game.getInitBoard()
    valids = game.getValidMoves(b, 1)
    assert len(valids) == 8
    assert np.all(valids[:7] == 1)
    assert valids[7] == 0  # Pass is invalid on empty board


def test_full_column_becomes_invalid(game):
    b = game.getInitBoard()
    col = 3
    # Fill column 3 with 6 alternating pieces
    for r in reversed(range(6)):
        b[r, col] = 1 if r % 2 == 0 else -1

    valids = game.getValidMoves(b, 1)
    assert valids[col] == 0
    # Other columns are still valid
    for c in range(7):
        if c != col:
            assert valids[c] == 1
    assert valids[7] == 0  # Pass still invalid


def test_invalid_move_raises_value_error(game):
    b = game.getInitBoard()
    col = 2
    # Fill column 2 completely
    for r in range(6):
        b[r, col] = 1

    # Attempting to drop into an invalid (full) column must raise ValueError
    with pytest.raises(ValueError, match="full"):
        game.getNextState(b, 1, col)


def test_full_board_pass_action_valid(game):
    full_board = np.ones((6, 7), dtype=np.int8)
    valids = game.getValidMoves(full_board, 1)
    assert np.all(valids[:7] == 0)
    assert valids[7] == 1  # Pass becomes valid when board is full


# ─── Canonical Form Tests ────────────────────────────────────────────────────

def test_canonical_form(game):
    b = game.getInitBoard()
    b[5, 3] = 1   # Player 1 at bottom center
    b[5, 4] = -1  # Player -1

    # Player 1's perspective: board is untouched
    canon_p1 = game.getCanonicalForm(b, 1)
    np.testing.assert_array_equal(canon_p1, b)

    # Player -1's perspective: signs are inverted
    canon_p2 = game.getCanonicalForm(b, -1)
    np.testing.assert_array_equal(canon_p2, -b)

    # In canonical form, current player's piece is ALWAYS +1
    assert canon_p1[5, 3] == 1   # P1's own piece
    assert canon_p2[5, 4] == 1   # P2's own piece is +1 in canon_p2


# ─── Symmetry Tests ──────────────────────────────────────────────────────────

def test_symmetries(game):
    b = game.getInitBoard()
    b[5, 0] = 1   # Disc at bottom-left
    pi = np.zeros(8, dtype=np.float32)
    pi[0] = 0.8   # Probability on column 0
    pi[6] = 0.2   # Probability on column 6

    syms = game.getSymmetries(b, pi)
    assert len(syms) == 2  # Original and horizontal flip

    orig_b, orig_pi = syms[0]
    flip_b, flip_pi = syms[1]

    # Verify original
    np.testing.assert_array_equal(orig_b, b)
    np.testing.assert_array_equal(orig_pi, pi)

    # Verify flipped board
    np.testing.assert_array_equal(flip_b, np.fliplr(b))
    assert flip_b[5, 6] == 1
    assert flip_b[5, 0] == 0

    # Verify flipped policy mirrors columns 0..6 while leaving pass action (index 7) untouched
    assert flip_pi[6] == 0.8
    assert flip_pi[0] == 0.2
    assert flip_pi[7] == 0.0

    # Policy distribution sums preserved
    assert pytest.approx(np.sum(orig_pi)) == 1.0
    assert pytest.approx(np.sum(flip_pi)) == 1.0
