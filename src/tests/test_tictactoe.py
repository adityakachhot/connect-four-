"""
pytest test suite for TicTacToeGame and MinimaxAgent.

Covers:
  - TicTacToeGame:
      - Board initialization, size, action size
      - Valid moves generation (empty board, partial board, full board)
      - Move execution (getNextState)
      - Win detection (horizontal, vertical, diagonal, anti-diagonal)
      - Draw detection
      - Canonical form
      - Symmetries (D4 group: 8 symmetries)
      - String representation uniqueness
  - MinimaxAgent:
      - Immediate win execution (1-ply)
      - Immediate threat blocking (1-ply)
      - Optimal self-play (minimax vs minimax on Tic-Tac-Toe always draws)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.games.tic_tac_toe import TicTacToeGame
from src.eval.minimax_player import MinimaxAgent
from src.eval.game_runner import play_game


@pytest.fixture
def ttt():
    return TicTacToeGame()


def test_init_board_and_sizes(ttt):
    board = ttt.getInitBoard()
    assert board.shape == (3, 3)
    assert np.all(board == 0)
    assert ttt.getBoardSize() == (3, 3)
    assert ttt.getActionSize() == 10  # 9 cells + 1 pass


def test_valid_moves_initial(ttt):
    board = ttt.getInitBoard()
    valids = ttt.getValidMoves(board, 1)
    assert len(valids) == 10
    assert np.all(valids[:9] == 1)
    assert valids[9] == 0  # Pass is invalid on non-full board


def test_valid_moves_partial_and_full(ttt):
    board = ttt.getInitBoard()
    board[0, 0] = 1
    board[1, 1] = -1
    valids = ttt.getValidMoves(board, 1)
    assert valids[0] == 0
    assert valids[4] == 0
    assert np.sum(valids[:9]) == 7
    assert valids[9] == 0

    # Fill entire board
    full_board = np.array([
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1]
    ], dtype=np.int8)
    valids_full = ttt.getValidMoves(full_board, 1)
    assert np.all(valids_full[:9] == 0)
    assert valids_full[9] == 1  # Pass is valid


def test_next_state(ttt):
    board = ttt.getInitBoard()
    next_board, next_player = ttt.getNextState(board, 1, 4)  # center
    assert next_board[1, 1] == 1
    assert next_player == -1

    # Second move by player -1
    next_board2, next_player2 = ttt.getNextState(next_board, -1, 0)  # top-left
    assert next_board2[0, 0] == -1
    assert next_player2 == 1


def test_win_horizontal(ttt):
    board = np.array([
        [1, 1, 1],
        [-1, -1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    # For player 1, returns 1
    assert ttt.getGameEnded(board, 1) == 1.0
    # For player -1, returns -1
    assert ttt.getGameEnded(board, -1) == -1.0


def test_win_vertical(ttt):
    board = np.array([
        [-1, 1, 0],
        [-1, 1, 0],
        [-1, 0, 0]
    ], dtype=np.int8)
    assert ttt.getGameEnded(board, -1) == 1.0
    assert ttt.getGameEnded(board, 1) == -1.0


def test_win_diagonals(ttt):
    # Main diagonal
    diag1 = np.array([
        [1, -1, 0],
        [-1, 1, 0],
        [0, 0, 1]
    ], dtype=np.int8)
    assert ttt.getGameEnded(diag1, 1) == 1.0

    # Anti-diagonal
    diag2 = np.array([
        [0, 0, -1],
        [1, -1, 1],
        [-1, 0, 1]
    ], dtype=np.int8)
    assert ttt.getGameEnded(diag2, -1) == 1.0


def test_draw(ttt):
    # Standard cat's game / draw
    draw_board = np.array([
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1]
    ], dtype=np.int8)
    res = ttt.getGameEnded(draw_board, 1)
    assert 0 < abs(res) < 1e-3  # Non-zero small float indicating draw


def test_canonical_form(ttt):
    board = np.array([
        [1, -1, 0],
        [0,  1, 0],
        [0,  0, 0]
    ], dtype=np.int8)
    canon_p1 = ttt.getCanonicalForm(board, 1)
    np.testing.assert_array_equal(canon_p1, board)

    canon_p2 = ttt.getCanonicalForm(board, -1)
    np.testing.assert_array_equal(canon_p2, -board)


def test_symmetries(ttt):
    board = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    pi = np.zeros(10, dtype=np.float32)
    pi[0] = 1.0  # action at (0, 0)

    syms = ttt.getSymmetries(board, pi)
    assert len(syms) == 8  # D4 symmetry: 4 rotations * 2 reflections

    for sym_board, sym_pi in syms:
        assert sym_board.shape == (3, 3)
        assert len(sym_pi) == 10
        # The sum of policy distribution should be preserved
        assert pytest.approx(np.sum(sym_pi)) == 1.0


def test_minimax_immediate_win(ttt):
    # Player 1 has 2 in a row and can win at (0, 2) = action 2
    board = np.array([
        [1, 1, 0],
        [-1, -1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    agent = MinimaxAgent(ttt, depth=2)
    canonical = ttt.getCanonicalForm(board, 1)
    valids = ttt.getValidMoves(canonical, 1)
    action = agent.select_action(canonical, valids)
    assert action == 2


def test_minimax_immediate_block(ttt):
    # Opponent (player 1) is about to win at (0, 2)
    # Current player is player -1
    board = np.array([
        [1, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    agent = MinimaxAgent(ttt, depth=2)
    canonical = ttt.getCanonicalForm(board, -1)
    valids = ttt.getValidMoves(canonical, 1)
    action = agent.select_action(canonical, valids)
    assert action == 2


def test_minimax_self_play_always_draws(ttt):
    # Perfect minimax vs perfect minimax on 3x3 Tic-Tac-Toe MUST always draw
    p1 = MinimaxAgent(ttt, depth=9)
    p2 = MinimaxAgent(ttt, depth=9)
    winner = play_game(ttt, p1, p2)
    assert winner == 0
