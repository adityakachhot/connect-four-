"""
tests/test_minimax_agent.py
============================
Tests verifying that MinimaxAgent never makes an illegal move:
  1. On Connect Four across multiple games (alternating roles against RandomAgent).
  2. On Tic-Tac-Toe across multiple games against RandomAgent and in self-play.
"""

import os
import random
import sys
import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.games.connect_four import ConnectFourGame
from src.games.tic_tac_toe import TicTacToeGame
from src.eval.minimax_player import MinimaxAgent
from src.eval.game_runner import RandomAgent


def test_minimax_never_illegal_connect_four():
    random.seed(42)
    np.random.seed(42)
    game = ConnectFourGame()
    minimax_depth2 = MinimaxAgent(game, depth=2)
    minimax_depth4 = MinimaxAgent(game, depth=4)
    rnd_agent = RandomAgent()

    # Play 6 full games: 3 with depth 2, 3 with depth 4, alternating first/second mover
    test_configs = [
        (minimax_depth2, True),   # MM is player 1
        (minimax_depth2, False),  # MM is player -1
        (minimax_depth2, True),
        (minimax_depth4, False),
        (minimax_depth4, True),
        (minimax_depth4, False),
    ]

    total_minimax_moves = 0

    for mm_agent, mm_is_p1 in test_configs:
        board = game.getInitBoard()
        cur_p = 1
        p1_agent = mm_agent if mm_is_p1 else rnd_agent
        p2_agent = rnd_agent if mm_is_p1 else mm_agent

        while True:
            canonical = game.getCanonicalForm(board, cur_p)
            valids = game.getValidMoves(canonical, 1)

            if (cur_p == 1 and mm_is_p1) or (cur_p == -1 and not mm_is_p1):
                # Minimax move
                action = mm_agent.select_action(canonical, valids)
                total_minimax_moves += 1

                # 1. Action must be within regular column range [0, 6]
                assert 0 <= action < 7, f"Minimax picked out-of-bounds column {action}"
                # 2. Action must be marked as legal in valids vector
                assert valids[action] == 1, f"Minimax picked illegal column {action} with valids {valids}"
                # 3. Target column top cell must be unoccupied
                assert board[0, action] == 0, f"Minimax picked full column {action}"
                # 4. Action must NOT be the pass action
                assert action != 7, "Minimax picked pass action during active play"
            else:
                action = p1_agent.select_action(canonical, valids) if cur_p == 1 else p2_agent.select_action(canonical, valids)

            board, cur_p = game.getNextState(board, cur_p, action)
            res = game.getGameEnded(board, 1)
            if res != 0:
                break

    assert total_minimax_moves >= 50, f"Expected >= 50 Minimax moves tested, got {total_minimax_moves}"


def test_minimax_never_illegal_tictactoe():
    random.seed(1337)
    np.random.seed(1337)
    game = TicTacToeGame()
    mm_agent = MinimaxAgent(game, depth=9)
    rnd_agent = RandomAgent()

    total_minimax_moves = 0

    # Test 10 games vs Random (5 as P1, 5 as P2) + 2 self-play games
    matchups = [(True, False)] * 5 + [(False, True)] * 5 + [(True, True)] * 2

    for p1_is_mm, p2_is_mm in matchups:
        board = game.getInitBoard()
        cur_p = 1
        p1 = mm_agent if p1_is_mm else rnd_agent
        p2 = mm_agent if p2_is_mm else rnd_agent

        while True:
            canonical = game.getCanonicalForm(board, cur_p)
            valids = game.getValidMoves(canonical, 1)
            active_agent = p1 if cur_p == 1 else p2
            action = active_agent.select_action(canonical, valids)

            is_mm_turn = (cur_p == 1 and p1_is_mm) or (cur_p == -1 and p2_is_mm)
            if is_mm_turn:
                total_minimax_moves += 1
                # Must be a valid board cell 0..8
                assert 0 <= action < 9, f"Tic-Tac-Toe Minimax chose invalid action {action}"
                assert valids[action] == 1, f"Tic-Tac-Toe Minimax chose illegal move {action}"
                # Target cell on actual board must be empty
                r, c = divmod(action, 3)
                assert board[r, c] == 0, f"Tic-Tac-Toe Minimax played on occupied cell ({r}, {c})"
                # Must never choose pass action (9)
                assert action != 9, "Minimax chose pass action during active play"

            board, cur_p = game.getNextState(board, cur_p, action)
            res = game.getGameEnded(board, 1)
            if res != 0:
                break

    assert total_minimax_moves >= 40, f"Expected >= 40 Minimax moves tested, got {total_minimax_moves}"
