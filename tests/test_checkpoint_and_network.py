"""
tests/test_checkpoint_and_network.py
=====================================
Tests that load the best trained checkpoints and verify:
  1. Neural network output shapes (policy vector of length action_size, value scalar).
  2. Policy masked over valid moves sums to 1.0 with non-negative probabilities.
  3. Value head output bounded in [-1.0, 1.0].
  4. Greedy agent selects a strictly legal action.
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
from src.nets.connect_four_nnet import ConnectFourNNet
from src.games.tic_tac_toe import TicTacToeGame
from src.nets.tictactoe_nnet import TicTacToeNNet
from src.eval.game_runner import NNetGreedyAgent


def test_connect_four_best_checkpoint_outputs_and_policy():
    game = ConnectFourGame()
    ckpt_path = os.path.join(_ROOT, "checkpoints", "connect_four_run3", "best.pth.tar")
    assert os.path.isfile(ckpt_path), f"Checkpoint missing at {ckpt_path}"

    folder, filename = os.path.split(ckpt_path)
    nnet = ConnectFourNNet(game)
    nnet.load_checkpoint(folder, filename)
    nnet.model.eval()

    # 1. Test on empty board
    init_board = game.getInitBoard()
    pi_init, v_init = nnet.predict(init_board)

    # Output shape checks
    assert isinstance(pi_init, np.ndarray)
    assert pi_init.shape == (game.getActionSize(),), f"Expected shape ({game.getActionSize()},), got {pi_init.shape}"
    assert isinstance(v_init, (float, np.floating))
    assert -1.0 <= v_init <= 1.0

    # Policy distribution over valid moves sums to 1
    valids_init = game.getValidMoves(init_board, 1)
    masked_pi = pi_init * valids_init.astype(np.float32)
    norm_pi = masked_pi / np.sum(masked_pi)

    assert np.isclose(np.sum(norm_pi), 1.0)
    assert np.all(norm_pi >= 0.0)
    assert norm_pi[7] == 0.0  # Pass is invalid

    # 2. Test on partially filled board with a full column
    board = game.getInitBoard()
    # Fill column 3 completely
    for r in range(6):
        board[r, 3] = 1 if r % 2 == 0 else -1
    # Place pieces in column 0 and 1
    board[5, 0] = 1
    board[5, 1] = -1

    canonical = game.getCanonicalForm(board, 1)
    valids = game.getValidMoves(canonical, 1)
    assert valids[3] == 0  # Column 3 is full

    pi, v = nnet.predict(canonical)
    assert pi.shape == (8,)
    assert -1.0 <= v <= 1.0

    # Mask and normalize
    masked = pi * valids.astype(np.float32)
    assert np.sum(masked) > 0.0
    normalized_policy = masked / np.sum(masked)

    # Verification: sums to 1 over valid moves
    assert np.isclose(np.sum(normalized_policy), 1.0)
    assert normalized_policy[3] == 0.0  # Full column receives 0 probability
    assert normalized_policy[7] == 0.0  # Pass receives 0 probability
    assert np.all(normalized_policy >= 0.0)

    # Verify NNetGreedyAgent chooses a strictly legal move
    agent = NNetGreedyAgent(nnet)
    chosen_action = agent.select_action(canonical, valids)
    assert valids[chosen_action] == 1
    assert chosen_action != 3
    assert 0 <= chosen_action < 7


def test_tictactoe_best_checkpoint_outputs_and_policy():
    game = TicTacToeGame()
    ckpt_path = os.path.join(_ROOT, "checkpoints", "tictactoe", "best.pth.tar")
    assert os.path.isfile(ckpt_path), f"Checkpoint missing at {ckpt_path}"

    folder, filename = os.path.split(ckpt_path)
    nnet = TicTacToeNNet(game)
    nnet.load_checkpoint(folder, filename)
    nnet.model.eval()

    # Test on partial board
    board = game.getInitBoard()
    board[1, 1] = 1   # Center taken
    board[0, 0] = -1  # Corner taken

    canonical = game.getCanonicalForm(board, 1)
    valids = game.getValidMoves(canonical, 1)
    assert valids[4] == 0  # Center invalid
    assert valids[0] == 0  # Corner invalid

    pi, v = nnet.predict(canonical)
    assert pi.shape == (game.getActionSize(),)
    assert -1.0 <= v <= 1.0

    masked = pi * valids.astype(np.float32)
    normalized_policy = masked / np.sum(masked)

    # Verification: policy sums to 1 over valid moves
    assert np.isclose(np.sum(normalized_policy), 1.0)
    assert normalized_policy[0] == 0.0
    assert normalized_policy[4] == 0.0
    assert normalized_policy[9] == 0.0  # Pass is invalid
    assert np.all(normalized_policy >= 0.0)
