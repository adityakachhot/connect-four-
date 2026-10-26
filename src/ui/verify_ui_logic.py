"""
verify_ui_logic.py
===================
Automated verification script that tests each of the 3 UI modes to full game completion:
  1. Play vs Agent (Human moves vs MCTS agent and Minimax agent)
  2. Watch Agent vs Agent (Full AI match between Run 1 and Run 3 checkpoints)
  3. Replay Transcript (Full stepping through all transcript plies)
"""

import os
import sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.games.connect_four import ConnectFourGame
from src.nets.connect_four_nnet import ConnectFourNNet
from src.eval.minimax_player import MinimaxAgent
from MCTS import MCTS
from utils import dotdict
from src.ui.app import (
    CHECKPOINT_CATALOG,
    find_dropped_row,
    render_board_html,
    parse_transcript_file,
)


def test_mode_1_play_vs_mcts_agent():
    print("\n--- Testing Mode 1: Play vs MCTS Agent (to completion) ---")
    game = ConnectFourGame()
    ckpt_path = CHECKPOINT_CATALOG["Run 3 Best (Warm-started, 50 iters)"]
    folder, filename = os.path.split(ckpt_path)
    nnet = ConnectFourNNet(game)
    nnet.load_checkpoint(folder, filename)
    nnet.model.eval()

    board = game.getInitBoard()
    cur_p = 1  # Human is 1, Agent is -1
    move_count = 0

    while True:
        move_count += 1
        if cur_p == 1:
            # Simulated human move: pick middle or first valid column
            valids = game.getValidMoves(board, 1)
            valid_cols = [c for c in range(7) if valids[c] == 1]
            chosen_col = 3 if 3 in valid_cols else valid_cols[0]
            board, cur_p = game.getNextState(board, 1, chosen_col)
            # print(f"Move {move_count}: Human played Col {chosen_col+1}")
        else:
            # Agent move with MCTS
            canonical = game.getCanonicalForm(board, -1)
            mcts = MCTS(game, nnet, dotdict({"numMCTSSims": 25, "cpuct": 1.0}))
            probs = mcts.getActionProb(canonical, temp=0)
            action = int(np.argmax(probs))
            pi, v = nnet.predict(canonical)
            assert pi.shape == (8,)
            assert -1.0 <= v <= 1.0
            board, cur_p = game.getNextState(board, -1, action)
            # print(f"Move {move_count}: Agent played Col {action+1} (Value: {v:+.2f})")

        res = game.getGameEnded(board, 1)
        if res != 0:
            outcome = "Human won" if res == 1 else ("Agent won" if res == -1 else "Draw")
            print(f"Game finished in {move_count} moves. Outcome: {outcome}")
            html = render_board_html(board)
            assert len(html) > 100
            break
    print("Mode 1 (MCTS) test passed!")


def test_mode_1_play_vs_minimax_agent():
    print("\n--- Testing Mode 1: Play vs Minimax Agent (to completion) ---")
    game = ConnectFourGame()
    board = game.getInitBoard()
    cur_p = 1
    move_count = 0
    mm_agent = MinimaxAgent(game, depth=2)

    while True:
        move_count += 1
        if cur_p == 1:
            valids = game.getValidMoves(board, 1)
            valid_cols = [c for c in range(7) if valids[c] == 1]
            chosen_col = valid_cols[0]
            board, cur_p = game.getNextState(board, 1, chosen_col)
        else:
            canonical = game.getCanonicalForm(board, -1)
            valids = game.getValidMoves(canonical, 1)
            action = mm_agent.select_action(canonical, valids)
            board, cur_p = game.getNextState(board, -1, action)

        res = game.getGameEnded(board, 1)
        if res != 0:
            outcome = "Human won" if res == 1 else ("Minimax won" if res == -1 else "Draw")
            print(f"Game finished in {move_count} moves. Outcome: {outcome}")
            break
    print("Mode 1 (Minimax) test passed!")


def test_mode_2_watch_agent_vs_agent():
    print("\n--- Testing Mode 2: Watch Agent vs Agent (to completion) ---")
    game = ConnectFourGame()
    p1_path = CHECKPOINT_CATALOG["Run 1 Early (Iter 5)"]
    p2_path = CHECKPOINT_CATALOG["Run 3 Best (Warm-started, 50 iters)"]

    nnet1 = ConnectFourNNet(game)
    f1, fn1 = os.path.split(p1_path)
    nnet1.load_checkpoint(f1, fn1)
    nnet1.model.eval()

    nnet2 = ConnectFourNNet(game)
    f2, fn2 = os.path.split(p2_path)
    nnet2.load_checkpoint(f2, fn2)
    nnet2.model.eval()

    board = game.getInitBoard()
    cur_p = 1
    move_count = 0
    moves = []

    while True:
        move_count += 1
        canonical = game.getCanonicalForm(board, cur_p)
        active_nnet = nnet1 if cur_p == 1 else nnet2
        mcts = MCTS(game, active_nnet, dotdict({"numMCTSSims": 10, "cpuct": 1.0}))
        probs = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(probs))

        r = find_dropped_row(board, action)
        board, cur_p = game.getNextState(board, cur_p, action)
        moves.append((r, action))

        res = game.getGameEnded(board, 1)
        if res != 0:
            outcome = "P1 won" if res == 1 else ("P2 won" if res == -1 else "Draw")
            print(f"Watch match finished in {move_count} moves. Outcome: {outcome}")
            break
    assert len(moves) == move_count
    print("Mode 2 (Watch) test passed!")


def test_mode_3_replay_transcript():
    print("\n--- Testing Mode 3: Replay Transcript (all files to completion) ---")
    report_dir = os.path.join(_ROOT, "src", "results", "report")
    for fname in ["transcript_early.txt", "transcript_late.txt"]:
        fpath = os.path.join(report_dir, fname)
        title, outcome, steps = parse_transcript_file(fpath)
        print(f"File {fname}: Title='{title}', Outcome='{outcome}', Steps={len(steps)}")
        assert len(steps) > 5
        # Step through every single ply
        for step_idx, step in enumerate(steps):
            b = step["board"]
            assert b.shape == (6, 7)
            html = render_board_html(b)
            assert len(html) > 50
        print(f"Stepped through all {len(steps)} steps of {fname} successfully.")
    print("Mode 3 (Replay) test passed!")


if __name__ == "__main__":
    test_mode_1_play_vs_mcts_agent()
    test_mode_1_play_vs_minimax_agent()
    test_mode_2_watch_agent_vs_agent()
    test_mode_3_replay_transcript()
    print("\n=============================================")
    print("ALL 3 UI MODES PASSED VERIFICATION WITH ZERO ERRORS!")
    print("=============================================")
