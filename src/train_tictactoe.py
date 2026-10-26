"""
Training entry-point for Tic-Tac-Toe with alpha-zero-general Coach.

Uses the same TimedCoach from train_connect_four.py and plugs in:
  - src/games/tic_tac_toe.py   → TicTacToeGame    (Game subclass)
  - src/nets/tictactoe_nnet.py → TicTacToeNNet     (NeuralNet subclass)

Default configuration (pipeline validation):
  numIters=10, numEps=25, numMCTSSims=25, arenaCompare=10

Usage
-----
    .venv/bin/python src/train_tictactoe.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import coloredlogs

# ── resolve paths ────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── reference-repo imports ──────────────────────────────────────────────────
from utils import dotdict

# ── my classes ──────────────────────────────────────────────────────────────
from src.games.tic_tac_toe import TicTacToeGame
from src.nets.tictactoe_nnet import TicTacToeNNet, TTTNNetArgs
from src.train_connect_four import TimedCoach, _verify_checkpoints

# ── logging ─────────────────────────────────────────────────────────────────
coloredlogs.install(level="INFO")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tic-Tac-Toe AlphaZero training run.")
    p.add_argument("--checkpoint",
                   default=os.path.join(_ROOT, "checkpoints", "tictactoe"),
                   help="Checkpoint directory")
    p.add_argument("--num-iters",  type=int, default=10)
    p.add_argument("--num-eps",    type=int, default=25)
    p.add_argument("--num-mcts",   type=int, default=25)
    p.add_argument("--arena-cmp",  type=int, default=10)
    p.add_argument("--save-every-iter", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    os.makedirs(cli.checkpoint, exist_ok=True)

    # ── Game ─────────────────────────────────────────────────────────────
    game = TicTacToeGame()
    log.info("Game        : %s  board=%dx%d  actions=%d",
             type(game).__name__, *game.getBoardSize(), game.getActionSize())

    # ── Neural net ───────────────────────────────────────────────────────
    net_args = TTTNNetArgs(
        num_channels=64,
        dropout=0.3,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
    )
    nnet = TicTacToeNNet(game, args=net_args)
    log.info("NeuralNet   : %s", nnet)

    # ── Coach args ───────────────────────────────────────────────────────
    args = dotdict({
        "numIters":                     cli.num_iters,
        "numEps":                       cli.num_eps,
        "numMCTSSims":                  cli.num_mcts,
        "arenaCompare":                 cli.arena_cmp,
        "tempThreshold":                5,      # temp=1 for first 5 moves
        "updateThreshold":              0.55,
        "maxlenOfQueue":                5_000,
        "cpuct":                        1.0,
        "checkpoint":                   cli.checkpoint,
        "load_model":                   False,
        "load_folder_file":             (cli.checkpoint, "best.pth.tar"),
        "numItersForTrainExamplesHistory": 5,
        "saveEveryIter":                cli.save_every_iter,
    })

    log.info("Coach args  : numIters=%d  numEps=%d  numMCTSSims=%d  arenaCompare=%d",
             args.numIters, args.numEps, args.numMCTSSims, args.arenaCompare)
    log.info("Checkpoint  : %s", cli.checkpoint)
    log.info("Starting the learning process 🎲")

    # ── Run ──────────────────────────────────────────────────────────────
    coach = TimedCoach(game, nnet, args)
    run_start = time.perf_counter()
    coach.learn()
    run_total = time.perf_counter() - run_start

    # ── Post-run verification ────────────────────────────────────────────
    print("\n  Verifying checkpoint files …")
    _verify_checkpoints(cli.checkpoint, cli.num_iters)

    log.info("Total wall time : %.1f s", run_total)


if __name__ == "__main__":
    main()
