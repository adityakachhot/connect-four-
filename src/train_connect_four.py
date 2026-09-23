"""
Training entry-point for Connect Four with alpha-zero-general Coach.

Plugs in:
  - src/games/connect_four.py  →  ConnectFourGame   (Game subclass)
  - src/nets/connect_four_nnet.py → ConnectFourNNet (NeuralNet subclass)

Smoke-test configuration (fast finish on M3 Air):
  numIters=3, numEps=10, numMCTSSims=15, arenaCompare=10

Reports per-iteration wall time at the end so you can project how long
a full training run would take.

Usage
-----
    # Fresh run (random init):
    .venv/bin/python src/train_connect_four.py [--checkpoint ./checkpoints/c4]

    # Resume from an existing checkpoint:
    .venv/bin/python src/train_connect_four.py \
        --load-checkpoint checkpoints/connect_four_run2/best.pth.tar \
        --checkpoint      checkpoints/connect_four_run3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from random import shuffle
from typing import List

import coloredlogs
import numpy as np

# ── resolve both src/ and reference-repo/ on the path ───────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── reference-repo imports (read-only) ──────────────────────────────────────
from Coach import Coach
from Arena import Arena
from MCTS import MCTS
from utils import dotdict

# ── my own classes ──────────────────────────────────────────────────────────
from src.games.connect_four import ConnectFourGame
from src.nets.connect_four_nnet import ConnectFourNNet, NNetArgs

# ── logging ─────────────────────────────────────────────────────────────────
coloredlogs.install(level="INFO")
log = logging.getLogger(__name__)


# ============================================================
# TimedCoach — adds per-iteration wall-clock reporting
# ============================================================

class TimedCoach(Coach):
    """
    Thin subclass of Coach that wraps each training iteration with a
    wall-clock timer and writes a summary at the end.

    No logic is changed; only timing instrumentation is added.
    """

    def learn(self) -> None:
        """Run the full AlphaZero training loop with per-iteration timing."""
        iter_times: List[float] = []

        for i in range(1, self.args.numIters + 1):
            iter_start = time.perf_counter()
            log.info(f"─── Starting Iter #{i}/{self.args.numIters} ───")

            # ── Phase 1: self-play data collection ──────────────────────────
            phase_start = time.perf_counter()
            if not self.skipFirstSelfPlay or i > 1:
                from tqdm import tqdm
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc=f"[Iter {i}] Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iterationTrainExamples += self.executeEpisode()
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Trimming oldest trainExamples from history.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)
            t_selfplay = time.perf_counter() - phase_start
            log.info(f"  Self-play      : {t_selfplay:.1f}s "
                     f"({len(iterationTrainExamples)} examples collected)")

            # ── Phase 2: network training ────────────────────────────────────
            phase_start = time.perf_counter()
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            pmcts = MCTS(self.game, self.pnet, self.args)

            log.info(f"  Training on {len(trainExamples)} examples …")
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)
            t_train = time.perf_counter() - phase_start
            log.info(f"  NN training    : {t_train:.1f}s")

            # ── Phase 3: arena evaluation ────────────────────────────────────
            phase_start = time.perf_counter()
            log.info("  Pitting new model against previous …")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            t_arena = time.perf_counter() - phase_start

            log.info(
                f"  Arena result   : new={nwins} prev={pwins} draws={draws}  "
                f"({t_arena:.1f}s)"
            )

            # ── Accept / reject new model ────────────────────────────────────
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info("  Decision       : REJECTING new model (reverting weights)")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            else:
                ckpt_name = self.getCheckpointFile(i)
                log.info(f"  Decision       : ACCEPTING new model → {ckpt_name}")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=ckpt_name)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")

            # ── Always save a per-iteration snapshot when requested ──────────
            if self.args.get("saveEveryIter", False):
                snap_name = f"iter_{i:03d}.pth.tar"
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=snap_name)
                log.info(f"  Snapshot       : {snap_name} (saveEveryIter=True)")

            # ── Per-iteration summary ────────────────────────────────────────
            iter_elapsed = time.perf_counter() - iter_start
            iter_times.append(iter_elapsed)
            log.info(
                f"  ── Iter #{i} done in {iter_elapsed:.1f}s  "
                f"(self-play={t_selfplay:.1f}s | train={t_train:.1f}s | arena={t_arena:.1f}s)"
            )

        # ── Final timing report ──────────────────────────────────────────────
        _print_timing_report(iter_times, self.args)


def _print_timing_report(iter_times: List[float], args: dotdict) -> None:
    """Print a concise timing table and projection to stdout."""
    total   = sum(iter_times)
    avg     = total / len(iter_times)
    fastest = min(iter_times)
    slowest = max(iter_times)

    print("\n" + "═" * 60)
    print("  TRAINING COMPLETE — TIMING REPORT")
    print("═" * 60)
    print(f"  {'Iteration':<12} {'Wall time':>12}")
    print(f"  {'─'*12} {'─'*12}")
    for idx, t in enumerate(iter_times, start=1):
        print(f"  {idx:<12} {t:>11.1f}s")
    print(f"  {'─'*12} {'─'*12}")
    print(f"  {'Average':<12} {avg:>11.1f}s")
    print(f"  {'Fastest':<12} {fastest:>11.1f}s")
    print(f"  {'Slowest':<12} {slowest:>11.1f}s")
    print(f"  {'TOTAL':<12} {total:>11.1f}s")
    print("═" * 60)
    print("\n  PROJECTION (based on average iteration time)")
    for target_iters in [10, 25, 50, 100, 200]:
        projected_s = avg * target_iters
        h, rem = divmod(projected_s, 3600)
        m, s   = divmod(rem, 60)
        print(f"    {target_iters:>4} iters → {int(h):02d}h {int(m):02d}m {int(s):02d}s")
    print()
    print(f"  Smoke-test config: numEps={args.numEps}  "
          f"numMCTSSims={args.numMCTSSims}  "
          f"arenaCompare={args.arenaCompare}")
    print("  ↳ Scale numEps and numMCTSSims up for real training.")
    print("═" * 60 + "\n")


# ============================================================
# Checkpoint verification helper
# ============================================================

def _verify_checkpoints(checkpoint_dir: str, num_iters: int) -> None:
    """Check that expected checkpoint files exist after training."""
    expected_always = ["temp.pth.tar"]  # always written
    found, missing = [], []

    for fname in expected_always:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            found.append((fname, size_kb))
        else:
            missing.append(fname)

    # Optional iteration checkpoints (only written when new model is accepted)
    optional = ["best.pth.tar"] + [
        f"checkpoint_{i}.pth.tar" for i in range(num_iters)
    ]
    for fname in optional:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            found.append((fname, size_kb))

    print("  Checkpoint files in", checkpoint_dir)
    for fname, size_kb in sorted(found):
        print(f"    ✓  {fname:<35} ({size_kb:.1f} KB)")
    for fname in missing:
        print(f"    ✗  {fname}  ← MISSING")
    if missing:
        raise RuntimeError(f"Expected checkpoint files not found: {missing}")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Connect Four AlphaZero smoke-test training run."
    )
    p.add_argument(
        "--checkpoint",
        default=os.path.join(_ROOT, "checkpoints", "connect_four"),
        help="Directory to store model checkpoints (default: checkpoints/connect_four/)",
    )
    p.add_argument("--num-iters",      type=int,            default=3,    help="Training iterations")
    p.add_argument("--num-eps",        type=int,            default=10,   help="Self-play episodes per iter")
    p.add_argument("--num-mcts",       type=int,            default=15,   help="MCTS simulations per move")
    p.add_argument("--arena-cmp",      type=int,            default=10,   help="Arena games per evaluation")
    p.add_argument("--save-every-iter",action="store_true", default=True, help="Write iter_NNN.pth.tar after every iteration (default: on)")
    p.add_argument(
        "--load-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a .pth.tar checkpoint whose model weights will be "
            "loaded as the starting point before self-play begins.  Only the "
            "'state_dict' key is used; training-example history is NOT loaded.  "
            "Has no effect on where new checkpoints are saved (use --checkpoint "
            "for that).  Omit to start from a freshly initialised network."
        ),
    )
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    os.makedirs(cli.checkpoint, exist_ok=True)

    # ── Game ─────────────────────────────────────────────────────────────────
    game = ConnectFourGame()
    log.info("Game        : %s  board=%dx%d  actions=%d",
             type(game).__name__, *game.getBoardSize(), game.getActionSize())

    # ── Neural net ───────────────────────────────────────────────────────────
    net_args = NNetArgs(
        num_channels=128,
        dropout=0.3,
        epochs=5,         # keep training epochs modest for the smoke test
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
    )
    nnet = ConnectFourNNet(game, args=net_args)
    log.info("NeuralNet   : %s", nnet)

    # ── Optional: load starting weights from a previous checkpoint ───────────
    if cli.load_checkpoint:
        import torch
        ckpt_path = os.path.abspath(cli.load_checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"--load-checkpoint: file not found: {ckpt_path!r}"
            )
        payload = torch.load(ckpt_path, map_location=nnet.device)
        nnet.model.load_state_dict(payload["state_dict"])
        log.info("Loaded weights  : %s  (device=%s)", ckpt_path, nnet.device)

    # ── Coach args ───────────────────────────────────────────────────────────
    args = dotdict({
        "numIters":                     cli.num_iters,
        "numEps":                       cli.num_eps,
        "numMCTSSims":                  cli.num_mcts,
        "arenaCompare":                 cli.arena_cmp,
        "tempThreshold":                10,    # use temp=1 for first 10 moves, then temp=0
        "updateThreshold":              0.55,  # accept new model if it wins ≥ 55 % of arena games
        "maxlenOfQueue":                5_000, # max training examples kept in history
        "cpuct":                        1.0,   # exploration constant
        "checkpoint":                   cli.checkpoint,
        "load_model":                   False,
        "load_folder_file":             (cli.checkpoint, "best.pth.tar"),
        "numItersForTrainExamplesHistory": 5,
        "saveEveryIter":                cli.save_every_iter,
    })

    log.info("Coach args  : numIters=%d  numEps=%d  numMCTSSims=%d  arenaCompare=%d",
             args.numIters, args.numEps, args.numMCTSSims, args.arenaCompare)
    log.info("Checkpoint  : %s", cli.checkpoint)
    log.info("Load weights: %s", cli.load_checkpoint or "<fresh init>")
    log.info("Starting the learning process 🎲")

    # ── Run ──────────────────────────────────────────────────────────────────
    coach = TimedCoach(game, nnet, args)
    run_start = time.perf_counter()
    coach.learn()
    run_total = time.perf_counter() - run_start

    # ── Post-run checkpoint verification ─────────────────────────────────────
    print("\n  Verifying checkpoint files …")
    _verify_checkpoints(cli.checkpoint, cli.num_iters)

    log.info("Total wall time : %.1f s", run_total)


if __name__ == "__main__":
    main()
