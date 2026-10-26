"""
eval_tictactoe.py — Evaluate Tic-Tac-Toe checkpoints against random and minimax.

Usage
-----
    python src/eval/eval_tictactoe.py [--checkpoint-dir checkpoints/tictactoe]

Steps:
1. Load each iter_NNN.pth.tar checkpoint.
2. Play 50 games vs RandomAgent and 50 vs MinimaxAgent(depth=9) per checkpoint.
3. Save results table to src/results/tictactoe_results.md
4. Save win-rate-over-iterations plot to src/results/tictactoe_winrate.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- resolve project root ---------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.games.tic_tac_toe import TicTacToeGame
from src.nets.tictactoe_nnet import TicTacToeNNet, TTTNNetArgs
from src.eval.game_runner import RandomAgent, NNetGreedyAgent, play_match
from src.eval.minimax_player import MinimaxAgent


# ===========================================================================
# Checkpoint discovery
# ===========================================================================

def find_iter_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    pattern = re.compile(r"^iter_(\d+)\.pth\.tar$")
    results = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            results.append((int(m.group(1)), os.path.join(checkpoint_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_checkpoints(
    checkpoints: List[Tuple[int, str]],
    game: TicTacToeGame,
    n_games: int = 50,
) -> List[dict]:
    """
    For each checkpoint, play n_games vs Random and n_games vs Minimax(depth=9).
    Depth 9 = perfect play for Tic-Tac-Toe (at most 9 moves in a game).
    """
    results = []
    random_agent  = RandomAgent()
    minimax_agent = MinimaxAgent(game, depth=9)  # perfect play

    nnet = TicTacToeNNet(game)
    total = len(checkpoints)

    for idx, (iteration, ckpt_path) in enumerate(checkpoints, start=1):
        import torch
        payload = torch.load(ckpt_path, map_location=nnet.device)
        nnet.model.load_state_dict(payload["state_dict"])
        nnet.model.eval()
        agent = NNetGreedyAgent(nnet)

        # vs Random
        t0 = time.time()
        vs_rand = play_match(game, agent, random_agent, n_games=n_games)
        t_rand = time.time() - t0

        # vs Minimax (perfect)
        t0 = time.time()
        vs_mm = play_match(game, agent, minimax_agent, n_games=n_games)
        t_mm = time.time() - t0

        row = {
            "iteration":      iteration,
            "vs_rand_wins":   vs_rand["wins_a"],
            "vs_rand_losses": vs_rand["wins_b"],
            "vs_rand_draws":  vs_rand["draws"],
            "vs_rand_wr":     vs_rand["win_rate_a"],
            "vs_mm_wins":     vs_mm["wins_a"],
            "vs_mm_losses":   vs_mm["wins_b"],
            "vs_mm_draws":    vs_mm["draws"],
            "vs_mm_wr":       vs_mm["win_rate_a"],
        }
        results.append(row)

        bar_filled = int(30 * idx / total)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)
        print(
            f"  [{bar}] iter {iteration:>2}/{checkpoints[-1][0]}  "
            f"vsRand: W={row['vs_rand_wins']:>2} L={row['vs_rand_losses']:>2} D={row['vs_rand_draws']:>2} ({row['vs_rand_wr']:.0%})  "
            f"vsMM:   W={row['vs_mm_wins']:>2} L={row['vs_mm_losses']:>2} D={row['vs_mm_draws']:>2} ({row['vs_mm_wr']:.0%})  "
            f"[{t_rand + t_mm:.1f}s]",
            flush=True,
        )

    return results


# ===========================================================================
# Results table
# ===========================================================================

def write_results_table(results: List[dict], output_path: str) -> None:
    lines = [
        "# Tic-Tac-Toe — Pipeline Validation Results",
        "",
        "**Training:** numIters=10, numEps=25, numMCTSSims=25  ",
        "**Evaluation:** 50 games each (alternating first mover)  ",
        "**Minimax:** depth=9 (perfect play for Tic-Tac-Toe)",
        "",
        "| Iter | vs Random W-L-D | Win Rate | vs Minimax(d=9) W-L-D | Win Rate |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['iteration']} | "
            f"{r['vs_rand_wins']}-{r['vs_rand_losses']}-{r['vs_rand_draws']} | "
            f"**{r['vs_rand_wr']:.1%}** | "
            f"{r['vs_mm_wins']}-{r['vs_mm_losses']}-{r['vs_mm_draws']} | "
            f"**{r['vs_mm_wr']:.1%}** |"
        )

    # Check for minimax losses
    final = results[-1]
    lines.append("")
    if final["vs_mm_losses"] == 0:
        lines.append("> ✅ **Final checkpoint never lost to minimax (perfect play).** "
                     "This confirms the trained agent plays optimally.")
    else:
        lines.append(f"> ⚠️ **Final checkpoint lost {final['vs_mm_losses']} game(s) to minimax.** "
                     "See investigation notes below.")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\n  Results table saved → {output_path}")


# ===========================================================================
# Plot
# ===========================================================================

PALETTE = {
    "bg":       "#0F1923",
    "panel_bg": "#162030",
    "text":     "#ECF0F1",
    "grid":     "#2C3E50",
    "random":   "#4FC3F7",
    "minimax":  "#FFB74D",
}


def plot_winrate_curve(results: List[dict], output_path: str) -> None:
    iters = [r["iteration"] for r in results]
    wr_rand = [r["vs_rand_wr"] for r in results]
    wr_mm   = [r["vs_mm_wr"]   for r in results]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel_bg"])

    ax.plot(iters, wr_rand, "o-", color=PALETTE["random"],
            linewidth=2, markersize=6, label="vs Random", zorder=3)
    ax.plot(iters, wr_mm, "s-", color=PALETTE["minimax"],
            linewidth=2, markersize=6, label="vs Minimax (d=9, perfect)", zorder=3)

    ax.axhline(0.5, color=PALETTE["text"], linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="#66BB6A", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_title("Tic-Tac-Toe — Win Rate Over Training Iterations",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], pad=10)
    ax.set_xlabel("Training Iteration", fontsize=11, color=PALETTE["text"])
    ax.set_ylabel("Win Rate (decisive games)", fontsize=11, color=PALETTE["text"])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(iters[0] - 0.5, iters[-1] + 0.5)

    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.4)

    ax.legend(loc="lower right", fontsize=10,
              facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"],
              framealpha=0.8, edgecolor=PALETTE["grid"])

    fig.text(0.5, 0.01,
             "50 games per checkpoint · first-mover alternated · NNet = greedy (no MCTS)",
             ha="center", fontsize=9, color=PALETTE["text"], alpha=0.7)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Tic-Tac-Toe checkpoints.")
    p.add_argument("--checkpoint-dir",
                   default=os.path.join("checkpoints", "tictactoe"),
                   help="Directory with iter_NNN.pth.tar files")
    p.add_argument("--n-games", type=int, default=50)
    p.add_argument("--output-dir", default="src/results")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    ckpt_dir   = os.path.join(_ROOT, cli.checkpoint_dir)
    output_dir = os.path.join(_ROOT, cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    game = TicTacToeGame()

    checkpoints = find_iter_checkpoints(ckpt_dir)
    if not checkpoints:
        # Fallback: try best.pth.tar
        best = os.path.join(ckpt_dir, "best.pth.tar")
        if os.path.isfile(best):
            checkpoints = [(0, best)]
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    print(f"\n  Found {len(checkpoints)} checkpoints in {ckpt_dir}")
    print(f"  Evaluating vs Random ({cli.n_games} games) and Minimax d=9 ({cli.n_games} games) …\n")

    results = evaluate_checkpoints(checkpoints, game, n_games=cli.n_games)

    # Write outputs
    table_path = os.path.join(output_dir, "tictactoe_results.md")
    plot_path  = os.path.join(output_dir, "tictactoe_winrate.png")
    json_path  = os.path.join(output_dir, "tictactoe_results.json")

    write_results_table(results, table_path)
    plot_winrate_curve(results, plot_path)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved → {json_path}")

    # Summary
    final = results[-1]
    print(f"\n  Final checkpoint (iter {final['iteration']}):")
    print(f"    vs Random:  {final['vs_rand_wr']:.0%} win rate")
    print(f"    vs Minimax: {final['vs_mm_wr']:.0%} win rate  "
          f"(losses={final['vs_mm_losses']})")

    if final["vs_mm_losses"] > 0:
        print("\n  ⚠️  Agent LOST to perfect minimax — investigation needed.")
    else:
        print("\n  ✅  Agent never lost to perfect minimax!")


if __name__ == "__main__":
    main()
