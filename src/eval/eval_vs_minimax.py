"""
eval_vs_minimax.py — Evaluate best checkpoints from run1, run2, run3
                     against minimax at depths 2, 4, and 6.

Usage
-----
    python src/eval/eval_vs_minimax.py [--n-games 40] [--output-dir src/results]

For each of the 3 runs × 3 depths = 9 matchups, plays n_games with first-mover
alternating every game.  Saves:
  - src/results/minimax_results.md    (plain-text table)
  - src/results/minimax_bar_chart.png (grouped bar chart)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

from src.games.connect_four import ConnectFourGame
from src.nets.connect_four_nnet import ConnectFourNNet, NNetArgs
from src.eval.game_runner import NNetGreedyAgent, play_match
from src.eval.minimax_player import MinimaxAgent


# ===========================================================================
# Configuration
# ===========================================================================

# Map run label → path to best.pth.tar  (relative to project root)
BEST_CHECKPOINTS = {
    "Run 1": os.path.join("checkpoints", "connect_four_50",  "best.pth.tar"),
    "Run 2": os.path.join("checkpoints", "connect_four_run2", "best.pth.tar"),
    "Run 3": os.path.join("checkpoints", "connect_four_run3", "best.pth.tar"),
}

MINIMAX_DEPTHS = [2, 4, 6]


# ===========================================================================
# Helpers
# ===========================================================================

def load_nnet_agent(game: ConnectFourGame, ckpt_path: str) -> NNetGreedyAgent:
    """Load a checkpoint into an NNetGreedyAgent."""
    import torch
    nnet = ConnectFourNNet(game)
    payload = torch.load(ckpt_path, map_location=nnet.device)
    nnet.model.load_state_dict(payload["state_dict"])
    nnet.model.eval()
    return NNetGreedyAgent(nnet)


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def run_evaluation(
    n_games: int = 40,
) -> List[dict]:
    """
    Returns a list of result dicts, one per (run, depth) matchup.
    Each dict has: run, depth, wins_nnet, wins_minimax, draws,
                   win_rate_nnet, n_games, elapsed_s
    """
    game = ConnectFourGame()
    results = []

    total_matchups = len(BEST_CHECKPOINTS) * len(MINIMAX_DEPTHS)
    idx = 0

    for run_label, rel_ckpt in BEST_CHECKPOINTS.items():
        ckpt_path = os.path.join(_ROOT, rel_ckpt)
        if not os.path.isfile(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found — skipping {run_label}")
            continue

        nnet_agent = load_nnet_agent(game, ckpt_path)
        print(f"\n  Loaded {run_label} checkpoint: {rel_ckpt}")

        for depth in MINIMAX_DEPTHS:
            idx += 1
            mm_agent = MinimaxAgent(game, depth=depth)

            print(f"  [{idx}/{total_matchups}] {run_label} vs Minimax(d={depth})"
                  f"  — {n_games} games …", end="", flush=True)

            t0 = time.time()
            match = play_match(game, nnet_agent, mm_agent, n_games=n_games)
            elapsed = time.time() - t0

            row = {
                "run":            run_label,
                "depth":          depth,
                "wins_nnet":      match["wins_a"],
                "wins_minimax":   match["wins_b"],
                "draws":          match["draws"],
                "win_rate_nnet":  match["win_rate_a"],
                "n_games":        n_games,
                "elapsed_s":      round(elapsed, 1),
            }
            results.append(row)

            print(
                f"  W={row['wins_nnet']:>2} L={row['wins_minimax']:>2} "
                f"D={row['draws']:>2}  "
                f"win%={row['win_rate_nnet']:.1%}  "
                f"({elapsed:.1f}s)"
            )

    return results


# ===========================================================================
# Results table (markdown)
# ===========================================================================

def write_results_table(results: List[dict], output_path: str) -> None:
    """Write a markdown-formatted results table."""
    lines = [
        "# Win Rate vs. Minimax Baseline",
        "",
        f"**Games per matchup:** {results[0]['n_games']}  ",
        "**Evaluation:** NNet greedy (single forward pass, no MCTS) vs Minimax with alpha-beta pruning  ",
        "**First mover:** alternated evenly (half the games each side)",
        "",
        "| Run | Minimax Depth | NNet Wins | Minimax Wins | Draws | Win Rate (decisive) | Time |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['run']} | {r['depth']} | {r['wins_nnet']} | "
            f"{r['wins_minimax']} | {r['draws']} | "
            f"**{r['win_rate_nnet']:.1%}** | {r['elapsed_s']:.1f}s |"
        )

    # Summary
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")

    # Group by run
    runs = sorted(set(r["run"] for r in results), key=lambda x: x)
    for run in runs:
        run_results = [r for r in results if r["run"] == run]
        avg_wr = np.mean([r["win_rate_nnet"] for r in run_results])
        lines.append(f"- **{run}**: avg win rate across depths = {avg_wr:.1%}")

    # Group by depth
    lines.append("")
    for d in MINIMAX_DEPTHS:
        depth_results = [r for r in results if r["depth"] == d]
        avg_wr = np.mean([r["win_rate_nnet"] for r in depth_results])
        lines.append(f"- **Depth {d}**: avg NNet win rate = {avg_wr:.1%}")

    text = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\n  Results table saved → {output_path}")


# ===========================================================================
# Bar chart
# ===========================================================================

PALETTE = {
    "bg":       "#0F1923",
    "panel_bg": "#162030",
    "text":     "#ECF0F1",
    "grid":     "#2C3E50",
    "run1":     "#4FC3F7",   # sky blue
    "run2":     "#FFB74D",   # amber
    "run3":     "#66BB6A",   # green
    "accent":   "#EF5350",   # red
}

RUN_COLOURS = {
    "Run 1": PALETTE["run1"],
    "Run 2": PALETTE["run2"],
    "Run 3": PALETTE["run3"],
}


def plot_bar_chart(results: List[dict], output_path: str) -> None:
    """
    Grouped bar chart: x-axis = minimax depth, grouped by run,
    y-axis = win rate (decisive games).
    """
    runs   = list(BEST_CHECKPOINTS.keys())
    depths = MINIMAX_DEPTHS
    n_runs = len(runs)

    # Build lookup: (run, depth) → win_rate
    lookup = {}
    for r in results:
        lookup[(r["run"], r["depth"])] = r["win_rate_nnet"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel_bg"])

    bar_width = 0.22
    x_base = np.arange(len(depths))

    for i, run in enumerate(runs):
        offsets = x_base + (i - n_runs / 2 + 0.5) * bar_width
        heights = [lookup.get((run, d), 0) for d in depths]
        colour = RUN_COLOURS.get(run, "#888888")

        bars = ax.bar(
            offsets, heights, bar_width,
            label=run, color=colour, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        # Value labels on top of bars
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.0%}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=9, fontweight="bold",
            )

    # 50% reference line
    ax.axhline(0.5, color=PALETTE["text"], linewidth=0.8,
               linestyle=":", alpha=0.5, zorder=2)
    ax.text(x_base[-1] + 0.55, 0.51, "50%",
            color=PALETTE["text"], fontsize=8, alpha=0.6)

    # Styling
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"Depth {d}" for d in depths], fontsize=11)
    ax.set_ylabel("NNet Win Rate (decisive games)", fontsize=11,
                  color=PALETTE["text"])
    ax.set_xlabel("Minimax Search Depth", fontsize=11,
                  color=PALETTE["text"])
    ax.set_title(
        "AlphaZero NNet (greedy) vs Minimax α-β  —  Win Rate by Depth",
        fontsize=13, fontweight="bold", color=PALETTE["text"], pad=12,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.12)

    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=0.4)

    ax.legend(
        loc="upper right", fontsize=10,
        facecolor=PALETTE["panel_bg"], labelcolor=PALETTE["text"],
        framealpha=0.8, edgecolor=PALETTE["grid"],
    )

    # Annotation: subtitle
    fig.text(
        0.5, 0.01,
        f"{results[0]['n_games']} games per matchup · first-mover alternated · NNet = single forward-pass greedy (no MCTS)",
        ha="center", fontsize=9, color=PALETTE["text"], alpha=0.7,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Bar chart saved → {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate best checkpoints vs minimax at various depths."
    )
    p.add_argument("--n-games", type=int, default=40,
                   help="Games per matchup (default 40, alternating first mover)")
    p.add_argument("--output-dir", default="src/results",
                   help="Directory to save results (default src/results)")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    output_dir = os.path.join(_ROOT, cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 64)
    print("  Evaluation: AlphaZero best checkpoints vs Minimax (α-β)")
    print("=" * 64)

    results = run_evaluation(n_games=cli.n_games)

    if not results:
        print("  ERROR: No results generated.  Check checkpoint paths.")
        sys.exit(1)

    # Save results
    table_path = os.path.join(output_dir, "minimax_results.md")
    chart_path = os.path.join(output_dir, "minimax_bar_chart.png")

    write_results_table(results, table_path)
    plot_bar_chart(results, chart_path)

    # Also save raw JSON for programmatic access
    json_path = os.path.join(output_dir, "minimax_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Raw JSON saved → {json_path}")

    print("\n" + "=" * 64)
    print("  Done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
