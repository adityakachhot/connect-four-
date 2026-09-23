"""
evaluate.py — Full evaluation pipeline for the 50-iteration Connect Four run.

Steps
-----
1. Load every iter_NNN.pth.tar checkpoint from a checkpoint directory.
2. For each checkpoint, play N_GAMES vs a RandomAgent; record win rate.
3. Parse the per-iteration promotion win-rate from the training log
   (new_wins / (new_wins + prev_wins), excluding all-draw iterations).
4. Plot both curves using matplotlib and save to src/results/.
5. Print a plain-text verdict on the win-rate-vs-random trend.

Usage
-----
    python src/eval/evaluate.py \
        --checkpoint-dir checkpoints/connect_four_50 \
        --log            src/logs/train_50iters_20260922_235523.log \
        --n-games        30 \
        --output-dir     src/results
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

# --- resolve project root on sys.path --------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO = os.path.join(_ROOT, "reference-repo")
for _p in (_ROOT, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.games.connect_four import ConnectFourGame
from src.nets.connect_four_nnet import ConnectFourNNet, NNetArgs
from src.eval.game_runner import RandomAgent, NNetGreedyAgent, play_match


# ===========================================================================
# 1.  Checkpoint discovery
# ===========================================================================

def find_iter_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    Return sorted list of (iteration_number, absolute_path) for every
    iter_NNN.pth.tar file found in checkpoint_dir.
    """
    pattern = re.compile(r"^iter_(\d+)\.pth\.tar$")
    results = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            iteration = int(m.group(1))
            results.append((iteration, os.path.join(checkpoint_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


# ===========================================================================
# 2.  Win-rate vs random baseline
# ===========================================================================

def evaluate_checkpoints_vs_random(
    checkpoints: List[Tuple[int, str]],
    game: ConnectFourGame,
    n_games: int = 30,
) -> Dict[int, dict]:
    """
    For each checkpoint, load it and play n_games vs RandomAgent.

    Returns dict mapping iteration -> match result dict.
    """
    results = {}
    random_agent = RandomAgent()
    total = len(checkpoints)

    # We keep one NNet instance and reload weights — avoids re-allocating the
    # model graph for every checkpoint (faster on MPS).
    nnet = ConnectFourNNet(game)

    for idx, (iteration, ckpt_path) in enumerate(checkpoints, start=1):
        # Load weights from this checkpoint
        import torch
        payload  = torch.load(ckpt_path, map_location=nnet.device)
        nnet.model.load_state_dict(payload["state_dict"])
        nnet.model.eval()

        nnet_agent = NNetGreedyAgent(nnet)

        match = play_match(game, nnet_agent, random_agent, n_games=n_games)
        results[iteration] = match

        bar_filled = int(30 * idx / total)
        bar        = "█" * bar_filled + "░" * (30 - bar_filled)
        win_pct    = match["win_rate_a"] * 100
        print(
            f"  [{bar}] iter {iteration:>3}/{checkpoints[-1][0]}  "
            f"W={match['wins_a']:>2} L={match['wins_b']:>2} D={match['draws']:>2}  "
            f"win%={win_pct:5.1f}%",
            flush=True,
        )

    return results


# ===========================================================================
# 3.  Parse promotion win-rates from training log
# ===========================================================================

def parse_promotion_rates(log_path: str) -> Dict[int, Optional[float]]:
    """
    Extract per-iteration arena results from a TimedCoach log file.

    Pattern:
        INFO ─── Starting Iter #N/50 ───
        ...
        INFO   Arena result   : new=X prev=Y draws=Z  (...)

    For iterations where new+prev == 0 (all draws), stores None.

    Returns dict: iteration -> promotion_win_rate or None.
    """
    iter_pattern  = re.compile(r"Starting Iter #(\d+)/\d+")
    arena_pattern = re.compile(
        r"Arena result\s*:\s*new=(\d+)\s+prev=(\d+)\s+draws=(\d+)"
    )

    rates: Dict[int, Optional[float]] = {}
    current_iter: Optional[int] = None

    with open(log_path, "r") as fh:
        for line in fh:
            m_iter = iter_pattern.search(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                continue

            m_arena = arena_pattern.search(line)
            if m_arena and current_iter is not None:
                new_w  = int(m_arena.group(1))
                prev_w = int(m_arena.group(2))
                decisive = new_w + prev_w
                rate     = new_w / decisive if decisive > 0 else None
                rates[current_iter] = rate
                current_iter = None   # consumed

    return rates


# ===========================================================================
# 4.  Trend analysis
# ===========================================================================

def compute_verdict(
    iterations: List[int],
    win_rates:  List[float],
    noise_threshold: float = 0.08,
    trend_threshold: float = 0.02,
) -> str:
    """
    Decide if the win-rate-vs-random curve is trending up, flat, or too noisy.

    Method
    ------
    - Fit a linear regression (slope in win-rate/iteration).
    - Measure residual std around the regression line.
    - If residual_std > noise_threshold: "TOO NOISY TO TELL"
    - Elif slope > +trend_threshold/iter: "TRENDING UP"
    - Elif slope < -trend_threshold/iter: "TRENDING DOWN"
    - Else: "FLAT"

    Also compare the mean of the first quarter vs the last quarter.
    """
    x = np.array(iterations, dtype=float)
    y = np.array(win_rates,  dtype=float)

    # Linear regression
    x_c    = x - x.mean()
    slope  = float(np.dot(x_c, y) / np.dot(x_c, x_c))     # win-rate / iter
    fitted = slope * x_c + y.mean()
    resid  = y - fitted
    resid_std = float(resid.std())

    n = len(x)
    q = max(1, n // 4)
    mean_first = float(y[:q].mean())
    mean_last  = float(y[-q:].mean())
    delta      = mean_last - mean_first

    return {
        "slope":      slope,
        "resid_std":  resid_std,
        "mean_first": mean_first,
        "mean_last":  mean_last,
        "delta":      delta,
        "verdict":    (
            "TOO NOISY TO TELL"  if resid_std > noise_threshold else
            "TRENDING UP"        if slope > trend_threshold / (x[-1] - x[0] + 1) else
            "TRENDING DOWN"      if slope < -trend_threshold / (x[-1] - x[0] + 1) else
            "FLAT"
        ),
    }


# ===========================================================================
# 5.  Plotting
# ===========================================================================

PALETTE = {
    "win_random":    "#4FC3F7",   # sky blue
    "win_random_bg": "#1A2633",
    "promo":         "#FFB74D",   # amber
    "trend":         "#EF5350",   # red
    "accept_line":   "#66BB6A",   # green
    "grid":          "#2C3E50",
    "bg":            "#0F1923",
    "panel_bg":      "#162030",
    "text":          "#ECF0F1",
}

def _rolling_mean(values: List[float], window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), lo + window)
        out[i] = arr[lo:hi].mean()
    return out


def plot_curves(
    vs_random:     Dict[int, dict],
    promo_rates:   Dict[int, Optional[float]],
    verdict_info:  dict,
    output_path:   str,
) -> None:
    """
    Two-panel figure:
      Top    — Win rate vs RandomAgent per checkpoint (bar + rolling mean + trend line)
      Bottom — Promotion win-rate from training log (scatter + rolling mean)
    """
    # --- data prep ----------------------------------------------------------
    iters_vr   = sorted(vs_random.keys())
    wr_vals    = [vs_random[i]["win_rate_a"] for i in iters_vr]
    roll_vr    = _rolling_mean(wr_vals, window=7)

    iters_pr   = sorted(k for k, v in promo_rates.items() if v is not None)
    pr_vals    = [promo_rates[i] for i in iters_pr]
    roll_pr    = _rolling_mean(pr_vals, window=5)

    # Trend line for top panel
    x_arr = np.array(iters_vr, dtype=float)
    x_c   = x_arr - x_arr.mean()
    slope = verdict_info["slope"]
    trend_y = slope * x_c + np.mean(wr_vals)

    accept_threshold = 0.55

    # --- figure setup -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), facecolor=PALETTE["bg"],
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35},
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["panel_bg"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.5)

    # ── Top panel: win rate vs random ────────────────────────────────────────
    ax1.bar(iters_vr, wr_vals, color=PALETTE["win_random"], alpha=0.5,
            width=0.7, label="Win rate vs random (per checkpoint)")
    ax1.plot(iters_vr, roll_vr, color=PALETTE["win_random"],
             linewidth=2.0, label="Rolling mean (w=7)")
    ax1.plot(iters_vr, trend_y, color=PALETTE["trend"], linewidth=1.5,
             linestyle="--", label=f"Linear trend  (slope={slope*50:.3f}/50 iters)")
    ax1.axhline(0.5,  color=PALETTE["text"],       linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.axhline(accept_threshold, color=PALETTE["accept_line"],
                linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.text(iters_vr[-1] + 0.3, 0.51,  "50%",  color=PALETTE["text"],
             fontsize=7, va="bottom", alpha=0.6)
    ax1.text(iters_vr[-1] + 0.3, accept_threshold + 0.01, "55%",
             color=PALETTE["accept_line"], fontsize=7, va="bottom")

    # Shade first / last quarter
    q = max(1, len(iters_vr) // 4)
    ax1.axvspan(iters_vr[0],  iters_vr[q-1],  alpha=0.08, color="white", label="First quarter")
    ax1.axvspan(iters_vr[-q], iters_vr[-1],   alpha=0.08, color="yellow", label="Last quarter")

    verdict_colour = {"TRENDING UP": "#66BB6A", "FLAT": "#FFB74D",
                      "TRENDING DOWN": "#EF5350", "TOO NOISY TO TELL": "#CE93D8"}
    vtext = verdict_info["verdict"]
    vcolour = verdict_colour.get(vtext, PALETTE["text"])
    ax1.set_title(
        f"Win Rate vs Random Baseline  —  Verdict: {vtext}",
        fontsize=13, fontweight="bold", pad=8, color=vcolour,
    )
    ax1.set_xlabel("Training iteration", fontsize=10)
    ax1.set_ylabel("Win rate (decisive games)", fontsize=10)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax1.set_xlim(iters_vr[0] - 0.5, iters_vr[-1] + 0.5)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="lower right", fontsize=8, facecolor=PALETTE["panel_bg"],
               labelcolor=PALETTE["text"], framealpha=0.7)

    # First/last quarter annotations
    ax1.annotate(
        f"First Q avg\n{verdict_info['mean_first']:.1%}",
        xy=(iters_vr[q // 2], verdict_info["mean_first"]),
        xytext=(iters_vr[q // 2] + 2, verdict_info["mean_first"] + 0.15),
        color=PALETTE["text"], fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=0.8),
    )
    ax1.annotate(
        f"Last Q avg\n{verdict_info['mean_last']:.1%}",
        xy=(iters_vr[-q // 2 - 1], verdict_info["mean_last"]),
        xytext=(iters_vr[-q // 2 - 1] - 8, verdict_info["mean_last"] + 0.15),
        color=PALETTE["text"], fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=0.8),
    )

    # ── Bottom panel: promotion win-rate from log ─────────────────────────────
    ax2.scatter(iters_pr, pr_vals, color=PALETTE["promo"], s=35, zorder=3,
                label="Promotion win rate (decisive arena games)")
    ax2.plot(iters_pr, roll_pr, color=PALETTE["promo"], linewidth=1.8,
             label="Rolling mean (w=5)")
    ax2.axhline(accept_threshold, color=PALETTE["accept_line"], linewidth=0.8,
                linestyle="--", label="55% acceptance threshold")

    # Shade all-draw iterations
    all_draw_iters = [k for k, v in promo_rates.items() if v is None]
    for it in all_draw_iters:
        ax2.axvspan(it - 0.5, it + 0.5, alpha=0.25, color="#CE93D8", linewidth=0)
    if all_draw_iters:
        ax2.plot([], [], color="#CE93D8", alpha=0.5, linewidth=8,
                 label=f"All-draw arena ({len(all_draw_iters)} iters)")

    ax2.set_title("Promotion Win Rate (Training Log — new vs prev model)",
                  fontsize=11, fontweight="bold", pad=6)
    ax2.set_xlabel("Training iteration", fontsize=10)
    ax2.set_ylabel("New model win rate", fontsize=10)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_xlim(iters_vr[0] - 0.5, iters_vr[-1] + 0.5)
    ax2.set_ylim(-0.05, 1.10)
    ax2.legend(loc="lower right", fontsize=8, facecolor=PALETTE["panel_bg"],
               labelcolor=PALETTE["text"], framealpha=0.7)

    # ── Global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Connect Four — AlphaZero Training Run 1  (50 iterations, numEps=30, numMCTSSims=25)",
        fontsize=12, color=PALETTE["text"], y=0.98,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"\n  Figure saved → {output_path}")


# ===========================================================================
# 6.  Verdict printer
# ===========================================================================

def print_verdict(verdict_info: dict, n_games: int) -> None:
    slope_per_50 = verdict_info["slope"] * 50
    v = verdict_info["verdict"]

    border = "═" * 62
    print(f"\n{border}")
    print("  EVALUATION VERDICT")
    print(border)
    print(f"  Benchmark        : {n_games} games vs RandomAgent per checkpoint")
    print(f"  First-quarter avg: {verdict_info['mean_first']:.1%}  "
          f"(iters 1–{len([1])*12})")
    print(f"  Last-quarter avg : {verdict_info['mean_last']:.1%}")
    print(f"  Delta (last-first): {verdict_info['delta']:+.1%}")
    print(f"  Linear trend     : {slope_per_50:+.3f} win-rate per 50 iters")
    print(f"  Residual std     : {verdict_info['resid_std']:.3f}  "
          f"({'low noise' if verdict_info['resid_std'] < 0.08 else 'high noise'})")
    print()

    verdict_msgs = {
        "TRENDING UP":    (
            "  ✓  VERDICT: TRENDING UP\n"
            "  The win-rate-vs-random curve shows a consistent upward slope.\n"
            "  The model is genuinely improving. Extending training is warranted.\n"
            "  RECOMMENDATION: Continue for another 50 iterations.\n"
            "  For better quality, consider numMCTSSims=50 to break draw plateau."
        ),
        "FLAT":           (
            "  ~  VERDICT: FLAT\n"
            "  The curve is roughly level. The model has likely plateaued at this\n"
            "  training scale. More iterations at the same config will not help much.\n"
            "  RECOMMENDATION: Increase numMCTSSims (25->50) before continuing."
        ),
        "TRENDING DOWN":  (
            "  ✗  VERDICT: TRENDING DOWN\n"
            "  Win rate appears to decrease over iterations. This is unusual and\n"
            "  may indicate catastrophic forgetting or training instability.\n"
            "  RECOMMENDATION: Investigate acceptance rate; consider tuning lr."
        ),
        "TOO NOISY TO TELL": (
            "  ?  VERDICT: TOO NOISY TO TELL\n"
            "  High variance across checkpoints — individual game results dominate.\n"
            "  The rolling mean may still show a trend (check the figure).\n"
            "  RECOMMENDATION: Increase n_games (try 60-100) for a cleaner signal."
        ),
    }
    print(verdict_msgs.get(v, f"  VERDICT: {v}"))
    print(border + "\n")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Connect Four checkpoints vs random baseline."
    )
    p.add_argument("--checkpoint-dir", default="checkpoints/connect_four_50",
                   help="Directory containing iter_NNN.pth.tar files")
    p.add_argument("--log", default=None,
                   help="Training log file (auto-detected from src/logs/ if omitted)")
    p.add_argument("--n-games", type=int, default=30,
                   help="Games vs random per checkpoint (default 30)")
    p.add_argument("--output-dir", default="src/results",
                   help="Directory to save the figure")
    return p.parse_args()


def _auto_detect_log(log_dir: str) -> Optional[str]:
    logs = sorted(
        f for f in os.listdir(log_dir)
        if f.startswith("train_") and f.endswith(".log")
    )
    return os.path.join(log_dir, logs[-1]) if logs else None


def main() -> None:
    cli = parse_args()
    os.makedirs(cli.output_dir, exist_ok=True)

    # ── Checkpoint discovery ────────────────────────────────────────────────
    ckpt_dir = os.path.join(_ROOT, cli.checkpoint_dir)
    checkpoints = find_iter_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No iter_NNN.pth.tar files found in {ckpt_dir!r}")
    print(f"\n  Found {len(checkpoints)} checkpoints in {ckpt_dir}")
    print(f"  Range: iter {checkpoints[0][0]} → iter {checkpoints[-1][0]}")

    # ── Log detection ───────────────────────────────────────────────────────
    log_path = cli.log or _auto_detect_log(os.path.join(_ROOT, "src", "logs"))
    if log_path and not os.path.isabs(log_path):
        log_path = os.path.join(_ROOT, log_path)
    if not log_path or not os.path.isfile(log_path):
        print("  WARNING: No training log found — promotion curve will be empty.")
        log_path = None

    # ── Game ────────────────────────────────────────────────────────────────
    game = ConnectFourGame()

    # ── Step 2: evaluate vs random ──────────────────────────────────────────
    print(f"\n  Evaluating {len(checkpoints)} checkpoints × {cli.n_games} games vs RandomAgent …")
    vs_random = evaluate_checkpoints_vs_random(checkpoints, game, n_games=cli.n_games)

    # ── Step 3: parse promotion rates ───────────────────────────────────────
    promo_rates: Dict[int, Optional[float]] = {}
    if log_path:
        promo_rates = parse_promotion_rates(log_path)
        decisive_count = sum(1 for v in promo_rates.values() if v is not None)
        draw_count     = sum(1 for v in promo_rates.values() if v is None)
        print(f"\n  Parsed {len(promo_rates)} arena results from log "
              f"({decisive_count} decisive, {draw_count} all-draw)")

    # ── Step 4: verdict ─────────────────────────────────────────────────────
    iters_vr = sorted(vs_random.keys())
    wr_vals  = [vs_random[i]["win_rate_a"] for i in iters_vr]
    verdict  = compute_verdict(iters_vr, wr_vals)

    # ── Step 5: plot ────────────────────────────────────────────────────────
    fig_path = os.path.join(
        os.path.join(_ROOT, cli.output_dir),
        "training_curves.png",
    )
    plot_curves(vs_random, promo_rates, verdict, fig_path)

    # ── Step 6: print verdict ───────────────────────────────────────────────
    print_verdict(verdict, cli.n_games)

    # ── Raw numbers table ───────────────────────────────────────────────────
    print("  Per-checkpoint win rate table")
    print(f"  {'Iter':>4}  {'W':>3}  {'L':>3}  {'D':>3}  {'Win%':>6}  {'Promo%':>7}")
    print("  " + "─" * 38)
    for it in iters_vr:
        r  = vs_random[it]
        pr = promo_rates.get(it)
        pr_str = f"{pr:6.1%}" if pr is not None else "  draw"
        print(f"  {it:>4}  {r['wins_a']:>3}  {r['wins_b']:>3}  "
              f"{r['draws']:>3}  {r['win_rate_a']:6.1%}  {pr_str:>7}")


if __name__ == "__main__":
    main()
