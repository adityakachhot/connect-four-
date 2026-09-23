"""
generate_report_assets.py
Generates all figures and game transcripts for src/REPORT.md.

Outputs (all in src/results/report/):
  fig_winrate_combined.png   — win-rate vs random, all 3 runs
  fig_arena_combined.png     — arena promotion win-rate, all 3 runs
  fig_losses.png             — pi/v loss curves, all 3 runs (subplots)
  transcript_early.txt       — game vs random, iter_005 of run 1
  transcript_late.txt        — game vs random, run 3 best checkpoint
"""

from __future__ import annotations
import os, re, sys, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reference-repo"))

OUT  = os.path.join(SRC, "results", "report")
os.makedirs(OUT, exist_ok=True)

from src.games.connect_four import ConnectFourGame
from src.nets.connect_four_nnet import ConnectFourNNet
from src.eval.game_runner import NNetGreedyAgent, RandomAgent, play_game

# ── palette ────────────────────────────────────────────────────────────────────
BG      = "#0F1923"
PANEL   = "#162030"
GRID    = "#2C3E50"
TEXT    = "#ECF0F1"
C = {
    "run1": "#4FC3F7",   # sky blue
    "run2": "#FFB74D",   # amber
    "run3": "#A5D6A7",   # sage green
}

def _fig(rows=1, cols=1, h=6, w=14):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h*rows),
                             facecolor=BG, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
    return fig, axes

def _roll(v, w=7):
    arr = np.array(v, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo, hi = max(0, i-w//2), min(len(arr), i-w//2+w)
        out[i] = arr[lo:hi].mean()
    return out

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Win-rate-vs-random  (hard-coded from evaluation outputs)
# ══════════════════════════════════════════════════════════════════════════════
WR = {
    "run1": [83.3,66.7,76.7,70.0,73.3,73.3,66.7,76.7,79.3,70.0,
             86.7,86.7,93.3,82.8,89.7,83.3,86.7,83.3,93.3,83.3,
             83.3,86.2,89.7,82.1,82.8,75.9,90.0,82.8,89.7,93.1,
             86.7,93.3,90.0,86.7,80.0,93.3,93.3,93.1,75.9,80.0,
             86.7,86.7,89.7,73.3,80.0,96.7,100.0,86.7,93.3,83.3],
    "run2": [66.7,63.3,60.0,50.0,83.3,76.7,83.3,56.7,70.0,76.7,
             72.4,76.7,86.7,80.0,86.7,83.3,86.7,79.3,96.7,86.7,
             86.7,76.0,82.1,82.8,76.7,82.8,85.2,76.7,88.9,86.2,
             85.2,79.3,82.8,90.0,84.6,90.0,90.0,82.8,81.5,86.7,
             96.7,85.7,85.2,86.7,89.7,89.7,69.0,89.3,93.3,82.1],
    "run3": [82.8,85.7,96.3,92.0,88.5,85.7,89.7,82.8,75.0,93.3,
             92.9,93.3,89.7,83.3,71.4,79.3,79.3,82.8,82.1,83.3,
             89.7,78.6,92.9,82.8,89.3,86.2,85.7,86.2,86.2,89.3,
             86.7,92.6,82.8,93.1,93.3,100.0,93.3,82.1,79.3,82.1,
             93.1,96.3,93.1,83.3,92.6,85.7,92.9,96.6,88.9,85.2],
}
ITERS = list(range(1, 51))

fig, axes = _fig(1, 1, h=6)
ax = axes[0, 0]
for tag, label in [("run1","Run 1 (25 sims, fresh)"),
                    ("run2","Run 2 (50 sims, fresh)"),
                    ("run3","Run 3 (50 sims, resumed)")]:
    vals = [v/100 for v in WR[tag]]
    ax.plot(ITERS, vals, color=C[tag], alpha=0.25, lw=0.8)
    ax.plot(ITERS, _roll(vals)/100 if max(vals)>1 else _roll(vals),
            color=C[tag], lw=2.2, label=label)
ax.axhline(0.5,  color=TEXT, lw=0.7, ls=":", alpha=0.5)
ax.axhline(0.9,  color=TEXT, lw=0.7, ls=":", alpha=0.4)
ax.set_title("Win Rate vs Random Baseline — All Runs (rolling mean w=7)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Training iteration"); ax.set_ylabel("Win rate (decisive games)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0.3, 1.08); ax.set_xlim(1, 50)
ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9, loc="lower right")
fig.suptitle("Connect Four AlphaZero — Win Rate vs Random", color=TEXT, fontsize=11, y=0.99)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_winrate_combined.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ fig_winrate_combined.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Arena promotion win-rate
# ══════════════════════════════════════════════════════════════════════════════
def parse_promo(log_path):
    iter_pat  = re.compile(r"Starting Iter #(\d+)")
    arena_pat = re.compile(r"Arena result\s*:\s*new=(\d+)\s+prev=(\d+)\s+draws=(\d+)")
    rates, cur = {}, None
    with open(log_path) as f:
        for line in f:
            m = iter_pat.search(line)
            if m: cur = int(m.group(1)); continue
            m = arena_pat.search(line)
            if m and cur:
                n, p = int(m.group(1)), int(m.group(2))
                rates[cur] = n/(n+p) if (n+p) > 0 else None
                cur = None
    return rates

LOGS = {
    "run1": os.path.join(SRC, "logs", "train_50iters_20260922_235523.log"),
    "run2": os.path.join(SRC, "logs", "train_run2_20260923_005937.log"),
    "run3": os.path.join(SRC, "logs", "train_run3.log"),
}
PROMO = {tag: parse_promo(path) for tag, path in LOGS.items()}

fig, axes = _fig(1, 1, h=6)
ax = axes[0, 0]
for tag, label in [("run1","Run 1"), ("run2","Run 2"), ("run3","Run 3")]:
    xs = sorted(k for k, v in PROMO[tag].items() if v is not None)
    ys = [PROMO[tag][k] for k in xs]
    ax.scatter(xs, ys, color=C[tag], s=25, alpha=0.6, zorder=3)
    if len(xs) > 3:
        ax.plot(xs, _roll(ys, w=5), color=C[tag], lw=2.0, label=label)
    else:
        ax.plot(xs, ys, color=C[tag], lw=2.0, label=label)
    draw_xs = [k for k, v in PROMO[tag].items() if v is None]
    for dx in draw_xs:
        ax.axvspan(dx-0.5, dx+0.5, alpha=0.08, color=C[tag], lw=0)
ax.axhline(0.55, color="#EF5350", lw=1, ls="--", alpha=0.7, label="55% threshold")
ax.set_title("Arena Promotion Win Rate (New vs Previous Best) — All Runs",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Training iteration"); ax.set_ylabel("New model win rate (decisive games)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(-0.05, 1.12); ax.set_xlim(1, 50)
ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_arena_combined.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ fig_arena_combined.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Loss curves
# ══════════════════════════════════════════════════════════════════════════════
def parse_losses(log_path):
    """Extract final-epoch (epoch 5/5) pi_loss and v_loss per training iteration."""
    iter_pat  = re.compile(r"Starting Iter #(\d+)")
    epoch_pat = re.compile(r"Epoch\s+5/5\s+pi_loss=([0-9.]+)\s+v_loss=([0-9.]+)")
    # Each iter has exactly one epoch-5 line
    pi_losses, v_losses = [], []
    cur_iter = None
    with open(log_path) as f:
        for line in f:
            m = iter_pat.search(line)
            if m: cur_iter = int(m.group(1))
            m = epoch_pat.search(line)
            if m:
                pi_losses.append((cur_iter, float(m.group(1))))
                v_losses.append((cur_iter, float(m.group(2))))
    pi_losses.sort(); v_losses.sort()
    return ([x[0] for x in pi_losses], [x[1] for x in pi_losses],
            [x[0] for x in v_losses],  [x[1] for x in v_losses])

LOSSES = {tag: parse_losses(path) for tag, path in LOGS.items()}

fig, axes = _fig(2, 1, h=4.5, w=14)
for row, (loss_idx, ylabel, title) in enumerate([
        (1, "Policy loss (cross-entropy)", "Policy Head Loss — Final Epoch per Iteration"),
        (3, "Value loss (MSE)",            "Value Head Loss — Final Epoch per Iteration"),
]):
    ax = axes[row, 0]
    for tag, label in [("run1","Run 1 (25 sims)"),
                        ("run2","Run 2 (50 sims, fresh)"),
                        ("run3","Run 3 (50 sims, resumed)")]:
        d = LOSSES[tag]
        xs, ys = d[0], d[loss_idx]
        ax.plot(xs, ys, color=C[tag], alpha=0.3, lw=0.8)
        ax.plot(xs, _roll(ys, w=5), color=C[tag], lw=2.0, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Training iteration"); ax.set_ylabel(ylabel)
    ax.set_xlim(1, 50); ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=9)

fig.suptitle("Neural Network Training Losses", color=TEXT, fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_losses.png"), dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ fig_losses.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Game transcripts
# ══════════════════════════════════════════════════════════════════════════════
import torch

ROWS, COLS = 6, 7
COL_LABELS = " ".join(str(c+1) for c in range(COLS))

def board_str(board, last_move_col=None):
    """Pretty-print a 6x7 board.  1=X (player1)  -1=O (player2)  0=."""
    lines = ["┌" + "─"*15 + "┐"]
    for r in range(ROWS):
        row_chars = []
        for c in range(COLS):
            v = board[r][c]
            ch = "X" if v == 1 else ("O" if v == -1 else "·")
            if c == last_move_col and r == next(
                (rr for rr in range(ROWS-1,-1,-1) if board[rr][c] != 0), -1):
                ch = f"\033[1m{ch}\033[0m"  # bold last move (terminal only)
            row_chars.append(ch)
        lines.append("│ " + " ".join(row_chars) + " │")
    lines.append("└" + "─"*15 + "┘")
    lines.append("  " + COL_LABELS)
    return "\n".join(lines)

def record_game(game, agent1, agent2, label1="NNet", label2="Rnd"):
    """Play a game and return a list of (move, board, player) tuples."""
    board  = game.getInitBoard()
    player = 1
    agents = {1: agent1, -1: agent2}
    names  = {1: label1, -1: label2}
    history = []
    move_num = 0
    while True:
        canonical = game.getCanonicalForm(board, player)
        valid     = game.getValidMoves(canonical, 1)
        action    = agents[player].select_action(canonical, valid)
        col       = action  # action == column index
        history.append((move_num, player, names[player], col, board.copy()))
        board, player = game.getNextState(board, player, action)
        move_num += 1
        result = game.getGameEnded(board, 1)
        if result != 0:
            history.append((move_num, None, "END", None, board.copy()))
            break
    return history, result

def transcript_str(history, result, ckpt_label):
    lines = [f"{'='*52}", f"  Game Transcript — {ckpt_label}", f"{'='*52}",
             "  X = NNet agent (player 1)   O = Random agent (player 2)", ""]
    for (mn, player, name, col, brd) in history:
        if name == "END":
            outcome = "X wins" if result == 1 else ("O wins" if result == -1 else "Draw")
            lines += [f"  ── Game over: {outcome} ──", board_str(brd)]
        else:
            lines += [f"  Move {mn+1:2d}  [{name}] → column {col+1}", board_str(brd), ""]
    return "\n".join(lines)

game = ConnectFourGame()
random.seed(42)

for ckpt_path, out_name, label in [
    (os.path.join(ROOT, "checkpoints", "connect_four_50", "iter_005.pth.tar"),
     "transcript_early.txt", "Run 1 — Iter 5 (early checkpoint)"),
    (os.path.join(ROOT, "checkpoints", "connect_four_run3", "best.pth.tar"),
     "transcript_late.txt",  "Run 3 — Best (latest checkpoint)"),
]:
    nnet = ConnectFourNNet(game)
    payload = torch.load(ckpt_path, map_location=nnet.device)
    nnet.model.load_state_dict(payload["state_dict"])
    nnet.model.eval()

    nnet_agent = NNetGreedyAgent(nnet)
    rnd_agent  = RandomAgent()

    # Play as player 1 (NNet goes first)
    history, result = record_game(game, nnet_agent, rnd_agent, "NNet", "Random")
    txt = transcript_str(history, result, label)
    out_path = os.path.join(OUT, out_name)
    with open(out_path, "w") as f:
        f.write(txt)
    print(f"✓ {out_name}  (result={'NNet wins' if result==1 else 'Random wins' if result==-1 else 'Draw'})")

print("\nAll assets generated →", OUT)
