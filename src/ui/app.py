"""
Connect Four — AlphaZero Studio
================================
A graphical interface for Connect Four powered by AlphaZero (MCTS + PyTorch CNN)
and classical Minimax baseline.

Modes:
  1. PLAY vs AGENT        — Play as Human vs Trained Agent (or Minimax)
  2. WATCH AGENT vs AGENT — Head-to-head match between two checkpoints with auto-play
  3. REPLAY TRANSCRIPT    — Step through recorded match transcripts from src/results/report/

Runs on Apple Silicon (MPS) or CPU.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ─── Resolve project root & reference-repo on sys.path ───────────────────────
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

# ─── Page Configuration & Styling ───────────────────────────────────────────
st.set_page_config(
    page_title="Connect Four — AlphaZero Studio",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header typography */
    .studio-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .studio-sub {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }

    /* Connect Four Board Container */
    .c4-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 12px 0 16px 0;
    }
    .c4-board {
        background: linear-gradient(160deg, #1d4ed8 0%, #1e3a8a 100%);
        padding: 16px;
        border-radius: 20px;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.7), inset 0 2px 4px rgba(255, 255, 255, 0.25);
        display: inline-grid;
        grid-template-columns: repeat(7, 54px);
        grid-template-rows: repeat(6, 54px);
        gap: 10px;
        border: 3px solid #3b82f6;
    }
    .c4-slot {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        position: relative;
        box-shadow: inset 0 4px 8px rgba(0, 0, 0, 0.7);
        transition: transform 0.2s ease;
    }
    .c4-empty {
        background: radial-gradient(circle at 40% 40%, #0f172a, #030712);
    }
    .c4-red {
        background: radial-gradient(circle at 35% 35%, #ff4b4b, #991b1b);
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.4);
    }
    .c4-yellow {
        background: radial-gradient(circle at 35% 35%, #fde047, #b45309);
        box-shadow: 0 4px 10px rgba(251, 191, 36, 0.5), inset 0 2px 4px rgba(255, 255, 255, 0.4);
    }
    .c4-highlight {
        outline: 3px solid #38bdf8;
        outline-offset: 2px;
        animation: pulseRing 1.5s infinite alternate;
    }
    @keyframes pulseRing {
        from { outline-color: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
        to { outline-color: #a855f7; box-shadow: 0 0 16px #a855f7; }
    }

    /* Move history cards */
    .history-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px;
        max-height: 420px;
        overflow-y: auto;
    }
    .history-item {
        padding: 6px 10px;
        border-bottom: 1px solid #1e293b;
        font-family: monospace;
        font-size: 0.9rem;
        display: flex;
        justify-content: space-between;
    }
    .history-item.latest {
        background: rgba(59, 130, 246, 0.15);
        border-left: 3px solid #3b82f6;
        font-weight: bold;
    }

    /* Diagnostics cards */
    .diag-card {
        background: #131d31;
        border: 1px solid #1e3a8a;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Checkpoint Catalog ───────────────────────────────────────────────────────
CHECKPOINT_CATALOG = {
    "Run 3 Best (Warm-started, 50 iters)": os.path.join(_ROOT, "checkpoints", "connect_four_run3", "best.pth.tar"),
    "Run 2 Best (50 MCTS sims, 50 iters)": os.path.join(_ROOT, "checkpoints", "connect_four_run2", "best.pth.tar"),
    "Run 1 Best (25 MCTS sims, 50 iters)": os.path.join(_ROOT, "checkpoints", "connect_four_50", "best.pth.tar"),
    "Run 1 Early (Iter 5)": os.path.join(_ROOT, "checkpoints", "connect_four_50", "iter_005.pth.tar"),
    "Run 1 Early (Iter 1)": os.path.join(_ROOT, "checkpoints", "connect_four_50", "iter_001.pth.tar"),
    "Run 2 Mid (Iter 25)": os.path.join(_ROOT, "checkpoints", "connect_four_run2", "iter_025.pth.tar"),
    "Run 3 Early (Iter 5)": os.path.join(_ROOT, "checkpoints", "connect_four_run3", "iter_005.pth.tar"),
}


# ─── Cached Model Loading (fast inference on MPS) ───────────────────────────
@st.cache_resource
def get_game() -> ConnectFourGame:
    return ConnectFourGame()


@st.cache_resource
def load_cached_nnet(checkpoint_path: str) -> ConnectFourNNet:
    game = get_game()
    nnet = ConnectFourNNet(game)
    folder, filename = os.path.split(checkpoint_path)
    nnet.load_checkpoint(folder, filename)
    nnet.model.eval()
    return nnet


# ─── Board HTML Renderer ─────────────────────────────────────────────────────
def render_board_html(board: np.ndarray, last_move: Optional[Tuple[int, int]] = None) -> str:
    """
    Render 6x7 Connect Four board as custom HTML.
    board: 6x7 array with 1 (Red), -1 (Yellow), 0 (Empty)
    """
    rows_html = []
    for r in range(6):
        for c in range(7):
            val = int(board[r, c])
            is_last = (last_move is not None and last_move[0] == r and last_move[1] == c)
            highlight_cls = " c4-highlight" if is_last else ""
            if val == 1:
                chip_cls = f"c4-red{highlight_cls}"
            elif val == -1:
                chip_cls = f"c4-yellow{highlight_cls}"
            else:
                chip_cls = "c4-empty"
            rows_html.append(f'<div class="c4-slot {chip_cls}"></div>')

    inner = "".join(rows_html)
    return f'<div class="c4-wrapper"><div class="c4-board">{inner}</div></div>'


def find_dropped_row(board: np.ndarray, col: int) -> int:
    """Find row where disc settles in column col."""
    for r in reversed(range(6)):
        if board[r, col] == 0:
            return r
    return 0


def parse_transcript_file(path: str) -> Tuple[str, Optional[str], List[Dict]]:
    """Parse transcript .txt file into (title, outcome, list of step dictionaries)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    title = lines[1].strip() if len(lines) > 1 else os.path.basename(path)

    steps = []
    cur_board = np.zeros((6, 7), dtype=int)
    steps.append({
        "move_num": 0,
        "player": "None",
        "col": None,
        "board": cur_board.copy(),
        "info": "Start of game",
    })

    move_pattern = re.compile(r"Move\s+(\d+)\s+\[(.*?)\]\s+→\s+column\s+(\d+)")
    end_pattern = re.compile(r"──\s+Game over:\s+(.*?)\s+──")

    current_move = None
    board_lines = []
    in_board = False
    outcome = None

    for line in lines:
        m = move_pattern.search(line)
        if m:
            current_move = (int(m.group(1)), m.group(2), int(m.group(3)) - 1)
            continue
        e = end_pattern.search(line)
        if e:
            outcome = e.group(1).strip()
            continue
        if "┌" in line:
            in_board = True
            board_lines = []
            continue
        if "└" in line:
            in_board = False
            grid = np.zeros((6, 7), dtype=int)
            for r_idx, r_line in enumerate(board_lines):
                chars = r_line.replace("│", "").strip().split()
                for c_idx, ch in enumerate(chars):
                    if ch == "X":
                        grid[r_idx, c_idx] = 1
                    elif ch == "O":
                        grid[r_idx, c_idx] = -1
            if current_move:
                steps.append({
                    "move_num": current_move[0],
                    "player": current_move[1],
                    "col": current_move[2],
                    "board": grid,
                    "info": f"Move {current_move[0]}: [{current_move[1]}] played Column {current_move[2]+1}",
                })
                current_move = None
            elif outcome:
                steps.append({
                    "move_num": steps[-1]["move_num"] if steps else 0,
                    "player": "END",
                    "col": None,
                    "board": grid,
                    "info": f"Game Over: {outcome}",
                })
            continue
        if in_board:
            board_lines.append(line)

    return title, outcome, steps


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.markdown("## 🧭 Navigation")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["🎮 Play vs Agent", "⚔️ Watch Agent vs Agent", "📜 Replay Transcript"],
    index=0,
)

# Display device info
device_name = "Apple Silicon (MPS)" if torch.backends.mps.is_available() else "CPU"
st.sidebar.markdown(f"**Compute Hardware:** `{device_name}`")
st.sidebar.markdown("---")

game = get_game()

# =============================================================================
# MODE 1: PLAY VS AGENT
# =============================================================================
if mode == "🎮 Play vs Agent":
    st.markdown('<div class="studio-header">Play vs Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="studio-sub">Challenge an AlphaZero neural network with MCTS or a classical Minimax searcher.</div>', unsafe_allow_html=True)

    # Sidebar settings
    st.sidebar.markdown("### ⚙️ Match Settings")
    opponent_type = st.sidebar.selectbox("Opponent Type", ["Trained Agent (AlphaZero)", "Minimax Baseline"])

    if opponent_type == "Trained Agent (AlphaZero)":
        ckpt_label = st.sidebar.selectbox("Checkpoint", list(CHECKPOINT_CATALOG.keys()), index=0)
        ckpt_path = CHECKPOINT_CATALOG[ckpt_label]
        difficulty_sims = st.sidebar.select_slider("MCTS Simulations (Difficulty)", options=[5, 25, 100], value=25)
        minimax_depth = None
    else:
        ckpt_path = CHECKPOINT_CATALOG["Run 3 Best (Warm-started, 50 iters)"]
        difficulty_sims = 25
        minimax_depth = st.sidebar.select_slider("Minimax Search Depth", options=[2, 4, 6], value=4)

    first_mover = st.sidebar.radio("Who Moves First?", ["Human (🔴 Red)", "Agent (🟡 Yellow)"], index=0)

    # Initialize session state for play mode
    if "play_board" not in st.session_state:
        st.session_state.play_board = game.getInitBoard()
        st.session_state.play_cur_player = 1
        st.session_state.play_game_over = False
        st.session_state.play_winner = 0
        st.session_state.play_last_move = None
        st.session_state.play_last_agent_val = None
        st.session_state.play_last_agent_pi = None
        st.session_state.play_history = []
        st.session_state.play_human_player = 1 if "Human" in first_mover else -1

    def start_new_play_game():
        st.session_state.play_board = game.getInitBoard()
        st.session_state.play_cur_player = 1
        st.session_state.play_game_over = False
        st.session_state.play_winner = 0
        st.session_state.play_last_move = None
        st.session_state.play_last_agent_val = None
        st.session_state.play_last_agent_pi = None
        st.session_state.play_history = []
        st.session_state.play_human_player = 1 if "Human" in first_mover else -1

        # If Agent moves first, take opening move
        if st.session_state.play_human_player == -1:
            make_agent_play_move()

    def make_agent_play_move():
        b = st.session_state.play_board
        agent_p = -st.session_state.play_human_player
        canonical = game.getCanonicalForm(b, agent_p)
        valids = game.getValidMoves(canonical, 1)

        # Policy & Value estimates
        nnet = load_cached_nnet(ckpt_path)
        pi, v = nnet.predict(canonical)

        if opponent_type == "Trained Agent (AlphaZero)":
            mcts = MCTS(game, nnet, dotdict({"numMCTSSims": difficulty_sims, "cpuct": 1.0}))
            probs = mcts.getActionProb(canonical, temp=0)
            action = int(np.argmax(probs))
            chosen_pi = probs[:7]
        else:
            mm_agent = MinimaxAgent(game, depth=minimax_depth)
            action = mm_agent.select_action(canonical, valids)
            chosen_pi = pi[:7]  # use nnet policy for chart visualization

        # Apply move
        r = find_dropped_row(b, action)
        new_board, next_p = game.getNextState(b, agent_p, action)
        st.session_state.play_board = new_board
        st.session_state.play_cur_player = next_p
        st.session_state.play_last_move = (r, action)
        st.session_state.play_last_agent_val = float(v)
        st.session_state.play_last_agent_pi = [float(p) for p in chosen_pi]
        st.session_state.play_history.append((f"Agent ({'🟡 Yellow' if agent_p == -1 else '🔴 Red'})", action + 1))

        # Check end condition
        end_val = game.getGameEnded(new_board, 1)
        if end_val != 0:
            st.session_state.play_game_over = True
            st.session_state.play_winner = 1 if end_val == 1 else (-1 if end_val == -1 else 0)

    if st.sidebar.button("🔄 New Game", use_container_width=True):
        start_new_play_game()
        st.rerun()

    # Layout: Board on Left (7 cols), Analytics on Right (5 cols)
    col_board, col_diag = st.columns([7, 5])

    with col_board:
        # Status banner
        human_p = st.session_state.play_human_player
        cur_p = st.session_state.play_cur_player
        is_over = st.session_state.play_game_over

        if is_over:
            w = st.session_state.play_winner
            if w == human_p:
                st.success("🎉 **VICTORY! You won the match!**")
            elif w == -human_p:
                st.error("🤖 **AGENT WINS! The machine found the winning combination.**")
            else:
                st.info("🤝 **DRAW! Full board stalemate.**")
        else:
            if cur_p == human_p:
                st.info(f"**Your turn ({'🔴 Red' if human_p == 1 else '🟡 Yellow'})** — Click a column button below to drop a disc.")
            else:
                st.warning("🤖 Agent is preparing move...")

        # Interactive Column Buttons (Col 1 to 7)
        btn_cols = st.columns(7)
        b = st.session_state.play_board
        valids = game.getValidMoves(b, cur_p)

        for c_idx in range(7):
            col_full = b[0, c_idx] != 0
            btn_disabled = is_over or (cur_p != human_p) or col_full
            with btn_cols[c_idx]:
                if st.button(f"⬇️ {c_idx+1}", key=f"drop_{c_idx}", disabled=btn_disabled, use_container_width=True):
                    # 1. Apply Human Move
                    r = find_dropped_row(b, c_idx)
                    new_b, next_p = game.getNextState(b, human_p, c_idx)
                    st.session_state.play_board = new_b
                    st.session_state.play_cur_player = next_p
                    st.session_state.play_last_move = (r, c_idx)
                    st.session_state.play_history.append((f"Human ({'🔴 Red' if human_p == 1 else '🟡 Yellow'})", c_idx + 1))

                    # 2. Check if Human won
                    res = game.getGameEnded(new_b, 1)
                    if res != 0:
                        st.session_state.play_game_over = True
                        st.session_state.play_winner = 1 if res == 1 else (-1 if res == -1 else 0)
                    else:
                        # 3. Agent replies automatically
                        make_agent_play_move()
                    st.rerun()

        # Render Board
        st.markdown(render_board_html(st.session_state.play_board, st.session_state.play_last_move), unsafe_allow_html=True)

    with col_diag:
        st.markdown("### 🧠 Agent Diagnostics")
        v = st.session_state.play_last_agent_val
        pi = st.session_state.play_last_agent_pi

        if v is not None and pi is not None:
            # Win probability estimate from value head (scaled from [-1, 1] to [0, 100%])
            win_prob = max(0.0, min(100.0, (v + 1.0) / 2.0 * 100.0))
            st.metric("Win Probability Estimate", f"{win_prob:.1f}%", delta=f"Value Head: {v:+.2f}")

            st.markdown("**Action Prior / Policy Distribution:**")
            policy_df = pd.DataFrame(
                {"Policy Prob (%)": [round(p * 100.0, 1) for p in pi]},
                index=[f"Col {i+1}" for i in range(7)],
            )
            st.bar_chart(policy_df, height=180)
        else:
            st.info("Make a move to view the agent's real-time neural network evaluation and policy distribution.")

        # Move list
        st.markdown("### 📋 Move History")
        if st.session_state.play_history:
            history_html = ['<div class="history-card">']
            for idx, (player_tag, col_num) in enumerate(st.session_state.play_history):
                is_latest = (idx == len(st.session_state.play_history) - 1)
                cls = "history-item latest" if is_latest else "history-item"
                history_html.append(f'<div class="{cls}"><span>#{idx+1} {player_tag}</span><span>Column {col_num}</span></div>')
            history_html.append("</div>")
            st.markdown("".join(history_html), unsafe_allow_html=True)
        else:
            st.caption("No moves played yet.")


# =============================================================================
# MODE 2: WATCH AGENT VS AGENT
# =============================================================================
elif mode == "⚔️ Watch Agent vs Agent":
    st.markdown('<div class="studio-header">Watch Agent vs Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="studio-sub">Pit two AI checkpoints against each other with real-time stepping or auto-play.</div>', unsafe_allow_html=True)

    st.sidebar.markdown("### 🤖 Player 1 (🔴 Red)")
    p1_label = st.sidebar.selectbox("Checkpoint P1", list(CHECKPOINT_CATALOG.keys()), index=0, key="p1_select")
    p1_sims = st.sidebar.select_slider("MCTS Sims P1", [5, 25, 50], value=25, key="p1_sims")

    st.sidebar.markdown("### 🤖 Player 2 (🟡 Yellow)")
    p2_label = st.sidebar.selectbox("Checkpoint P2", list(CHECKPOINT_CATALOG.keys()), index=3, key="p2_select")
    p2_sims = st.sidebar.select_slider("MCTS Sims P2", [5, 25, 50], value=25, key="p2_sims")

    st.sidebar.markdown("### ⚡ Playback Controls")
    auto_play = st.sidebar.toggle("Auto-Play", value=False)
    play_speed = st.sidebar.slider("Delay between moves (s)", 0.05, 1.5, 0.3, step=0.05)

    # State initialization
    if "watch_board" not in st.session_state:
        st.session_state.watch_board = game.getInitBoard()
        st.session_state.watch_cur_player = 1
        st.session_state.watch_game_over = False
        st.session_state.watch_winner = 0
        st.session_state.watch_last_move = None
        st.session_state.watch_moves = []
        st.session_state.watch_tally = {"p1_wins": 0, "p2_wins": 0, "draws": 0}

    def reset_watch_game():
        st.session_state.watch_board = game.getInitBoard()
        st.session_state.watch_cur_player = 1
        st.session_state.watch_game_over = False
        st.session_state.watch_winner = 0
        st.session_state.watch_last_move = None
        st.session_state.watch_moves = []

    def step_watch_move():
        if st.session_state.watch_game_over:
            return

        b = st.session_state.watch_board
        cur_p = st.session_state.watch_cur_player
        canonical = game.getCanonicalForm(b, cur_p)

        ckpt = CHECKPOINT_CATALOG[p1_label] if cur_p == 1 else CHECKPOINT_CATALOG[p2_label]
        sims = p1_sims if cur_p == 1 else p2_sims

        nnet = load_cached_nnet(ckpt)
        mcts = MCTS(game, nnet, dotdict({"numMCTSSims": sims, "cpuct": 1.0}))
        probs = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(probs))

        r = find_dropped_row(b, action)
        new_b, next_p = game.getNextState(b, cur_p, action)
        name = f"P1: {p1_label.split()[0]}" if cur_p == 1 else f"P2: {p2_label.split()[0]}"
        color_tag = "🔴 Red" if cur_p == 1 else "🟡 Yellow"

        st.session_state.watch_board = new_b
        st.session_state.watch_cur_player = next_p
        st.session_state.watch_last_move = (r, action)
        st.session_state.watch_moves.append((f"{name} ({color_tag})", action + 1))

        # Check end
        res = game.getGameEnded(new_b, 1)
        if res != 0:
            st.session_state.watch_game_over = True
            if res == 1:
                st.session_state.watch_winner = 1
                st.session_state.watch_tally["p1_wins"] += 1
            elif res == -1:
                st.session_state.watch_winner = -1
                st.session_state.watch_tally["p2_wins"] += 1
            else:
                st.session_state.watch_winner = 0
                st.session_state.watch_tally["draws"] += 1

    # Tally Header
    tally = st.session_state.watch_tally
    total_g = tally["p1_wins"] + tally["p2_wins"] + tally["draws"]
    t1, t2, t3, t4, t5 = st.columns([3, 3, 3, 3, 2])
    t1.metric("🔴 P1 Wins", tally["p1_wins"])
    t2.metric("🟡 P2 Wins", tally["p2_wins"])
    t3.metric("🤝 Draws", tally["draws"])
    t4.metric("📊 Total Games", total_g)
    with t5:
        st.write("")
        if st.button("Reset Tally"):
            st.session_state.watch_tally = {"p1_wins": 0, "p2_wins": 0, "draws": 0}
            reset_watch_game()
            st.rerun()

    st.markdown("---")

    col_board, col_history = st.columns([7, 5])

    with col_board:
        if st.session_state.watch_game_over:
            w = st.session_state.watch_winner
            if w == 1:
                st.success(f"🏆 **Winner: Player 1 (🔴 Red — {p1_label})!**")
            elif w == -1:
                st.warning(f"🏆 **Winner: Player 2 (🟡 Yellow — {p2_label})!**")
            else:
                st.info("🤝 **Game drawn.**")
        else:
            turn_label = f"🔴 Player 1 ({p1_label.split()[0]})" if st.session_state.watch_cur_player == 1 else f"🟡 Player 2 ({p2_label.split()[0]})"
            st.info(f"**Current Turn:** {turn_label}")

        # Controls
        c_next, c_new, _ = st.columns([3, 3, 4])
        with c_next:
            if st.button("▶ Next Move", disabled=st.session_state.watch_game_over or auto_play, use_container_width=True):
                step_watch_move()
                st.rerun()
        with c_new:
            if st.button("🔄 New Game", use_container_width=True):
                reset_watch_game()
                st.rerun()

        st.markdown(render_board_html(st.session_state.watch_board, st.session_state.watch_last_move), unsafe_allow_html=True)

    with col_history:
        st.markdown("### 📋 Live Move Feed")
        if st.session_state.watch_moves:
            history_html = ['<div class="history-card">']
            for idx, (p_tag, col_num) in enumerate(st.session_state.watch_moves):
                is_latest = (idx == len(st.session_state.watch_moves) - 1)
                cls = "history-item latest" if is_latest else "history-item"
                history_html.append(f'<div class="{cls}"><span>#{idx+1} {p_tag}</span><span>Column {col_num}</span></div>')
            history_html.append("</div>")
            st.markdown("".join(history_html), unsafe_allow_html=True)
        else:
            st.caption("No moves yet. Click 'Next Move' or enable 'Auto-Play'.")

    # Auto-play execution loop
    if auto_play and not st.session_state.watch_game_over:
        time.sleep(play_speed)
        step_watch_move()
        st.rerun()


# =============================================================================
# MODE 3: REPLAY TRANSCRIPT
# =============================================================================
elif mode == "📜 Replay Transcript":
    st.markdown('<div class="studio-header">Replay Match Transcript</div>', unsafe_allow_html=True)
    st.markdown('<div class="studio-sub">Load recorded matches from <code>src/results/report/</code> and inspect every ply with board visualization.</div>', unsafe_allow_html=True)

    report_dir = os.path.join(_ROOT, "src", "results", "report")
    transcript_files = sorted(glob.glob(os.path.join(report_dir, "*.txt")))

    if not transcript_files:
        st.error(f"No transcript files (.txt) found in `{report_dir}`.")
    else:
        file_options = {os.path.basename(f): f for f in transcript_files}
        chosen_file = st.sidebar.selectbox("Select Transcript File", list(file_options.keys()), index=0)
        file_path = file_options[chosen_file]

        title, outcome, steps = parse_transcript_file(file_path)

        st.sidebar.markdown("### 📄 Metadata")
        st.sidebar.markdown(f"**Match Title:** {title}")
        st.sidebar.markdown(f"**Outcome:** `{outcome or 'Completed'}`")
        st.sidebar.markdown(f"**Total Steps:** {len(steps)-1}")

        # Replay index in session state
        step_key = f"replay_step_{chosen_file}"
        if step_key not in st.session_state:
            st.session_state[step_key] = 0

        total_steps = len(steps) - 1

        # Playback navigation bar
        nav_start, nav_prev, nav_next, nav_end = st.columns(4)
        with nav_start:
            if st.button("⏮ Start", use_container_width=True):
                st.session_state[step_key] = 0
                st.rerun()
        with nav_prev:
            if st.button("◀ Previous", use_container_width=True):
                st.session_state[step_key] = max(0, st.session_state[step_key] - 1)
                st.rerun()
        with nav_next:
            if st.button("Next ▶", use_container_width=True):
                st.session_state[step_key] = min(total_steps, st.session_state[step_key] + 1)
                st.rerun()
        with nav_end:
            if st.button("⏭ End", use_container_width=True):
                st.session_state[step_key] = total_steps
                st.rerun()

        # Step slider
        slider_val = st.slider(
            "Jump to step",
            0,
            total_steps,
            value=st.session_state[step_key],
            key=f"slider_{chosen_file}",
        )
        if slider_val != st.session_state[step_key]:
            st.session_state[step_key] = slider_val
            st.rerun()

        cur_step_idx = st.session_state[step_key]
        cur_step = steps[cur_step_idx]

        st.markdown(f"### 📍 Step {cur_step_idx} of {total_steps}: {cur_step['info']}")

        # Layout: Board on left, timeline on right
        col_board, col_timeline = st.columns([7, 5])

        with col_board:
            # Highlight column of current step
            last_move_coord = None
            if cur_step["col"] is not None:
                # Find disc location
                for r in range(6):
                    if cur_step["board"][r, cur_step["col"]] != 0:
                        last_move_coord = (r, cur_step["col"])
                        break

            st.markdown(render_board_html(cur_step["board"], last_move_coord), unsafe_allow_html=True)

        with col_timeline:
            st.markdown("### 📜 Match Timeline")
            t_html = ['<div class="history-card">']
            for s_idx, s in enumerate(steps):
                is_active = (s_idx == cur_step_idx)
                cls = "history-item latest" if is_active else "history-item"
                t_html.append(f'<div class="{cls}"><span>Step {s_idx}</span><span>{s["info"]}</span></div>')
            t_html.append("</div>")
            st.markdown("".join(t_html), unsafe_allow_html=True)
