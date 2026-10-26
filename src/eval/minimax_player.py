"""
minimax_player.py — Minimax agent with alpha-beta pruning for grid games.

Supports both column-based games (Connect Four) and cell-based games
(Tic-Tac-Toe) — uses the Game interface generically.

Uses *only* the ConnectFourGame interface (getValidMoves, getNextState,
getGameEnded, getCanonicalForm).  No neural network, no MCTS — a pure
classical game-tree search with a hand-crafted heuristic evaluation function.

The heuristic evaluates board positions by counting windows of `win_length`
cells across all rows, columns, and diagonals, scoring each window according
to how many pieces each player has in it, with a strong bonus for center
column occupation.

The agent implements the Agent interface from game_runner.py so it can be
used directly with `play_match()` and `play_game()`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.eval.game_runner import Agent


class MinimaxAgent(Agent):
    """
    Alpha-beta pruning minimax player for grid-based two-player games.

    Works with any game that exposes rows, cols, win_length attributes
    and implements the alpha-zero-general Game interface.

    Parameters
    ----------
    game : Game
        Game instance (used for rules only, never mutated).
    depth : int
        Maximum search depth (half-moves / plies).  Typical values:
          2 — very weak, fast  (~instant)
          4 — moderate, still fast
          6 — reasonably strong, a few seconds per move
          9 — perfect play for Tic-Tac-Toe
    """

    def __init__(self, game, depth: int = 4):
        self.game = game
        self.depth = depth
        self.rows = game.rows
        self.cols = game.cols
        self.win_length = game.win_length
        # Number of real actions (excluding pass); works for both
        # column-based (Connect Four: cols) and cell-based (TTT: rows*cols)
        self.n_actions = game.getActionSize() - 1

    # ------------------------------------------------------------------ #
    #  Agent interface                                                      #
    # ------------------------------------------------------------------ #

    def select_action(
        self, canonical_board: np.ndarray, valid_moves: np.ndarray
    ) -> int:
        """
        Given a canonical board (current player's pieces = +1) and a
        valid-move mask, return the best action according to minimax.

        The canonical board convention means the maximising player is
        always +1 at the root.
        """
        best_score = -math.inf
        best_action = -1
        valid_actions = np.where(valid_moves[:self.n_actions] == 1)[0]

        # Move ordering: centre-biased for better pruning
        centre = self.n_actions // 2
        valid_actions = sorted(valid_actions, key=lambda a: abs(a - centre))

        for action in valid_actions:
            # Apply move for player +1 (the canonical current player)
            next_board, _ = self.game.getNextState(canonical_board, 1, action)
            # Recurse as minimiser (opponent = player -1 on the canonical board)
            score = self._minimax(
                next_board,
                depth=self.depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                maximising=False,
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ------------------------------------------------------------------ #
    #  Minimax with alpha-beta pruning                                      #
    # ------------------------------------------------------------------ #

    def _minimax(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximising: bool,
    ) -> float:
        """
        Standard alpha-beta minimax.

        `board` is always in "raw" form (player +1 pieces = +1,
        player -1 pieces = -1) because getNextState preserves that.

        `maximising=True` ⇔ it is player +1's turn.
        """
        # --- Terminal / depth-0 check ---
        # getGameEnded returns result from the specified player's POV
        player = 1 if maximising else -1
        ended = self.game.getGameEnded(board, player)

        if ended != 0:
            if abs(ended) < 0.5:          # draw sentinel (1e-4)
                return 0.0
            # ended == 1  ⇒ `player` won
            # ended == -1 ⇒ `player` lost
            if maximising:
                return ended * 1_000_000  # +1M if player 1 won, -1M if lost
            else:
                # When minimising, `player` is -1.  ended=1 means -1 won
                # (bad for maximiser), ended=-1 means +1 won (good for max).
                return -ended * 1_000_000

        if depth == 0:
            return self._evaluate(board)

        # --- Determine valid columns ---
        # Use raw board + current player to get valid moves
        valid = self.game.getValidMoves(board, player)
        valid_actions = np.where(valid[:self.n_actions] == 1)[0]

        # Move ordering: centre-biased for better pruning
        centre = self.n_actions // 2
        valid_actions = sorted(valid_actions, key=lambda a: abs(a - centre))

        if maximising:
            value = -math.inf
            for action in valid_actions:
                next_board, _ = self.game.getNextState(board, 1, action)
                value = max(value, self._minimax(next_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # β cut-off
            return value
        else:
            value = math.inf
            for action in valid_actions:
                next_board, _ = self.game.getNextState(board, -1, action)
                value = min(value, self._minimax(next_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # α cut-off
            return value

    # ------------------------------------------------------------------ #
    #  Static evaluation heuristic                                          #
    # ------------------------------------------------------------------ #

    def _evaluate(self, board: np.ndarray) -> float:
        """
        Heuristic evaluation from player +1's perspective.

        Components
        ----------
        1. Window scoring: for every window of `win_length` cells in every
           direction, score based on piece counts.
        2. Centre column bonus: pieces in the centre column get extra weight
           because they participate in more potential four-in-a-rows.
        """
        score = 0.0
        wl = self.win_length

        # --- Centre column bonus ---
        centre_col = self.cols // 2
        centre_array = board[:, centre_col]
        score += 3 * int(np.sum(centre_array == 1))
        score -= 3 * int(np.sum(centre_array == -1))

        # --- Window scoring (horizontal, vertical, both diagonals) ---
        # Horizontal windows
        for r in range(self.rows):
            for c in range(self.cols - wl + 1):
                window = board[r, c : c + wl]
                score += self._score_window(window)

        # Vertical windows
        for c in range(self.cols):
            for r in range(self.rows - wl + 1):
                window = board[r : r + wl, c]
                score += self._score_window(window)

        # Positive diagonal (top-left → bottom-right)
        for r in range(self.rows - wl + 1):
            for c in range(self.cols - wl + 1):
                window = np.array(
                    [board[r + i, c + i] for i in range(wl)], dtype=np.int8
                )
                score += self._score_window(window)

        # Negative diagonal (top-right → bottom-left)
        for r in range(self.rows - wl + 1):
            for c in range(wl - 1, self.cols):
                window = np.array(
                    [board[r + i, c - i] for i in range(wl)], dtype=np.int8
                )
                score += self._score_window(window)

        return score

    @staticmethod
    def _score_window(window: np.ndarray) -> float:
        """
        Score a single window of length `win_length`.

        Heuristic (player +1 perspective):
          4 in a row   → +100  (shouldn't reach here; terminal detected earlier)
          3 + 1 empty  → +5
          2 + 2 empty  → +2
          Opponent:
          opp 3 + 1 empty → -4  (urgent threat — almost as important as own 3)
        """
        n_plus = int(np.sum(window == 1))
        n_minus = int(np.sum(window == -1))
        n_empty = int(np.sum(window == 0))

        # If both players have pieces in the window, it's blocked — worth nothing
        if n_plus > 0 and n_minus > 0:
            return 0.0

        wl = len(window)

        # Player +1 scoring
        if n_plus == wl:
            return 100.0
        if n_plus == wl - 1 and n_empty == 1:
            return 5.0
        if n_plus == wl - 2 and n_empty == 2:
            return 2.0

        # Player -1 scoring (threats against maximiser)
        if n_minus == wl:
            return -100.0
        if n_minus == wl - 1 and n_empty == 1:
            return -4.0

        return 0.0

    def __repr__(self) -> str:
        return f"MinimaxAgent(depth={self.depth})"
