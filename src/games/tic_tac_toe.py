"""
Tic-Tac-Toe — alpha-zero-general Game subclass.

Board layout
------------
  3×3 grid, indexed by (row, col) with row 0 at the top.
  No gravity — pieces are placed directly at the chosen cell.

Board encoding
--------------
  numpy int8 array of shape (3, 3).
    1  → piece belonging to player 1 (X)
   -1  → piece belonging to player 2 (O)
    0  → empty cell

Action space
------------
  Actions 0..8 represent placing a piece at row = action // 3,
  col = action % 3.  Action 9 is a pass/no-op that is only valid
  when the board is completely full (draw position), matching the
  convention used in ConnectFourGame.

Canonical form
--------------
  board * player — flips the sign of every cell when it is player -1's
  turn, so the NN always sees its own pieces as +1.

Symmetry
--------
  The dihedral group D4 of the square: 4 rotations × 2 reflections = 8
  symmetries.  Both the board and the policy vector are transformed
  accordingly.
"""

import sys
import os
import numpy as np

# Allow importing Game from reference-repo without modifying it
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference-repo"))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from Game import Game  # noqa: E402  (reference-repo/Game.py)

ROWS = 3
COLS = 3
WIN_LEN = 3


class TicTacToeGame(Game):
    """
    Tic-Tac-Toe implementation of the alpha-zero-general Game interface.

    Attributes (exposed for compatibility with MinimaxAgent):
        rows, cols, win_length
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS, win_length: int = WIN_LEN):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.n_cells = rows * cols  # 9

    # ------------------------------------------------------------------ #
    #  Game interface                                                       #
    # ------------------------------------------------------------------ #

    def getInitBoard(self) -> np.ndarray:
        """Return an empty 3×3 board (all zeros)."""
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def getBoardSize(self):
        """Return (rows, cols)."""
        return (self.rows, self.cols)

    def getActionSize(self) -> int:
        """
        n_cells + 1 actions: one per cell (0..8) plus one pass slot (9)
        that is only valid when the board is full (draw).
        """
        return self.n_cells + 1

    def getNextState(self, board: np.ndarray, player: int, action: int):
        """
        Place player's piece at cell `action`.

        action = row * cols + col   for cells 0..8
        action = n_cells            for the pass action

        Returns (next_board, -player).
        """
        if action == self.n_cells:
            # Pass action — board unchanged, turn changes
            return (board.copy(), -player)

        r = action // self.cols
        c = action % self.cols

        if board[r, c] != 0:
            raise ValueError(f"Cell ({r},{c}) is already occupied — invalid action {action}.")

        next_board = board.copy()
        next_board[r, c] = player
        return (next_board, -player)

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return a binary vector of length getActionSize().
        Cell i is valid iff it is empty.
        The pass action (index n_cells) is valid only when all cells are filled.
        """
        valid = np.zeros(self.getActionSize(), dtype=np.int8)
        for i in range(self.n_cells):
            r = i // self.cols
            c = i % self.cols
            if board[r, c] == 0:
                valid[i] = 1

        if valid[:self.n_cells].sum() == 0:
            # Board is full — allow pass
            valid[self.n_cells] = 1

        return valid

    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        """
        Returns
        -------
        0      : game is not over
        1      : `player` has won
        -1     : `player` has lost (the opponent won)
        1e-4   : draw (board full, no winner)
        """
        if self._has_won(board, 1):
            return 1 if player == 1 else -1
        if self._has_won(board, -1):
            return 1 if player == -1 else -1
        if np.all(board != 0):
            return 1e-4
        return 0

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """Return board * player so the current player's pieces are always +1."""
        return (board * player).astype(np.int8)

    def getSymmetries(self, board: np.ndarray, pi):
        """
        Return all 8 symmetries of the square (D4 dihedral group):
        4 rotations × {identity, reflection}.

        The policy vector pi has n_cells + 1 entries.  The first n_cells
        entries correspond to cells and are permuted; the pass action
        (last entry) stays fixed.
        """
        pi = list(pi)
        symmetries = []

        # Convert pi's cell entries to a 3×3 grid for transformation
        pi_board = np.array(pi[:self.n_cells]).reshape(self.rows, self.cols)
        pass_prob = pi[self.n_cells]

        for rotation in range(4):
            # Rotate board and pi_board by 90° * rotation
            rot_board = np.rot90(board, k=rotation)
            rot_pi    = np.rot90(pi_board, k=rotation)
            # Flatten pi back to vector form
            rot_pi_vec = list(rot_pi.flatten()) + [pass_prob]
            symmetries.append((rot_board.copy(), rot_pi_vec[:]))

            # Also add the horizontal flip of this rotation
            flip_board = np.fliplr(rot_board)
            flip_pi    = np.fliplr(rot_pi)
            flip_pi_vec = list(flip_pi.flatten()) + [pass_prob]
            symmetries.append((flip_board.copy(), flip_pi_vec[:]))

        return symmetries

    def stringRepresentation(self, board: np.ndarray) -> bytes:
        """Return a compact, hashable bytes representation of the board."""
        return board.tobytes()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _has_won(self, board: np.ndarray, player: int) -> bool:
        """
        Return True if `player` has win_length pieces in a row anywhere
        on the board (horizontal, vertical, or either diagonal).
        """
        wl = self.win_length

        # --- Horizontal ---
        for r in range(self.rows):
            for c in range(self.cols - wl + 1):
                if all(board[r, c + i] == player for i in range(wl)):
                    return True

        # --- Vertical ---
        for c in range(self.cols):
            for r in range(self.rows - wl + 1):
                if all(board[r + i, c] == player for i in range(wl)):
                    return True

        # --- Diagonal (top-left → bottom-right) ---
        for r in range(self.rows - wl + 1):
            for c in range(self.cols - wl + 1):
                if all(board[r + i, c + i] == player for i in range(wl)):
                    return True

        # --- Diagonal (top-right → bottom-left) ---
        for r in range(self.rows - wl + 1):
            for c in range(wl - 1, self.cols):
                if all(board[r + i, c - i] == player for i in range(wl)):
                    return True

        return False

    # ------------------------------------------------------------------ #
    #  Display utility                                                      #
    # ------------------------------------------------------------------ #

    def display(self, board: np.ndarray) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        print("  " + " ".join(str(c) for c in range(self.cols)))
        print("  " + "-" * (self.cols * 2 - 1))
        for r in range(self.rows):
            row_str = " ".join(symbols[int(board[r, c])] for c in range(self.cols))
            print(f"  {row_str}")
        print()
