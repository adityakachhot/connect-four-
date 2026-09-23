"""
Connect Four — alpha-zero-general Game subclass.

Board layout
------------
  Rows 0..ROWS-1, columns 0..COLS-1.
  Row 0 is the TOP of the visual board; row ROWS-1 is the bottom.
  Gravity: a piece dropped into column c lands in the highest-indexed
  (lowest on screen) empty cell in that column.

Board encoding
--------------
  numpy int8 array of shape (ROWS, COLS).
    1  → piece belonging to player 1
   -1  → piece belonging to player 2
    0  → empty cell

Action space
------------
  Actions 0..COLS-1 each represent dropping a piece into that column.
  Action COLS is a "no-op / pass" included so the framework can handle
  the rare fully-drawn position without ambiguity; it is only valid when
  all columns are full (i.e. immediately before detecting a draw).
  In practice the game ends at the draw check, so action COLS never
  actually gets executed.

Canonical form
--------------
  board * player — flips the sign of every cell when it is player -1's
  turn, so the NN always sees its own pieces as +1.

Symmetry
--------
  A horizontal flip (left-right column mirror) is the only non-trivial
  symmetry of Connect Four.  Both the board and the corresponding policy
  vector (column probabilities) are flipped.
"""

import sys
import os
import numpy as np

# Allow importing Game from reference-repo without modifying it
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference-repo"))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from Game import Game  # noqa: E402  (reference-repo/Game.py)

ROWS = 6
COLS = 7
WIN_LEN = 4  # pieces in a row needed to win


class ConnectFourGame(Game):
    """
    Connect Four implementation of the alpha-zero-general Game interface.

    Parameters
    ----------
    rows : int
        Number of rows (default 6).
    cols : int
        Number of columns (default 7).
    win_length : int
        Number of consecutive pieces required to win (default 4).
    """

    def __init__(self, rows: int = ROWS, cols: int = COLS, win_length: int = WIN_LEN):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.win_length = win_length

    # ------------------------------------------------------------------ #
    #  Game interface                                                       #
    # ------------------------------------------------------------------ #

    def getInitBoard(self) -> np.ndarray:
        """Return an empty rows×cols board (all zeros)."""
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def getBoardSize(self):
        """Return (rows, cols)."""
        return (self.rows, self.cols)

    def getActionSize(self) -> int:
        """
        cols + 1 actions: one per column (0..cols-1) plus one pass slot
        (cols) that is used only to signal a completed draw so that the
        framework's action-vector is never zero-length on a terminal board.
        """
        return self.cols + 1

    def getNextState(self, board: np.ndarray, player: int, action: int):
        """
        Drop player's piece into column `action`.

        Returns
        -------
        (next_board, next_player) where next_player == -player.

        Notes
        -----
        The caller is responsible for only passing valid actions (i.e. actions
        where the column is not full).  Passing a full column raises ValueError.
        """
        if action == self.cols:
            # Pass action — board unchanged, turn changes
            return (board.copy(), -player)

        col = action
        # Find the lowest empty row in this column (highest row index)
        empty_rows = np.where(board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            raise ValueError(f"Column {col} is full — invalid action.")
        row = empty_rows[-1]  # lowest empty row (gravity)

        next_board = board.copy()
        next_board[row, col] = player
        return (next_board, -player)

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return a binary vector of length getActionSize().
        Column c is valid iff its top cell (row 0) is empty.
        The pass action (index cols) is always invalid during normal play;
        it becomes valid only when every column is full (draw).
        """
        valid = np.zeros(self.getActionSize(), dtype=np.int8)
        for col in range(self.cols):
            if board[0, col] == 0:   # top row is empty → column not full
                valid[col] = 1

        if valid[:self.cols].sum() == 0:
            # Every column is full — allow the pass slot so MCTS has one action
            valid[self.cols] = 1

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
        # Check if player 1 has won
        if self._has_won(board, 1):
            return 1 if player == 1 else -1
        # Check if player -1 has won
        if self._has_won(board, -1):
            return 1 if player == -1 else -1
        # Draw: board is completely full
        if np.all(board != 0):
            return 1e-4
        return 0

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return board * player so the current player's pieces are always +1.
        """
        return (board * player).astype(np.int8)

    def getSymmetries(self, board: np.ndarray, pi):
        """
        Return [(board, pi)] pairs for the original and the horizontal flip.

        Horizontal flip: column c maps to column (cols-1-c), which reverses
        the policy vector's first `cols` entries (the pass slot stays put).
        """
        pi = list(pi)
        # Original
        original = (board.copy(), pi[:])

        # Horizontal flip
        flipped_board = np.fliplr(board)
        # Flip the column probabilities; preserve the pass-action slot
        flipped_pi = pi[:self.cols][::-1] + [pi[self.cols]]
        flipped = (flipped_board, flipped_pi)

        return [original, flipped]

    def stringRepresentation(self, board: np.ndarray) -> bytes:
        """
        Return a compact, hashable bytes representation of the board.
        Uses tobytes() (NumPy ≥ 1.9) for speed and uniqueness.
        """
        return board.tobytes()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _has_won(self, board: np.ndarray, player: int) -> bool:
        """
        Return True if `player` has win_length pieces in a row anywhere
        on the board (horizontal, vertical, or either diagonal).
        """
        target = np.full(self.win_length, player, dtype=np.int8)

        # --- Horizontal ---
        for r in range(self.rows):
            for c in range(self.cols - self.win_length + 1):
                if np.array_equal(board[r, c:c + self.win_length], target):
                    return True

        # --- Vertical ---
        for c in range(self.cols):
            for r in range(self.rows - self.win_length + 1):
                if np.array_equal(board[r:r + self.win_length, c], target):
                    return True

        # --- Diagonal (top-left → bottom-right) ---
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols - self.win_length + 1):
                segment = np.array([board[r + i, c + i] for i in range(self.win_length)], dtype=np.int8)
                if np.array_equal(segment, target):
                    return True

        # --- Diagonal (top-right → bottom-left) ---
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.win_length - 1, self.cols):
                segment = np.array([board[r + i, c - i] for i in range(self.win_length)], dtype=np.int8)
                if np.array_equal(segment, target):
                    return True

        return False

    # ------------------------------------------------------------------ #
    #  Display utility (not required by the interface, handy for debugging) #
    # ------------------------------------------------------------------ #

    def display(self, board: np.ndarray) -> None:
        symbols = {1: "X", -1: "O", 0: "."}
        col_header = " ".join(str(c) for c in range(self.cols))
        print(f"  {col_header}")
        print("  " + "-" * (self.cols * 2 - 1))
        for r in range(self.rows):
            row_str = " ".join(symbols[int(board[r, c])] for c in range(self.cols))
            print(f"  {row_str}")
        print()
