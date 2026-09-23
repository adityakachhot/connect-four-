"""
game_runner.py — independent two-agent game runner for ConnectFourGame.

Does NOT use reference-repo/Arena.py.  Implements its own:
  - RandomAgent       : picks uniformly from valid columns
  - NNetGreedyAgent   : single forward-pass -> masked softmax -> argmax
  - play_game()       : complete game loop, returns result from player-1's view

Result convention (matches getGameEnded):
   1   player 1 won
  -1   player 1 lost
   0   draw  (we map the 1e-4 sentinel to 0 here for legibility)
"""

from __future__ import annotations

import random
import numpy as np


# --- Agent interface --------------------------------------------------------

class Agent:
    """Base agent: given a canonical board, return an action index."""
    def select_action(self, canonical_board: np.ndarray, valid_moves: np.ndarray) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """Selects uniformly at random from the set of valid moves."""

    def select_action(self, canonical_board: np.ndarray, valid_moves: np.ndarray) -> int:
        valid_indices = np.where(valid_moves == 1)[0]
        return int(random.choice(valid_indices))


class NNetGreedyAgent(Agent):
    """
    Uses a loaded NeuralNet to score moves (single forward pass, no MCTS).

    Policy is masked to valid moves, then re-normalised, then argmax is taken.
    Much faster than MCTS; sufficient for benchmarking per-checkpoint progress.
    """

    def __init__(self, nnet):
        self.nnet = nnet

    def select_action(self, canonical_board: np.ndarray, valid_moves: np.ndarray) -> int:
        pi, _ = self.nnet.predict(canonical_board)

        # Mask invalid moves and re-normalise
        masked = pi * valid_moves.astype(np.float32)
        total  = masked.sum()
        if total < 1e-8:
            # Fallback: uniform over valid moves
            masked = valid_moves.astype(np.float32)
            total  = masked.sum()
        masked /= total

        return int(np.argmax(masked))


# --- Game loop --------------------------------------------------------------

def play_game(game, agent1: Agent, agent2: Agent) -> int:
    """
    Play one complete game.

    agent1 plays as player  1.
    agent2 plays as player -1.

    Returns
    -------
     1   agent1 won
    -1   agent2 won
     0   draw
    """
    board  = game.getInitBoard()
    player = 1
    agents = {1: agent1, -1: agent2}

    while True:
        canonical = game.getCanonicalForm(board, player)
        valid     = game.getValidMoves(canonical, 1)

        action = agents[player].select_action(canonical, valid)
        board, player = game.getNextState(board, player, action)

        result = game.getGameEnded(board, 1)   # always from player-1 POV
        if result != 0:
            if abs(result) < 0.5:              # 1e-4 draw sentinel -> 0
                return 0
            return int(np.sign(result))        # +1 or -1


def play_match(game, agent_a: Agent, agent_b: Agent, n_games: int = 30) -> dict:
    """
    Play n_games total, splitting evenly so each agent goes first half the time.

    Returns dict: wins_a, wins_b, draws, win_rate_a, total
    """
    wins_a = wins_b = draws = 0
    half = n_games // 2

    # agent_a as player 1
    for _ in range(half):
        r = play_game(game, agent_a, agent_b)
        if   r ==  1: wins_a += 1
        elif r == -1: wins_b += 1
        else:         draws  += 1

    # agent_a as player 2
    for _ in range(n_games - half):
        r = play_game(game, agent_b, agent_a)
        if   r ==  1: wins_b += 1
        elif r == -1: wins_a += 1
        else:         draws  += 1

    decisive   = wins_a + wins_b
    win_rate_a = wins_a / decisive if decisive > 0 else 0.5

    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws,
            "win_rate_a": win_rate_a, "total": n_games}
