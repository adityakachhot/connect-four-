"""
NeuralNet implementation for ConnectFourGame.

Two classes live here:

  ConnectFourCNN   — raw torch.nn.Module, purely architectural.
  ConnectFourNNet  — alpha-zero-general NeuralNet adapter that wraps the above,
                     handles device placement, training loop, and checkpointing.

Architecture
------------
The CNN treats the board as a single-channel 2-D image (H × W = 6 × 7).
Four convolutional blocks (Conv2d + BatchNorm2d + ReLU) with padding=1 preserve
spatial dimensions throughout, so the design is board-size agnostic.  A shared
trunk then feeds two separate heads:

  Policy head → log_softmax over all actions (for use with NLLLoss)
  Value head  → tanh scalar ∈ [−1, 1]

Device priority: MPS (Apple Silicon GPU) → CPU.
CUDA is intentionally not included here; add it later if needed.
"""

from __future__ import annotations

import os
import sys
import math
import random
import dataclasses
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── resolve reference-repo on sys.path without modifying it ────────────────
_REPO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reference-repo")
)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from NeuralNet import NeuralNet  # reference-repo/NeuralNet.py


# ============================================================
# Device helper
# ============================================================

def _select_device() -> torch.device:
    """Return MPS if available on Apple Silicon, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Hyper-parameters (independent of game dimensions)
# ============================================================

@dataclasses.dataclass
class NNetArgs:
    """Training and architecture hyper-parameters."""
    # Architecture
    num_channels: int = 128     # conv filter count in every block
    dropout:      float = 0.3   # dropout probability in the trunk FC layers

    # Training
    epochs:       int   = 10    # training epochs per Coach iteration
    batch_size:   int   = 64    # mini-batch size
    lr:           float = 1e-3  # Adam learning rate
    weight_decay: float = 1e-4  # L2 regularisation


# ============================================================
# Raw torch.nn.Module — architecture only
# ============================================================

class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU, padding=1 to preserve spatial size."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConnectFourCNN(nn.Module):
    """
    Convolutional backbone + dual-head for Connect Four.

    Parameters
    ----------
    rows, cols    : board dimensions (default 6 × 7)
    action_size   : length of the policy vector (cols + 1 for the pass action)
    args          : NNetArgs instance with channel count and dropout
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        action_size: int,
        args: NNetArgs,
    ):
        super().__init__()
        C = args.num_channels

        # ── Four convolutional blocks (all padding=1 → spatial dims unchanged) ──
        self.conv_trunk = nn.Sequential(
            _ConvBlock(1,  C),   # input: (B, 1, rows, cols)
            _ConvBlock(C,  C),
            _ConvBlock(C,  C),
            _ConvBlock(C,  C),   # output: (B, C, rows, cols)
        )

        # ── Shared fully-connected trunk ──────────────────────────────────────
        flat_size = C * rows * cols
        self.fc_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
        )

        # ── Policy head ───────────────────────────────────────────────────────
        # Outputs log-probabilities; use NLLLoss for training, exp() for inference.
        self.policy_head = nn.Linear(128, action_size)

        # ── Value head ────────────────────────────────────────────────────────
        self.value_head = nn.Linear(128, 1)

        # Weight initialisation (Kaiming for ReLU activations)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, rows, cols) float tensor — batch of canonical boards

        Returns
        -------
        log_pi : (B, action_size) — log-softmax policy
        v      : (B, 1)           — tanh value
        """
        # Add channel dimension: (B, rows, cols) → (B, 1, rows, cols)
        x = x.unsqueeze(1)

        features = self.conv_trunk(x)        # (B, C, rows, cols)
        shared   = self.fc_trunk(features)   # (B, 128)

        log_pi = F.log_softmax(self.policy_head(shared), dim=-1)  # (B, action_size)
        v      = torch.tanh(self.value_head(shared))               # (B, 1)

        return log_pi, v


# ============================================================
# NeuralNet adapter — what Coach and MCTS interact with
# ============================================================

class ConnectFourNNet(NeuralNet):
    """
    alpha-zero-general NeuralNet adapter for ConnectFourCNN.

    Usage
    -----
    game = ConnectFourGame()
    nnet = ConnectFourNNet(game)
    nnet.train(examples)          # list of (board, pi, v)
    pi, v = nnet.predict(board)   # board is a canonical numpy array
    nnet.save_checkpoint('ckpt/', 'best.pth.tar')
    nnet.load_checkpoint('ckpt/', 'best.pth.tar')
    """

    def __init__(self, game, args: NNetArgs | None = None):
        super().__init__(game)
        self.rows, self.cols = game.getBoardSize()
        self.action_size      = game.getActionSize()
        self.args             = args or NNetArgs()
        self.device           = _select_device()

        self.model = ConnectFourCNN(
            self.rows, self.cols, self.action_size, self.args
        ).to(self.device)

    # ------------------------------------------------------------------ #
    #  NeuralNet interface                                                  #
    # ------------------------------------------------------------------ #

    def train(self, examples: List[Tuple]) -> None:
        """
        Train on a list of (board, pi, v) examples from self-play.

        Each board is a canonical numpy int8 array of shape (rows, cols).
        pi  is a list/array of length action_size (MCTS-improved policy).
        v   is a float in {-1, 0, 1} (game outcome from this player's view).
        """
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        # Build tensors once — avoids repeated allocation inside the epoch loop
        boards_np = np.array([e[0] for e in examples], dtype=np.float32)
        pis_np    = np.array([e[1] for e in examples], dtype=np.float32)
        vs_np     = np.array([e[2] for e in examples], dtype=np.float32)

        dataset    = TensorDataset(
            torch.from_numpy(boards_np),
            torch.from_numpy(pis_np),
            torch.from_numpy(vs_np),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            total_pi_loss = 0.0
            total_v_loss  = 0.0
            n_batches     = 0

            for boards_b, pi_b, v_b in loader:
                boards_b = boards_b.to(self.device)
                pi_b     = pi_b.to(self.device)
                v_b      = v_b.to(self.device)

                log_pi, v_pred = self.model(boards_b)

                # Policy loss: negative log-likelihood (target is a soft label)
                # = -sum(pi * log_pi)  averaged over batch
                pi_loss = -(pi_b * log_pi).sum(dim=-1).mean()

                # Value loss: mean squared error
                v_loss  = F.mse_loss(v_pred.squeeze(-1), v_b)

                loss = pi_loss + v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_pi_loss += pi_loss.item()
                total_v_loss  += v_loss.item()
                n_batches     += 1

            if n_batches:
                avg_pi = total_pi_loss / n_batches
                avg_v  = total_v_loss  / n_batches
                print(
                    f"  Epoch {epoch:>2}/{self.args.epochs}  "
                    f"pi_loss={avg_pi:.4f}  v_loss={avg_v:.4f}"
                )

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run a single-board inference pass.

        Parameters
        ----------
        board : canonical numpy array of shape (rows, cols)

        Returns
        -------
        pi : numpy float array of shape (action_size,)  — probabilities (not log)
        v  : float scalar ∈ [−1, 1]
        """
        board_t = (
            torch.from_numpy(board.astype(np.float32))
            .unsqueeze(0)          # (1, rows, cols)
            .to(self.device)
        )

        self.model.eval()
        with torch.no_grad():
            log_pi, v = self.model(board_t)

        pi = torch.exp(log_pi).squeeze(0).cpu().numpy()   # (action_size,)
        v  = v.squeeze().cpu().item()                      # scalar float
        return pi, v

    def save_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        """Save model weights and args to folder/filename."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "args":       dataclasses.asdict(self.args),
                "board_size": (self.rows, self.cols),
                "action_size": self.action_size,
            },
            path,
        )
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        """Load model weights from folder/filename."""
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at {path!r}")
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        print(f"Checkpoint loaded ← {path}  (device={self.device})")

    # ------------------------------------------------------------------ #
    #  Convenience                                                          #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return (
            f"ConnectFourNNet("
            f"board={self.rows}×{self.cols}, "
            f"actions={self.action_size}, "
            f"params={n_params:,}, "
            f"device={self.device})"
        )
