"""
NeuralNet implementation for TicTacToeGame.

Architecture is identical in structure to ConnectFourNNet but scaled down
for the 3×3 board: fewer convolutional channels and smaller FC layers.

Device priority: MPS (Apple Silicon GPU) → CPU.
"""

from __future__ import annotations

import os
import sys
import dataclasses
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── resolve reference-repo on sys.path ─────────────────────────────────────
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
# Hyper-parameters
# ============================================================

@dataclasses.dataclass
class TTTNNetArgs:
    """Training and architecture hyper-parameters for Tic-Tac-Toe."""
    num_channels: int = 64      # conv filter count (smaller than C4's 128)
    dropout:      float = 0.3
    epochs:       int   = 10
    batch_size:   int   = 64
    lr:           float = 1e-3
    weight_decay: float = 1e-4


# ============================================================
# Raw torch.nn.Module
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


class TicTacToeCNN(nn.Module):
    """
    CNN backbone + dual-head for Tic-Tac-Toe.

    Smaller than ConnectFourCNN: 3 conv blocks instead of 4,
    64 channels instead of 128, smaller FC trunk.
    """

    def __init__(self, rows: int, cols: int, action_size: int, args: TTTNNetArgs):
        super().__init__()
        C = args.num_channels

        self.conv_trunk = nn.Sequential(
            _ConvBlock(1,  C),
            _ConvBlock(C,  C),
            _ConvBlock(C,  C),
        )

        flat_size = C * rows * cols
        self.fc_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
        )

        self.policy_head = nn.Linear(64, action_size)
        self.value_head  = nn.Linear(64, 1)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)                      # (B, 1, rows, cols)
        features = self.conv_trunk(x)            # (B, C, rows, cols)
        shared   = self.fc_trunk(features)       # (B, 64)
        log_pi = F.log_softmax(self.policy_head(shared), dim=-1)
        v      = torch.tanh(self.value_head(shared))
        return log_pi, v


# ============================================================
# NeuralNet adapter
# ============================================================

class TicTacToeNNet(NeuralNet):
    """alpha-zero-general NeuralNet adapter for TicTacToeCNN."""

    def __init__(self, game, args: TTTNNetArgs | None = None):
        super().__init__(game)
        self.rows, self.cols = game.getBoardSize()
        self.action_size      = game.getActionSize()
        self.args             = args or TTTNNetArgs()
        self.device           = _select_device()

        self.model = TicTacToeCNN(
            self.rows, self.cols, self.action_size, self.args
        ).to(self.device)

    def train(self, examples: List[Tuple]) -> None:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        boards_np = np.array([e[0] for e in examples], dtype=np.float32)
        pis_np    = np.array([e[1] for e in examples], dtype=np.float32)
        vs_np     = np.array([e[2] for e in examples], dtype=np.float32)

        dataset = TensorDataset(
            torch.from_numpy(boards_np),
            torch.from_numpy(pis_np),
            torch.from_numpy(vs_np),
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size,
                            shuffle=True, drop_last=False)

        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            total_pi_loss = total_v_loss = 0.0
            n_batches = 0

            for boards_b, pi_b, v_b in loader:
                boards_b = boards_b.to(self.device)
                pi_b     = pi_b.to(self.device)
                v_b      = v_b.to(self.device)

                log_pi, v_pred = self.model(boards_b)
                pi_loss = -(pi_b * log_pi).sum(dim=-1).mean()
                v_loss  = F.mse_loss(v_pred.squeeze(-1), v_b)
                loss = pi_loss + v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_pi_loss += pi_loss.item()
                total_v_loss  += v_loss.item()
                n_batches     += 1

            if n_batches:
                print(f"  Epoch {epoch:>2}/{self.args.epochs}  "
                      f"pi_loss={total_pi_loss/n_batches:.4f}  "
                      f"v_loss={total_v_loss/n_batches:.4f}")

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, float]:
        board_t = (
            torch.from_numpy(board.astype(np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        self.model.eval()
        with torch.no_grad():
            log_pi, v = self.model(board_t)
        pi = torch.exp(log_pi).squeeze(0).cpu().numpy()
        v  = v.squeeze().cpu().item()
        return pi, v

    def save_checkpoint(self, folder: str = "checkpoint",
                        filename: str = "checkpoint.pth.tar") -> None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save(
            {
                "state_dict":  self.model.state_dict(),
                "args":        dataclasses.asdict(self.args),
                "board_size":  (self.rows, self.cols),
                "action_size": self.action_size,
            },
            path,
        )
        print(f"Checkpoint saved → {path}")

    def load_checkpoint(self, folder: str = "checkpoint",
                        filename: str = "checkpoint.pth.tar") -> None:
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at {path!r}")
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        print(f"Checkpoint loaded ← {path}  (device={self.device})")

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return (
            f"TicTacToeNNet("
            f"board={self.rows}×{self.cols}, "
            f"actions={self.action_size}, "
            f"params={n_params:,}, "
            f"device={self.device})"
        )
