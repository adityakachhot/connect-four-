# Tic-Tac-Toe — Pipeline Validation Results

**Training (v2):** numIters=25, numEps=40, numMCTSSims=50  
**Evaluation:** 50 games each (alternating first mover)  
**Minimax:** depth=9 (perfect play for Tic-Tac-Toe)

| Iter | vs Random W-L-D | Win Rate | vs Minimax(d=9) W-L-D | Win Rate |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 44-2-4 | **95.7%** | 2-25-23 | **7.4%** |
| 2 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 3 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 4 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 5 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 6 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 7 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 8 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 9 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 10 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 11 | 47-0-3 | **100.0%** | 0-25-25 | **0.0%** |
| 12 | 47-0-3 | **100.0%** | 0-1-49 | **0.0%** |
| 13 | 48-0-2 | **100.0%** | 0-0-50 | **—** |
| 14–25 | 47-0-3 | **100.0%** | 0-0-50 | **—** |

> ✅ **Final checkpoint never lost to minimax (perfect play).** The agent achieves
> game-theoretically optimal play (all draws against perfect minimax) from iteration 13 onward.

### Convergence Summary

| Phase | Iterations | Minimax Losses/50 | What Changed |
|---|:---:|:---:|---|
| Early training | 1 | 25 | Policy nearly uniform; loses every 2nd-player game |
| Plateau | 2–11 | 25 | Still plays edges vs center opening |
| Breakthrough | 12 | 1 | Learns corner response; 1 residual loss |
| Perfect play | 13–25 | 0 | Zero losses — all draws against perfect minimax ✅ |

### Investigation: Why iter 1–11 all lose 25/50

The NNet initially responds to a center opening with an **edge move** (e.g. (2,1)) instead of
a **corner**. In Tic-Tac-Toe, an edge response to center is the *only* opening that guarantees
a loss against perfect play. With only 25 MCTS sims in the initial self-play data, the search
isn't deep enough to discover this. The breakthrough at iter 12 coincides with the training
data accumulating enough games where edge-response lines lead to losses, allowing the policy
to shift probability mass toward corners.
