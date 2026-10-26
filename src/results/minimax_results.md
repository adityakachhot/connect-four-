# Win Rate vs. Minimax Baseline

**Games per matchup:** 40  
**Evaluation:** NNet greedy (single forward pass, no MCTS) vs Minimax with alpha-beta pruning  
**First mover:** alternated evenly (half the games each side)

| Run | Minimax Depth | NNet Wins | Minimax Wins | Draws | Win Rate (decisive) | Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | 2 | 36 | 3 | 1 | **92.3%** | 1.2s |
| Run 1 | 4 | 22 | 13 | 5 | **62.9%** | 8.0s |
| Run 1 | 6 | 15 | 22 | 3 | **40.5%** | 143.0s |
| Run 2 | 2 | 38 | 1 | 1 | **97.4%** | 1.0s |
| Run 2 | 4 | 27 | 10 | 3 | **73.0%** | 7.5s |
| Run 2 | 6 | 17 | 20 | 3 | **45.9%** | 149.1s |
| Run 3 | 2 | 38 | 1 | 1 | **97.4%** | 1.1s |
| Run 3 | 4 | 28 | 9 | 3 | **75.7%** | 6.5s |
| Run 3 | 6 | 18 | 19 | 3 | **48.6%** | 137.7s |

### Key Observations

- **Run 1**: avg win rate across depths = 65.2%
- **Run 2**: avg win rate across depths = 72.1%
- **Run 3**: avg win rate across depths = 73.9%

- **Depth 2**: avg NNet win rate = 95.7%
- **Depth 4**: avg NNet win rate = 70.5%
- **Depth 6**: avg NNet win rate = 45.0%
