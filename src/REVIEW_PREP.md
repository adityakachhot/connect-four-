# Reviewer Presentation & Defense Preparation

> **Project:** Connect Four — AlphaZero Reimplementation & Generalization  
> **Platform:** Apple Silicon (M3 Air, MPS Backend)  
> **Companion Document:** [`src/REPORT.md`](file:///Users/adityakachhot/RL%20project/src/REPORT.md)  
> **Interactive UI:** [`src/ui/app.py`](file:///Users/adityakachhot/RL%20project/src/ui/app.py) (`make ui`)

---

## 1. 5-Minute Presentation Outline

This presentation is structured for a strict 5-minute technical briefing (6 slides, ~45–55 seconds per slide). Each slide is summarized in a single core takeaway line along with the exact visual asset or UI mode to present.

```
Slide 1 [0:00–0:45] │ Architecture & Goal    │ Pipeline & MCTS Diagrams (REPORT.md §2)
Slide 2 [0:45–1:35] │ Search Budget Scaling  │ fig_winrate_combined.png (Random baseline)
Slide 3 [1:35–2:25] │ Arena Draw-Saturation │ fig_arena_combined.png (Arena win rate & draw bands)
Slide 4 [2:25–3:15] │ Minimax Benchmark      │ minimax_bar_chart.png (Depths 2, 4, 6)
Slide 5 [3:15–4:10] │ Pipeline Validation    │ tictactoe_winrate.png (Tic-Tac-Toe convergence)
Slide 6 [4:10–5:00] │ System Deliverables    │ Streamlit UI Studio (src/ui/app.py) & Future Work
```

---

### Slide 1: Problem Formulation & System Architecture (0:00 – 0:45)
* **One-Line Takeaway:** We implemented a tabula rasa AlphaZero reinforcement learning pipeline on Connect Four that learns human-level play from zero domain knowledge using deep neural guidance and MCTS on consumer Apple Silicon.
* **Visual to Show:** Training Pipeline & MCTS Flowcharts from [`REPORT.md §2.1 & §2.2`](file:///Users/adityakachhot/RL%20project/src/REPORT.md#L23-L85).
* **Speaker Notes (45s):**
  * "Good morning/afternoon. The goal of this project was to independently reimplement and validate DeepMind's AlphaZero algorithm on Connect Four without relying on opening books, human game records, or hand-crafted evaluation heuristics."
  * "The system operates as a closed self-play loop: the current best dual-headed CNN guides Monte Carlo Tree Search via PUCT exploration to generate game trajectories, which train candidate networks on an experience replay buffer."
  * "All core game logic, CNN architectures, Apple Silicon MPS bindings, evaluation harnesses, and test suites were written independently in `src/` and `tests/`, preserving the reference framework strictly as an abstract base."

---

### Slide 2: Training Progression & MCTS Scaling (0:45 – 1:35)
* **One-Line Takeaway:** Doubling MCTS search depth from 25 to 50 simulations per move accelerated learning throughput by 63% and broke through the early draw plateau.
* **Visual to Show:** [`src/results/report/fig_winrate_combined.png`](file:///Users/adityakachhot/RL%20project/src/results/report/fig_winrate_combined.png) (Win rate vs. random baseline across Runs 1, 2, and 3).
* **Speaker Notes (50s):**
  * "Here we see our primary capability metric across three training runs: win rate against a random baseline using greedy network inference with zero tree search."
  * "Run 1 used 25 simulations per move. It climbed from 75% to 86%, but plateaued around iteration 32 when self-play games began ending in repeated draws."
  * "In Run 2, we doubled the search budget to 50 simulations. This produced a much steeper learning slope (+16.6 percentage points vs. +10.2 pp) because deeper search trees exposed the network to tactical multi-threats."
  * "Run 3 warm-started directly from Run 2's weights, immediately starting at 88% and demonstrating that training progress compounds across iterations without catastrophic forgetting."

---

### Slide 3: The Arena Draw-Saturation Phenomenon (1:35 – 2:25)
* **One-Line Takeaway:** At near-optimal skill levels, symmetric Connect Four matches produce 90% draws, making external baseline benchmarks strictly necessary to decouple real capability from internal arena noise.
* **Visual to Show:** [`src/results/report/fig_arena_combined.png`](file:///Users/adityakachhot/RL%20project/src/results/report/fig_arena_combined.png) (Arena promotion rate with highlighted all-draw bands).
* **Speaker Notes (50s):**
  * "One of our most significant empirical findings was the arena draw-saturation effect seen in this figure."
  * "The standard AlphaZero promotion criterion requires a candidate model to win at least 55% of decisive games against the current champion. However, in Run 3, 46 out of 50 iterations produced 100% drawn arena matches."
  * "To investigate whether learning had actually stalled, we conducted a post-hoc 40-game MCTS arena match between Run 2 best and Run 3 best. Exactly 90% of games (36 of 40) were draws, with decisive games splitting 2–2."
  * "This proved that the arena was suffering from draw saturation due to small sample sizes (10–20 games), while our external random and minimax baselines proved that the network's underlying positional understanding continued to advance."

---

### Slide 4: Classical Benchmark — AlphaZero vs. Alpha-Beta Minimax (2:25 – 3:15)
* **One-Line Takeaway:** A single forward pass through our trained neural network without search plays at roughly the tactical strength of a 5-ply exhaustive classical minimax lookahead.
* **Visual to Show:** [`src/results/minimax_bar_chart.png`](file:///Users/adityakachhot/RL%20project/src/results/minimax_bar_chart.png) (Win rate vs. Minimax at Depths 2, 4, 6).
* **Speaker Notes (50s):**
  * "To establish absolute playing strength against an objective non-neural opponent, we built an independent minimax player with alpha-beta pruning and streak-counting heuristics."
  * "Against Depth-2 Minimax (instant lookahead), our Run 3 best model achieved a 97.4% decisive win rate."
  * "Against Depth-4 Minimax (7 seconds per game), the raw network won 75.7% of decisive encounters without any tree search."
  * "At Depth 6 (140 seconds per game), exhaustive search flips the advantage to minimax, though our models still win 48.6% of decisive games. Crucially, each successive training run showed progressive gains against deeper classical search (40% to 46% to 49%), confirming that self-play quality translates directly into classical tactical resilience."

---

### Slide 5: Pipeline Generalization — Tic-Tac-Toe Validation (3:15 – 4:10)
* **One-Line Takeaway:** Swapping game rules with zero framework modifications produced a provably optimal agent that self-discovered the mandatory corner opening defense at Iteration 12.
* **Visual to Show:** [`src/results/tictactoe_winrate.png`](file:///Users/adityakachhot/RL%20project/src/results/tictactoe_winrate.png) (Tic-Tac-Toe Win Rate & Minimax Losses).
* **Speaker Notes (55s):**
  * "To confirm our pipeline is truly game-agnostic, we implemented Tic-Tac-Toe with $D_4$ dihedral symmetries and trained it with the exact same Coach and MCTS code."
  * "In early iterations, the network lost 25 out of 50 games against perfect depth-9 Minimax. Our investigation revealed that when Minimax opened in the center, the early network responded with an edge move, which guarantees a loss."
  * "At Iteration 12, a dramatic breakthrough occurred: accumulated replay experiences penalized edge responses, dropping losses from 25 to 1."
  * "From Iteration 13 onward, the agent achieved zero losses against perfect minimax (all 50/50 draws) and 100% win rate against random play, verifying that the self-play architecture generalizes seamlessly to other zero-sum domains."

---

### Slide 6: System Deliverables & Future Directions (4:10 – 5:00)
* **One-Line Takeaway:** We packaged the project into an interactive 3-mode Streamlit studio and an 84-test verified suite, laying the groundwork for GPU-batched parallel MCTS.
* **Visual to Show:** Streamlit UI Studio (`http://localhost:8501`) or [`README.md §6`](file:///Users/adityakachhot/RL%20project/README.md#L143-L153).
* **Speaker Notes (50s):**
  * "In addition to the algorithmic results, we delivered a complete software artifact: an 84-test pytest suite passing with zero failures, automated Makefile targets, and an interactive Streamlit UI studio."
  * "The UI provides live value-head win probability estimates and policy distributions during human play, automated AI vs. AI exhibition matches, and transcript replays."
  * "Looking forward, the key architectural upgrade is batched parallel MCTS on Metal GPU to scale search beyond 200 simulations, coupled with Gumbel AlphaZero search to minimize simulation requirements."
  * "I'd now like to give a quick 2-minute live demo of the graphical interface."

---

## 2. 2-Minute Live Demo Script (Streamlit UI)

### Prerequisites & Preparation
1. Activate environment and ensure the Streamlit server is running:
   ```bash
   cd "/Users/adityakachhot/RL project"
   source .venv/bin/activate
   make ui   # Or: streamlit run src/ui/app.py
   ```
2. Open browser at `http://localhost:8501`. Have the browser window positioned side-by-side with your terminal or notes.

---

### Chronological Demo Execution (Total: 2m 00s)

```
0:00 ─── [Setup] ─────────────── Introduce UI & Hardware (Apple Silicon MPS)
0:15 ─── [Part 1: Watch Mode] ── Configure Run 1 Early vs Run 3 Best
0:30 ─── [Part 1: Auto-Play] ─── Observe vertical stack vs multi-threat counter
1:00 ─── [Part 2: Play Mode] ─── Switch to Play vs Agent; Human drops in Col 4
1:15 ─── [Part 2: Diagnostics] ─ Highlight Value Win Prob & Policy Prior Chart
1:40 ─── [Part 2: Conclusion] ── Complete tactical sequence; point out banner
2:00 ─── [Wrap-Up] ───────────── Transition to reviewer Q&A
```

#### Step 1: Interface Overview (0:00 – 0:15)
* **Action:** Direct attention to the web browser header. Point to the sidebar displaying `Compute Hardware: Apple Silicon (MPS)`.
* **Spoken Cue:**  
  *"This is our Connect Four Studio built in Streamlit. All network forward passes run directly on the Apple Silicon GPU via PyTorch MPS, cached in memory via Streamlit resource caching for sub-second responses."*

---

#### Step 2: Watch Agent vs. Agent Exhibition (0:15 – 1:00)
* **Action:**
  1. In the sidebar radio, click **`⚔️ Watch Agent vs Agent`**.
  2. Under Player 1 (🔴 Red), select **`Run 1 Early (Iter 5)`**. Set MCTS Sims to `25`.
  3. Under Player 2 (🟡 Yellow), select **`Run 3 Best (Warm-started, 50 iters)`**. Set MCTS Sims to `25`.
  4. Set the speed slider to `0.25s`.
  5. Toggle **`Auto-Play`** to `ON`.
* **What Happens On-Screen:**
  * The board rapidly populates with red and yellow discs.
  * The live move feed on the right logs every ply in real time.
* **Spoken Cue:**  
  *"Let's pit our early 5-iteration model in red against our final 50-iteration model in yellow. Notice Player 1's naive strategy: it fixates on single-column vertical stacking in Column 3 with zero threat awareness."*  
  *(Wait 10 seconds for move 10–14)*  
  *"Watch Player 2 in yellow: it opens in center Column 4, establishes horizontal forks across rows 5 and 6, and at move 13, it immediately detects and blocks Player 1's vertical threat before securing its own winning alignment."*  
  *(Turn Auto-Play OFF once game concludes; point out yellow victory alert and running scoreboard tally).*

---

#### Step 3: Interactive Play vs. Agent & Neural Diagnostics (1:00 – 1:55)
* **Action:**
  1. In the sidebar radio, select **`🎮 Play vs Agent`**.
  2. Keep opponent as `Trained Agent (AlphaZero)` and checkpoint as `Run 3 Best`.
  3. Set difficulty to `25 MCTS Sims`.
  4. Click the **`⬇️ 4`** column button to play into the center column.
* **What Happens On-Screen:**
  * Red disc drops into column 4.
  * Within 200 milliseconds, the agent calculates MCTS and drops a yellow counter-disc.
  * The **Agent Diagnostics** panel on the right populates instantly.
* **Spoken Cue:**  
  *"Now let's switch to Play mode. I'll drop my opening disc into center column 4. The agent responds immediately on MPS."*  
  *(Point cursor to the Diagnostics panel on the right)*  
  *"Look at the real-time neural diagnostics: the value head estimates the agent's win probability at approximately 52%, reflecting the slight first-mover disadvantage. Below it, the policy distribution bar chart shows the network's raw action prior: over 80% of its probability mass was focused on adjacent Columns 3 and 4 to contest the center board."*  
* **Action:**
  * Drop another piece (e.g. `⬇️ 4` or `⬇️ 3`).
  * Show the agent reacting defensively or building a counter-threat.
  * If the game reaches a terminal state, highlight the result banner. If short on time, click `🔄 New Game` to show instantaneous state reset.

---

#### Step 4: Demo Conclusion (1:55 – 2:00)
* **Spoken Cue:**  
  *"The third mode in the sidebar allows stepping through static transcripts ply-by-ply. That concludes our live demonstration. I am happy to take your questions."*

---

## 3. 15 Reviewer Questions & Concise, Honest Answers

### Q1: Why did your arena promotion results saturate into draws during late training?
**Answer:**  
Connect Four is mathematically solved as a first-player win under perfect play (*Allis, 1988*), but under heuristic or bounded search, high-level counter-play between two well-matched models rapidly produces draws. In Run 3, both candidate and champion models initialized from the same mature Run 2 weights, causing 46 of 50 training iterations to result in 100% drawn arena matches. Because the acceptance criterion requires winning $\ge 55\%$ of *decisive* games, 0 decisive games yields a 0% win rate, causing candidates to be rejected. Our post-hoc 40-game MCTS arena between Run 2 and Run 3 confirmed this: **90% of games (36/40) ended in draws**, with decisive games splitting 2–2. This was draw saturation, not stalled learning, as confirmed by our external random and minimax baselines.

---

### Q2: Why did you increase `numMCTSSims` from 25 to 50 between Run 1 and Run 2?
**Answer:**  
In Run 1, 25 simulations per move was insufficient for MCTS to resolve tactical branching beyond 3–4 plies, causing arena matches to collapse into complete draw stalemates (0-0-20) from iteration 32 onward. Doubling to 50 simulations in Run 2 expanded the search horizon, allowing MCTS to detect deeper 4-in-a-row traps and convert stalemates into decisive wins. This broke the draw plateau (reducing all-draw iterations from 26% in Run 1 to 14% in Run 2) and steepened the learning slope from +10.2 percentage points to +16.6 pp.

---

### Q3: What exactly did you implement yourself versus what did the reference repository provide?
**Answer:**  
We strictly respected the architectural boundary. The reference repository (`suragnair/alpha-zero-general`) provided abstract base classes (`Game`, `NeuralNet`), the generic MCTS tree search algorithm, and the `Coach`/`Arena` loops for Othello.  
**We implemented 100% of the game and project code in `src/` and `tests/`:**
1. `ConnectFourGame` (6×7 grid, gravity drops, four-in-a-row bit/array detection, $\mathbb{Z}_2$ symmetry).
2. `ConnectFourCNN` (PyTorch 4-block CNN trunk, dual policy/value heads, Apple Silicon MPS acceleration).
3. `TicTacToeGame` and `TicTacToeCNN` for cross-game pipeline validation ($D_4$ dihedral symmetry group).
4. `MinimaxAgent` with alpha-beta pruning and hand-crafted streak/center heuristics.
5. `TimedCoach` and independent evaluation harnesses (`game_runner.py`) decoupled from `Arena.py`.
6. The interactive Streamlit studio (`src/ui/app.py`), the full 84-test pytest suite, and project automation (`Makefile`).  
Not a single file in `reference-repo/` was modified.

---

### Q4: Why is there no dataset in your repository?
**Answer:**  
AlphaZero is a *tabula rasa* reinforcement learning paradigm. It explicitly rejects static training datasets, grandmaster game logs, and human opening books. All training data is generated dynamically on-the-fly through self-play episodes. The network predicts prior probabilities and values; MCTS refines them into a stronger search policy; and the self-play game outcome provides the terminal reward ($+1$, $-1$, $0$). These triples are cached in a rolling experience replay buffer (`.examples` files) and discarded after several iterations.

---

### Q5: What are the primary hardware and algorithmic limitations of this project?
**Answer:**  
1. **Compute Scale Discrepancy:** We trained on a single consumer laptop (Apple M3 Air, fanless passive cooling with thermal throttling under sustained load). This represents roughly six orders of magnitude less compute than the original AlphaZero paper, which utilized 5,000 Google TPUs generating millions of games at 800 simulations per move.
2. **Search Horizon:** At 50 MCTS simulations per move on a 42-ply game tree, the search explores a tiny fraction ($\approx 2,100$ nodes per game) of Connect Four's $4.5 \times 10^{12}$ reachable positions.
3. **Python MCTS Bottleneck:** Tree traversal and node expansions are executed sequentially in single-threaded Python rather than via GPU-batched leaf expansions with virtual loss.
4. **Arena Statistical Noise:** Testing candidate models across only 10–20 arena games introduced variance in model promotion.

---

### Q6: Why did Run 3 (warm-started from Run 2) not show a higher win rate against Run 2 in head-to-head play?
**Answer:**  
In our post-hoc 40-game MCTS arena at 50 sims/ply, 36 of 40 games (90%) ended in draws, with decisive games splitting exactly 2–2 (50.0% decisive win rate). Because Run 3 initialized from Run 2's best weights, both models were operating at the resolution ceiling of a 50-simulation MCTS search on Connect Four. Continued self-play reinforced mature positional play but could not escape the draw plateau without either deeper search ($\ge 200$ simulations) or a more expressive network architecture.

---

### Q7: Why do you evaluate checkpoints using a greedy policy (temperature=0, single forward pass) rather than full MCTS?
**Answer:**  
First, speed: greedy evaluation takes milliseconds per game, enabling 40–50 game evaluations across multiple baselines in seconds. Second, isolation: greedy evaluation tests the *raw representation quality* learned by the neural network without confounding it with search depth. Because greedy evaluation omits MCTS lookahead, the reported ~88% random win rate and ~76% depth-4 minimax win rate represent a strict lower bound on the model's true playing strength.

---

### Q8: Why did Tic-Tac-Toe lose 25 out of 50 games against Minimax in iterations 1 through 11?
**Answer:**  
In Tic-Tac-Toe, when Player 1 opens in the center cell (1,1), an edge response (e.g. (0,1) or (2,1)) is the *only* response that guarantees a loss against perfect play; only a corner response guarantees a draw. In iterations 1–11, the network's policy was nearly uniform (~11% per cell), so `argmax` frequently selected an edge when playing second (exactly 25 of the 50 games). At Iteration 12, the accumulated replay buffer finally gathered enough loss trajectories to penalize edge lines, shifting >98% probability mass to corners and achieving zero losses (50/50 draws) from Iteration 13 onward.

---

### Q9: How does canonical board representation work, and why is it necessary?
**Answer:**  
The board array is multiplied by the active player's integer identifier: `board * player`. When it is Player 1's turn ($+1$), the board is unchanged. When it is Player 2's turn ($-1$), all pieces flip signs: $-1 \times -1 = +1$ (own pieces) and $+1 \times -1 = -1$ (opponent pieces). This normalizes the state so the neural network only ever needs to learn how to evaluate the position from the perspective of "self = $+1$", cutting the effective state space in half and eliminating the need for two separate networks.

---

### Q10: How is your classical Minimax baseline implemented, and is it a fair opponent?
**Answer:**  
It is an independent, non-neural alpha-beta searcher implemented in `src/eval/minimax_player.py` using only the `ConnectFourGame` interface. It uses standard domain heuristics: a weight bonus for center-column control and sliding 4-cell window scoring across rows, columns, and diagonals (+100 for 4-in-a-row, +5 for 3-in-a-row, +2 for 2-in-a-row, and $-1000$ for opponent unblocked 3-in-a-rows). It provides a deterministic, rigorous sanity benchmark that scales strictly with ply depth (2, 4, 6) without any neural network or MCTS bias.

---

### Q11: How did you handle board symmetries, and why is it important?
**Answer:**  
In Connect Four, we exploit bilateral horizontal symmetry ($\mathbb{Z}_2$), flipping the board left-to-right (`np.fliplr`) and mirroring the policy vector columns ($0 \leftrightarrow 6, 1 \leftrightarrow 5, 2 \leftrightarrow 4$) while leaving the pass action unchanged. In Tic-Tac-Toe, we exploit the full dihedral group $D_4$ (8 rotations and reflections). Symmetries provide a $2\times$ data augmentation multiplier for Connect Four and an $8\times$ multiplier for Tic-Tac-Toe at zero simulation cost, dramatically improving sample efficiency.

---

### Q12: Why did you choose a plain 4-block ConvNet rather than a modern ResNet or Transformer?
**Answer:**  
Connect Four is played on a compact $6 \times 7$ grid. Four convolutional layers with $3 \times 3$ kernels and padding=1 yield a receptive field of $9 \times 9$, which already spans the entire board. With 1.85 million parameters, the network trained in 45–88 seconds per iteration on consumer Apple Silicon with zero gradient vanishing issues. A deeper ResNet would have increased inference latency during MCTS without measurable benefit at 50 simulations per move.

---

### Q13: How do you know the network didn't just overfit to self-play games?
**Answer:**  
Three empirical results prove generalisation:
1. **Random Baseline:** Maintained an 82–100% win rate against a stochastic random player that generates erratic, non-standard board states never encountered in self-play.
2. **Minimax Baseline:** Achieved a 75.7% win rate against a depth-4 alpha-beta searcher using human heuristics the network never observed.
3. **Loss Convergence:** Value loss stabilized at ~0.02 and policy loss at ~0.20 across rolling replay buffers containing thousands of distinct board states without diverging.

---

### Q14: Why does Connect Four have 8 actions when the board only has 7 columns?
**Answer:**  
Actions 0 through 6 represent dropping a piece into columns 0 through 6. Action 7 is a dummy "pass" action required by the reference MCTS tree implementation when a game reaches a full-board terminal draw position. In our `getValidMoves()`, action 7 is strictly masked to 0 during active play and only becomes valid if all 7 columns are completely full.

---

### Q15: If you had an NVIDIA A100 GPU and 48 hours, what would you change first?
**Answer:**  
First, I would implement **batched parallel MCTS on GPU** using virtual loss, scaling search from 50 to 800 simulations per move. Second, I would increase the arena sample size from 10 to 100+ games to eliminate promotion noise. Third, I would implement **Gumbel AlphaZero** (*Danihelka et al., 2022*) to achieve policy improvement guarantees with fewer rollouts, and evaluate an 8-block ResNet with squeeze-and-excitation channel attention.

---

*Document compiled in `src/REVIEW_PREP.md` for project presentation and technical oral examination.*
