# alpha-zero-general — Framework Architecture Notes

> **Purpose**: Documentation-only notes on how `Game`, `NeuralNet`, `Coach`, `MCTS`,
> and `Arena` fit together, plus a concrete walkthrough of the Othello reference
> implementation. No game-specific code of your own yet.

---

## 1. `Game` — the abstract game interface

**File**: `reference-repo/Game.py`

`Game` is a plain Python base class (no `ABC` machinery, just `pass` bodies) that
every game must subclass and override. All methods operate on _boards_, whose
internal representation is entirely up to the implementer — the only constraint is
that the same representation is used everywhere.

The framework assumes **two-player, zero-sum, turn-based, perfect-information**
games. Players are always represented as integers: `1` for player 1, `-1` for
player 2.

### Abstract methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `getInitBoard` | `()` | board | The starting board state. Ideally shaped to be a direct NN input. |
| `getBoardSize` | `()` | `(x, y)` | Board dimensions as a tuple (rows, cols). Used by the NN to size its input layer. |
| `getActionSize` | `()` | `int` | Total number of discrete actions. Used to size the NN policy head and valid-move vectors. |
| `getNextState` | `(board, player, action)` | `(nextBoard, nextPlayer)` | Apply one legal action. `nextPlayer` should always be `-player`. |
| `getValidMoves` | `(board, player)` | `np.array` of 0/1, length `getActionSize()` | Binary mask: 1 = legal, 0 = illegal. |
| `getGameEnded` | `(board, player)` | `0 or 1 or -1 or small != 0` | `0` = still going; `1` = player wins; `-1` = player loses; small non-zero = draw. |
| `getCanonicalForm` | `(board, player)` | board | Normalise the board to always look "from player 1's perspective", regardless of whose turn it is. |
| `getSymmetries` | `(board, pi)` | `list[(board, pi)]` | Data-augmentation: return all symmetric board+policy pairs (rotations, reflections). |
| `stringRepresentation` | `(board)` | `str` or `bytes` | Fast, hashable encoding used by MCTS as a dict key. |

---

## 2. `NeuralNet` — the abstract value/policy network interface

**File**: `reference-repo/NeuralNet.py`

`NeuralNet` is a thin adapter interface. It always receives boards in **canonical
form** — it never needs to know whose turn it is. The wrapper class (not the raw
`nn.Module`) is what Coach and MCTS hold a reference to.

### Abstract methods

| Method | Signature | Returns | Notes |
|---|---|---|---|
| `train` | `(examples)` | — | `examples` is a list of `(board, pi, v)` tuples. `pi` is the MCTS-refined policy vector; `v` is the game outcome. Called once per Coach iteration after self-play. |
| `predict` | `(board)` | `(pi, v)` | Single inference: returns a policy vector (probabilities, length `getActionSize()`) and scalar `v in [-1, 1]`. Called by MCTS at every leaf node. |
| `save_checkpoint` | `(folder, filename)` | — | Persist weights to `folder/filename`. |
| `load_checkpoint` | `(folder, filename)` | — | Restore weights from `folder/filename`. |

---

## 3. `Coach.py` — orchestrator: call sequence per iteration

**File**: `reference-repo/Coach.py`

```
Coach.learn()  (runs numIters times)
│
├─ Phase 1 — Self-play data collection
│   └─ for each episode (numEps):
│       Coach.executeEpisode()
│       │
│       └─ loop until game ends:
│           ├─ game.getCanonicalForm(board, curPlayer)         ← Game
│           ├─ mcts.getActionProb(canonicalBoard, temp)        ← MCTS (calls NNet internally)
│           ├─ game.getSymmetries(canonicalBoard, pi)          ← Game
│           ├─ np.random.choice(len(pi), p=pi)                 ← pick action
│           ├─ game.getNextState(board, curPlayer, action)     ← Game
│           └─ game.getGameEnded(board, curPlayer)             ← Game (loop exit check)
│
├─ Phase 2 — Network training
│   ├─ nnet.save_checkpoint(folder, 'temp.pth.tar')           ← NeuralNet (snapshot current)
│   ├─ pnet.load_checkpoint(folder, 'temp.pth.tar')           ← NeuralNet (competitor = old)
│   └─ nnet.train(trainExamples)                              ← NeuralNet (update current)
│
└─ Phase 3 — Arena evaluation
    ├─ Arena.playGames(arenaCompare)
    │   └─ per game:
    │       ├─ game.getInitBoard()                             ← Game
    │       ├─ game.getGameEnded(board, player)               ← Game (loop condition)
    │       ├─ game.getCanonicalForm(board, curPlayer)         ← Game
    │       ├─ player_fn(canonicalBoard)  → action            ← MCTS.getActionProb(temp=0) lambda
    │       ├─ game.getValidMoves(...)                        ← Game (validity check)
    │       └─ game.getNextState(board, curPlayer, action)    ← Game
    │
    └─ if new net win-rate >= updateThreshold:
        nnet.save_checkpoint(..., 'best.pth.tar')             ← NeuralNet (accept new model)
       else:
        nnet.load_checkpoint(..., 'temp.pth.tar')             ← NeuralNet (revert to old)
```

### Key ordering invariants

1. **`getCanonicalForm` before every NN call**: The board is always normalised to
   player-1 perspective before being fed to `mcts.getActionProb` or `nnet.predict`.
2. **Symmetries collected during self-play, not at training time**: `getSymmetries`
   is called inside `executeEpisode` to augment examples in place.
3. **`save_checkpoint` / `load_checkpoint` bracket every training step**: The old
   weights are frozen as `pnet` before `nnet.train()` runs, so the arena comparison
   is always new-vs-prior.
4. **MCTS trees are discarded per episode**: `self.mcts = MCTS(...)` is reset at the
   start of each `executeEpisode`, so cached Q-values don't bleed between games.

---

## 4. `MCTS.py` — interaction with `Game` and `NeuralNet`

**File**: `reference-repo/MCTS.py`

MCTS maintains five in-memory dicts keyed by `s = game.stringRepresentation(board)`:

| Dict | Stores |
|---|---|
| `Ps[s]` | Prior policy from NN (masked to valid moves, renormalised) |
| `Vs[s]` | Valid moves binary vector |
| `Ns[s]` | Visit count for state `s` |
| `Qsa[(s,a)]` | Running-average Q-value for (state, action) pair |
| `Nsa[(s,a)]` | Visit count for edge (s, a) |
| `Es[s]` | Terminal result cache for state `s` |

### Per-simulation call sequence (`MCTS.search` recursion)

```
MCTS.getActionProb(canonicalBoard)
│
├─ for i in range(numMCTSSims):
│   MCTS.search(canonicalBoard)                 ← recursive tree walk
│   │
│   ├─ game.stringRepresentation(board)         ← Game: hash current state
│   │
│   ├─ [terminal check]
│   │   game.getGameEnded(board, player=1)      ← Game: is this a leaf?
│   │   if terminal → return -outcome           (no NN call)
│   │
│   ├─ [first visit — leaf expansion]
│   │   nnet.predict(board)                     ← NeuralNet: get (P, v)
│   │   game.getValidMoves(board, player=1)     ← Game: mask invalid moves in P
│   │   return -v                              (backup value to parent)
│   │
│   └─ [interior node — UCB action selection]
│       for a in range(game.getActionSize()):   ← Game: iterate action space
│           UCB = Q(s,a) + cpuct * P[s][a] * sqrt(Ns[s]) / (1 + Nsa[s,a])
│       pick best_act
│       game.getNextState(board, 1, best_act)   ← Game
│       game.getCanonicalForm(next_s, next_pl)  ← Game
│       v = MCTS.search(next_canonical)         ← recurse
│       update Qsa, Nsa, Ns
│       return -v
│
└─ build policy from Nsa visit counts
   game.stringRepresentation(board)             ← Game (final count lookup)
   game.getActionSize()                         ← Game (size output vector)
```

### Design points

- **MCTS always calls `nnet.predict` with a canonical board**: The caller normalises
  before `getActionProb`; MCTS also normalises again via `getCanonicalForm` at each
  recursive expansion.
- **Value sign-flip at every level**: The recursive call returns `−v`, naturally
  handling the zero-sum perspective flip without the NN needing to know the player.
- **Invalid-move masking happens inside MCTS, not in the NN**: The NN returns a raw
  prior `P`; MCTS multiplies by `getValidMoves` and renormalises. If all valid moves
  get zero prior, MCTS falls back to a uniform prior over valid moves and logs an error.

---

## 5. Othello reference implementation

### `OthelloGame` — concrete `Game` subclass

**File**: `reference-repo/othello/OthelloGame.py`

Constructor takes `n` (board side length; default 6 in `main.py`).
Board state = `numpy.ndarray` of shape `(n, n)` with values in `{-1, 0, 1}`.
Move logic delegated to `OthelloLogic.Board`.

| Abstract method | Othello implementation |
|---|---|
| `getInitBoard` | Creates `Board(n)`, returns `np.array(b.pieces)` — standard 4-piece starting position. |
| `getBoardSize` | Returns `(n, n)`. |
| `getActionSize` | Returns `n*n + 1`. The `+1` encodes a _pass_ move for when a player has no legal moves. |
| `getNextState` | If `action == n*n` (pass): return `(board, -player)` unchanged. Else reconstruct `Board`, call `execute_move((row, col), player)`, return `(b.pieces, -player)`. |
| `getValidMoves` | Calls `Board.get_legal_moves(player)`. If empty, sets pass slot (`valids[-1] = 1`). Maps `(x, y)` to flat index `n*x + y`. |
| `getGameEnded` | Uses `Board.has_legal_moves` for both players; if neither can move, uses `Board.countDiff(player)` to determine winner (`1` or `-1`). No draw value returned. |
| `getCanonicalForm` | `player * board` — since player is ±1, this flips piece signs for player −1 so the NN always sees its own pieces as `+1`. |
| `getSymmetries` | 4 rotations × 2 reflections = **8 augmentations** via `np.rot90` + `np.fliplr`. The pass-move slot `pi[-1]` is preserved unchanged in each copy. |
| `stringRepresentation` | `board.tostring()` — bytes representation of the raw array. **Requires NumPy < 2.0** (`.tostring()` was removed in NumPy 2.0; the equivalent is `.tobytes()`). |

### `NNetWrapper` + `OthelloNNet` — concrete `NeuralNet` subclass

**Files**:
- `reference-repo/othello/pytorch/NNet.py` — the `NeuralNet` adapter (Coach/MCTS interface)
- `reference-repo/othello/pytorch/OthelloNNet.py` — the `nn.Module` (raw PyTorch model)

#### Architecture (`OthelloNNet`)

```
Input:  (batch, 1, n, n)  — board as single-channel "image"

Conv2d(1 → 512ch, 3×3, pad=1) + BN + ReLU     → (b, 512, n,   n  )
Conv2d(512 → 512ch, 3×3, pad=1) + BN + ReLU   → (b, 512, n,   n  )
Conv2d(512 → 512ch, 3×3, no-pad) + BN + ReLU  → (b, 512, n-2, n-2)
Conv2d(512 → 512ch, 3×3, no-pad) + BN + ReLU  → (b, 512, n-4, n-4)

Flatten → Linear(512*(n-4)^2 → 1024) + BN + Dropout
         → Linear(1024 → 512) + BN + Dropout

Policy head:  Linear(512 → action_size) → log_softmax   π ∈ R^{action_size}
Value head:   Linear(512 → 1)           → tanh           v ∈ [-1, 1]
```

> For n=6: flatten size = 512 × 2 × 2 = 2,048. For n=8: 512 × 4 × 4 = 8,192.
> Architecture requires n ≥ 5.

#### `NNetWrapper` method implementations

| Method | Implementation |
|---|---|
| `train(examples)` | `args.epochs` (10) passes, Adam optimizer, batch size 64, random mini-batch sampling. Loss = policy cross-entropy + value MSE. |
| `predict(board)` | Converts board to `FloatTensor (1, n, n)`, eval mode, `torch.no_grad()`. Returns `(exp(log_pi), v)` as NumPy — **probabilities**, not log-probs. |
| `save_checkpoint` | `torch.save({'state_dict': nnet.state_dict()}, path)` |
| `load_checkpoint` | `torch.load(path, map_location)` + `load_state_dict`. `map_location='cpu'` when no CUDA. |

> **MPS note**: `NNetWrapper` only checks `torch.cuda.is_available()`, so on an M3 Air
> it defaults to CPU. Adding MPS support requires patching the device selection in the
> wrapper — that is a later step, not covered here.

---

## 6. Master call-site reference table

| Caller | Method | On |
|---|---|---|
| `Coach.executeEpisode` | `getInitBoard` | Game |
| `Coach.executeEpisode` | `getCanonicalForm` | Game |
| `Coach.executeEpisode` | `mcts.getActionProb` | MCTS |
| `Coach.executeEpisode` | `getSymmetries` | Game |
| `Coach.executeEpisode` | `getNextState` | Game |
| `Coach.executeEpisode` | `getGameEnded` | Game |
| `Coach.learn` | `nnet.save_checkpoint` | NeuralNet |
| `Coach.learn` | `pnet.load_checkpoint` | NeuralNet |
| `Coach.learn` | `nnet.train` | NeuralNet |
| `Coach.learn` | `Arena.playGames` | Arena |
| `Coach.learn` | `nnet.save_checkpoint` / `load_checkpoint` | NeuralNet |
| `Arena.playGame` | `getInitBoard` | Game |
| `Arena.playGame` | `getGameEnded` | Game |
| `Arena.playGame` | `getCanonicalForm` | Game |
| `Arena.playGame` | `player_fn` (wraps `mcts.getActionProb(temp=0)`) | MCTS |
| `Arena.playGame` | `getValidMoves` | Game |
| `Arena.playGame` | `getNextState` | Game |
| `MCTS.search` | `stringRepresentation` | Game |
| `MCTS.search` | `getGameEnded` | Game |
| `MCTS.search` | `nnet.predict` | NeuralNet |
| `MCTS.search` | `getValidMoves` | Game |
| `MCTS.search` | `getActionSize` | Game |
| `MCTS.search` | `getNextState` | Game |
| `MCTS.search` | `getCanonicalForm` | Game |
| `MCTS.getActionProb` | `stringRepresentation` | Game |
| `MCTS.getActionProb` | `getActionSize` | Game |
