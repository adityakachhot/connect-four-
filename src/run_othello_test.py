"""
Quick test runner for reference-repo's built-in Othello example.
Runs 1 iteration with reduced episode/simulation count to verify the framework
runs end-to-end without modifying any files inside reference-repo/.
"""
import os
import sys
import time

# Ensure reference-repo is on sys.path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reference-repo"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import main

def run_quick_iteration():
    print(f"Loaded reference-repo from: {REPO_DIR}")
    
    # Configure 1 quick iteration
    main.args.numIters = 1
    main.args.numEps = 2         # 2 self-play episodes
    main.args.numMCTSSims = 10   # 10 MCTS simulations per move
    main.args.arenaCompare = 2   # 2 arena games to compare new vs old model
    main.args.checkpoint = os.path.join(os.path.dirname(__file__), "..", "temp")
    
    print("\nStarting 1 quick iteration of Othello...")
    start_time = time.time()
    main.main()
    elapsed = time.time() - start_time
    print(f"\nQuick iteration finished successfully in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    run_quick_iteration()
