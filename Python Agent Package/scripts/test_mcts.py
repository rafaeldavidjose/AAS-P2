from __future__ import annotations

import os
import sys
import time
import csv
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.ms_pacman_env import MsPacmanEnv
from agents.agent_mcts import MCTSAgent


# Configuration
NUM_EPISODES = 10
PORT_MAIN = 5000
PORT_ROLLOUT = 5001
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
BASE_SEED = 123

# Agent factory
def make_agent(seed: int):
    return MCTSAgent(
        seed=seed,
        simulations=30,
        rollout_depth=20,
    )

# Main
def main():
    env = MsPacmanEnv(port=PORT_MAIN)
    agent = make_agent(BASE_SEED)

    # Global accumulators
    global_heatmap = defaultdict(int)
    all_decision_times = []
    metrics_rows = []

    # Episodes loop
    for ep in range(1, NUM_EPISODES + 1):
        seed = BASE_SEED + ep

        # Memory tracking
        tracemalloc.start()

        agent.set_episode_seed(seed)
        obs = env.reset(seed)
        done = False
        steps = 0
        decision_times = []
        episode_heatmap = defaultdict(int)

        # Step loop
        while not done:
            t0 = time.perf_counter()
            action = agent.act(obs)
            dt = time.perf_counter() - t0

            decision_times.append(dt)
            all_decision_times.append(dt)

            obs, reward, done, info = env.step(action)

            loc = obs["pac_loc"]
            episode_heatmap[loc] += 1
            global_heatmap[loc] += 1
            steps += 1
            print(f"Step {steps:03d}: Action={action} | PacLoc={loc} | Reward={reward} | Score={obs['score']} | Lives={obs['lives']}")

        # Memory usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        score = obs["score"]
        success = int(obs.get("cur_level", 1) > 1)

        metrics_rows.append([
            ep,
            seed,
            score,
            steps,
            np.mean(decision_times) * 1000,
            max(decision_times) * 1000,
            success,
            peak_mem / (1024 * 1024),
        ])

        print(
            f"[EP {ep:02d}] "
            f"Score={score:5d} | "
            f"Steps={steps:4d} | "
            f"Success={bool(success)}"
        )

    agent.close()
    env.close()

    # Save metrics to CSV
    csv_path = os.path.join(OUT_DIR, "mcts_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "seed",
            "score",
            "steps",
            "avg_decision_time_ms",
            "max_decision_time_ms",
            "success",
            "peak_memory_mb",
        ])
        writer.writerows(metrics_rows)

    # Save heatmap data
    max_tile = max(global_heatmap.keys())
    heatmap_array = np.zeros(max_tile + 1)

    for tile, count in global_heatmap.items():
        heatmap_array[tile] = count

    np.save(os.path.join(OUT_DIR, "mcts_heatmap.npy"), heatmap_array)

    # Plot heatmap
    plt.figure(figsize=(14, 3))
    plt.imshow(
        heatmap_array.reshape(1, -1),
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(label="Visit Frequency")
    plt.yticks([])
    plt.xlabel("Tile Index (pac_loc)")
    plt.title("MCTS Agent – Movement Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mcts_heatmap.png"), dpi=200)
    plt.close()

    # Console summary
    scores = [r[2] for r in metrics_rows]
    successes = sum(r[6] for r in metrics_rows)

    print("=" * 70)
    print("Benchmark Summary - MCTS Agent")
    print(f"Episodes:{NUM_EPISODES}")
    print(f"Avg Score: {np.mean(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Success Rate: {successes / NUM_EPISODES * 100:.1f}%")
    print(f"Avg Decision Time: {np.mean(all_decision_times)*1000:.3f} ms")
    print("=" * 70)
    print(f"Saved:")
    print(f" {csv_path}")
    print(f" heuristic_heatmap.npy")
    print(f" heuristic_heatmap.png")


if __name__ == "__main__":
    main()