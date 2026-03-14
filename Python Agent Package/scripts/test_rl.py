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
from agents.agent_rl import AgentRL


# Configuration
NUM_EPISODES = 10
PORT = 5000
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
BASE_SEED = 123
MODEL_PATH = "dqn_actions_best.pt"


# Agent factory
def make_agent():
    agent = AgentRL.load(MODEL_PATH)
    agent.set_training_mode(False) # Evaluation mode
    agent.set_epsilon(0.0) # No exploration
    return agent


# Main
def main():
    env = MsPacmanEnv(port=PORT)
    agent = make_agent()

    # Global accumulators
    global_heatmap = defaultdict(int)
    all_decision_times = []
    metrics_rows = []

    for ep in range(1, NUM_EPISODES + 1):
        seed = BASE_SEED + ep

        # Memory tracking
        tracemalloc.start()

        obs = env.reset(seed)
        done = False
        steps = 0
        decision_times = []
        episode_heatmap = defaultdict(int)

        while not done:
            t0 = time.perf_counter()
            action = agent.act(obs, env)
            dt = time.perf_counter() - t0

            decision_times.append(dt)
            all_decision_times.append(dt)

            obs, reward, done, info = env.step(action)

            loc = obs["pac_loc"]
            episode_heatmap[loc] += 1
            global_heatmap[loc] += 1
            steps += 1

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

    env.close()

    # Save metrics to CSV
    csv_path = os.path.join(OUT_DIR, "rl_metrics.csv")
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

    np.save(os.path.join(OUT_DIR, "rl_heatmap.npy"), heatmap_array)

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
    plt.title("RL Agent (DQN) - Movement Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rl_heatmap.png"), dpi=200)
    plt.close()

    # Console summary
    scores = [r[2] for r in metrics_rows]
    successes = sum(r[6] for r in metrics_rows)

    print("=" * 70)
    print("Benchmark Summary - RL Agent (DQN)")
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