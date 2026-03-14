import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
from agents.agent_rl import AgentRL
from env.ms_pacman_env import MsPacmanEnv

# Configuration
NUM_EPISODES = 1500
MAX_STEPS_PER_EP = 10000
SAVE_EVERY = 50
# Model paths
MODEL_PATH = "dqn_actions.pt"
BEST_MODEL_PATH = "dqn_actions_best.pt"

# Main
def main():
    print("=" * 70)
    print(" Starting RL Training ")
    print("=" * 70)

    agent = AgentRL()
    agent.set_training_mode(True)

    scores = []
    best_score = -1
    total_steps = 0

    # Training loop
    for ep in range(1, NUM_EPISODES + 1):
        try:
            env = MsPacmanEnv(port=5000)
            obs = env.reset(seed=ep)
        except Exception as e:
            print(f"[EP {ep}] Env connection failed: {e}")
            continue

        done = False
        steps = 0

        # Episode loop
        while not done and steps < MAX_STEPS_PER_EP:
            action = agent.act(obs, env)
            obs, reward, done, _ = env.step(action)
            steps += 1
            total_steps += 1

        env.close()

        score = int(obs.get("score", 0))
        scores.append(score)

        avg50 = np.mean(scores[-50:]) if len(scores) >= 10 else np.mean(scores)

        if score > best_score:
            best_score = score
            agent.save(BEST_MODEL_PATH)

        if ep % SAVE_EVERY == 0:
            agent.save(MODEL_PATH)

        # Console log
        print(
            f"[EP {ep:4d}] "
            f"Score={score:5d} | "
            f"Avg50={avg50:7.1f} | "
            f"Best={best_score:5d} | "
            f"Eps={agent.epsilon:.3f} | "
            f"Steps={total_steps}"
        )
    
    # Final save
    agent.save(MODEL_PATH)
    print("\nTraining finished.")
    print(f"Best score: {best_score}")
    print(f"Final Avg50: {np.mean(scores[-50:]):.1f}")


if __name__ == "__main__":
    main()