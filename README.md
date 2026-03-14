# Pac-Man Playing Agents: Heuristic, MCTS, and Reinforcement Learning

**Autonomous Agent Systems — Master in Artificial Intelligence for Games**  
**Universidade Lusófona** | Rafael José (a22505223)

---

## Overview

This project implements and compares three decision-making agents for the **Ms. Pac-Man** environment, built on top of a Java–Python framework. The goal is to evaluate different approaches to real-time game control, from rule-based methods to planning and learning-based techniques.

| Agent | Approach |
|---|---|
| **Heuristic** | Hand-crafted rules + graph-based BFS distances |
| **MCTS** | Forward planning via Monte Carlo Tree Search |
| **RL (DQN)** | Deep Q-Learning with experience replay |

---

## Repository Structure

```
AAS-P2/
├── Java Server and Visual Package/
│   ├── EnvServer.jar           # Java game server
│   ├── PythonPolicyPacMan.jar  # Visual policy server
│   └── README.md
├── Python Agent Package/
│   ├── agents/                 # Heuristic, MCTS, and RL agent implementations
│   ├── datasets/               # Episode data / logs
│   ├── env/                    # Python environment interface
│   ├── models/                 # Saved model weights
│   ├── readmes/                # Per-agent documentation
│   ├── results/                # Evaluation outputs and plots
│   ├── scripts/                # Training and evaluation scripts
│   ├── servers/                # Server utilities
│   ├── dqn_actions.pt          # DQN model weights (last checkpoint)
│   ├── dqn_actions_best.pt     # DQN model weights (best checkpoint)
│   ├── generate_plots.py       # Script to generate result plots
│   └── report.pdf              # Full project report
```

---

## Agent Architectures

### 1. Heuristic Agent
Rule-based agent that builds an internal graph incrementally from observations and uses **BFS** to compute distances. Operates in four exclusive modes with strict priority ordering:

- **FLEE** — activated when a dangerous ghost is within the danger radius
- **CHASE_EDIBLE** — pursues edible ghosts when safe to do so
- **PILLS** — navigates toward the nearest pill or power pill
- **EXPLORE** — weighted random exploration at junctions

### 2. Monte Carlo Tree Search (MCTS) Agent
Planning agent that simulates future trajectories before acting. Follows the standard four-phase cycle: **Selection → Expansion → Simulation → Backpropagation**.

- Uses UCB1 for tree traversal with exploration constant `c = 1.4`
- Rollouts use random valid actions up to a fixed depth
- Tree is advanced after each real action; reset on maze changes
- Adaptive simulation budget reduces cost during long episodes

### 3. Reinforcement Learning Agent (Double DQN)
Learns a control policy through environment interaction using **Double DQN** with experience replay and reward shaping.

- **State:** 36-dimensional feature vector (direction, ghost distances, pill proximity, action validity, etc.)
- **Network:** `36 → 128 (ReLU) → 128 (ReLU) → 4`
- **Training:** ε-greedy exploration decaying from 1.0 to 0.05 over 500k steps
- **Replay buffer:** 200,000 transitions, warm-up of 5,000 steps
- **Reward shaping:** pill bonuses, life penalties, distance-based ghost avoidance

---

## Results

Performance comparison over **10 episodes** per agent:

| Metric | Heuristic | MCTS | RL (DQN) |
|---|---|---|---|
| Average Score | **2176** | 1487 | 1587 |
| Score Std. Dev. | 526 | 568 | 772 |
| Best Score | 3130 | 2480 | **3230** |
| Average Steps | 742 | 891 | 1374 |
| Avg. Decision Time (ms) | 0.92 | 1950.23 | **0.60** |
| Avg. Peak Memory (MB) | **0.23** | 1.64 | 0.24 |

**Key takeaways:**
- The **Heuristic agent** achieves the best average score with the lowest variance and near-zero computational overhead.
- The **MCTS agent** is the most computationally expensive (~2 seconds/step), which limits responsiveness and overall episode performance.
- The **RL agent** achieves the best individual score and the lowest inference latency, but shows higher variance across runs.

---

## Setup & Usage

### Requirements

**Java Server** (run first):
```bash
cd "Java Server and Visual Package"
java -jar EnvServer.jar        # headless mode (port 5000)
java -jar PythonPolicyPacMan.jar  # visual mode (port 5001, optional)
```

**Python environment:**
```bash
cd "Python Agent Package"
pip install -r requirements.txt  # torch, numpy, matplotlib, etc.
```

### Running an Agent

```bash
# Run the heuristic agent
python scripts/run_heuristic.py

# Train the RL agent
python scripts/train_dqn.py

# Evaluate all agents and generate plots
python generate_plots.py
```

> Refer to the individual READMEs inside `readmes/` for per-agent configuration details.

### Using Pre-trained RL Weights

Pre-trained model weights are included:
- `dqn_actions_best.pt` — best checkpoint during training
- `dqn_actions.pt` — last checkpoint
