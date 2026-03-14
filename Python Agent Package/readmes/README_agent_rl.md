# Reinforcement Learning Agent (Deep Q-Network)

**File:** `agent_rl.py`  
**Agent Type:** Deep Reinforcement Learning (Double DQN)  
**Interface:** `BaseAgent.act(obs, env=None) -> int`

---

## Description

This agent implements a **Deep Q-Learning** approach using a neural network to approximate the action-value function Q(s, a).
Learning is performed online through interaction with the environment, using experience replay, a target network,
and epsilon-greedy exploration.

Unlike the Heuristic and MCTS agents, this agent learns a policy across episodes rather than relying on fixed rules
or forward simulation.

The implementation follows a **Double DQN** setup to reduce overestimation bias and incorporates reward shaping
to accelerate learning in the Ms. Pac-Man environment.

---

## Execution Modes

### Training Mode (Default)

- `act(obs, env)` is called
- Transitions are stored in a replay buffer
- The policy network is updated periodically
- Epsilon-greedy exploration is enabled
- Target network updates occur at fixed intervals

### Inference Mode

- Enabled via `set_training_mode(False)`
- No learning is performed
- Epsilon is forced to zero
- Actions are selected greedily from the learned Q-function

---

## Core Algorithm

1. Observe current state
2. Select action using epsilon-greedy policy
3. Execute action and observe reward and next state
4. Store transition in replay buffer
5. Sample mini-batches from replay buffer
6. Update Q-network using Double DQN targets
7. Periodically update target network

---

## Neural Network Architecture

- Input dimension: 36
- Linear(36 -> 128) + ReLU
- Linear(128 -> 128) + ReLU
- Linear(128 -> 4)

Two networks with identical architecture are used:
- **Policy network** (online network)
- **Target network** (periodically synchronized)

---

## State Representation

Each observation is converted into a 36-dimensional feature vector combining:

- Pac-Man direction (one-hot encoding)
- Junction flag
- Normalized number of lives
- Normalized remaining pills and power pills
- Minimum distance to nearest dangerous ghost
- Minimum distance to nearest edible ghost
- Presence flags for danger and edible ghosts
- Per-direction danger estimates (graph-based BFS)
- Per-direction pill adjacency
- Per-direction power pill adjacency
- Per-direction neighbour existence
- Per-direction valid action mask
- Ratio of valid actions
- Presence of pills and power pills
- Bias term

No explicit grid or tile map is used. All spatial information is derived from neighbour relationships provided by the environment.

---

## Graph-Based Features

The agent builds an undirected graph incrementally from observed neighbour information:

- Nodes correspond to integer location identifiers
- Edges are added whenever neighbour relationships are observed
- The graph is reset when a maze change is detected

Breadth-First Search (BFS) over this graph is used to estimate distances to dangerous ghosts, which are included
as part of the state representation.

---

## Action Selection

Actions are selected using an epsilon-greedy policy with action masking:

- Only valid actions provided by the environment are considered
- Invalid actions are masked with very negative Q-values
- A small anti-reversal bias discourages oscillatory behaviour during both exploration and exploitation

---

## Reward Shaping

The reward signal is composed of:

- Scaled score difference
- Small per-step penalty to encourage efficiency
- Bonus for collecting pills
- Larger bonus for collecting power pills
- Strong penalty for losing a life
- Distance-based shaping relative to non-edible ghosts:
  - Penalty for moving closer when in danger
  - Bonus for increasing distance when in danger

---

## Learning Details

- Algorithm: Double DQN
- Loss function: Smooth L1 (Huber loss)
- Optimizer: Adam
- Experience replay with uniform sampling
- Gradient clipping to stabilize training

Learning starts only after a warm-up period to ensure sufficient transition diversity.

---

## Parameters and Flags

### Configuration Constants

Defined at the top of `agent_rl.py`:

| Parameter | Type | Description |
|---|---:|---|
| `GAMMA` | float | Discount factor used in the Bellman target. |
| `LR` | float | Learning rate for the Adam optimizer. |
| `BATCH_SIZE` | int | Mini-batch size sampled from the replay buffer. |
| `BUFFER_SIZE` | int | Maximum number of transitions stored in replay buffer. |
| `TARGET_UPDATE` | int | Steps between target network synchronizations. |
| `LEARN_EVERY` | int | Learning frequency after warm-up. |
| `WARMUP_STEPS` | int | Number of steps collected before learning starts. |
| `MAX_GRAD_NORM` | float | Gradient clipping threshold. |
| `EPS_START` | float | Initial epsilon for exploration. |
| `EPS_END` | float | Final epsilon after decay. |
| `EPS_DECAY_STEPS` | int | Steps over which epsilon decays linearly. |
| `STEP_PENALTY` | float | Per-step reward penalty. |
| `DEATH_PENALTY` | float | Penalty applied when a life is lost. |
| `PILL_BONUS` | float | Bonus for collecting a regular pill. |
| `POWER_BONUS` | float | Bonus for collecting a power pill. |
| `SCORE_SCALE` | float | Scale factor applied to score delta. |
| `DANGER_RADIUS` | int | Distance threshold to consider danger. |
| `DANGER_DECREASE_PENALTY` | float | Penalty when distance to danger decreases. |
| `DANGER_INCREASE_BONUS` | float | Bonus when distance to danger increases. |
| `ACTIONS` | list[int] | Set of movement actions. |
| `OPPOSITE` | dict[int,int] | Mapping of opposite directions. |
| `DEVICE` | torch.device | Computation device (`cuda` or `cpu`). |

---

### Agent Controls

- `set_training_mode(mode: bool)`  
  Enables or disables learning and exploration.

- `set_epsilon(epsilon: float)`  
  Manually sets epsilon in the range `[0, 1]`.

---

### Model Persistence

- `save(path)` stores:
  - policy and target network weights
  - optimizer state
  - training step counter
  - epsilon value

- `AgentRL.load(path)`:
  - restores all saved parameters
  - validates feature dimensionality
  - switches agent to inference mode
  - sets epsilon to zero

---

## Limitations

- Training is computationally expensive
- Performance depends strongly on reward shaping parameters
- Feature design constrains the representational capacity
- No explicit planning or long-horizon lookahead is performed

---

## BaseAgent Interface

```python
def act(self, obs, env=None) -> int
```

The returned action is one of `{-1, 0, 1, 2, 3}`.
