# MCTS Agent

**File:** `agent_mcts.py`  
**Agent Type:** Monte Carlo Tree Search  
**Interface:** `BaseAgent.act(obs, env=None) -> int`

---

## Description

This agent implements a Monte Carlo Tree Search (MCTS) algorithm.

Action selection is performed using:
- Tree traversal guided by UCB1
- Expansion over valid actions
- Random rollouts using a secondary environment
- Backpropagation of normalized rollout rewards

The agent relies on explicit forward simulation and does not perform learning across episodes.

---

## Execution Modes and Limitations

### Headless Mode (EnvServer)

- `act(obs, env)` is called
- Full MCTS logic is executed
- Rollouts are performed using a separate `MsPacmanEnv` instance
- Environment state is reconstructed deterministically using a shared seed

### Visual Mode (PythonPolicyPacMan)

- `act(obs)` is called with `env=None`
- **MCTS is not executed**
- The agent returns a random valid action

This limitation exists because state reconstruction and rollout environments are not available in visual mode.

---

## Core Algorithm

Each decision follows the standard MCTS phases:

1. **Selection**  
   Traverse the tree using UCB1:
   
   UCB1 = (Q / N) + c * sqrt(ln(N_p) / N)

2. **Expansion**  
   Add a child node corresponding to an untried valid action.

3. **Simulation (Rollout)**  
   Perform a random rollout from the expanded node using a secondary environment.

4. **Backpropagation**  
   Propagate the normalized rollout value back to the root.

---

## State Reconstruction

To evaluate a node, the agent:

1. Resets the rollout environment using `episode_seed`
2. Replays the full action history
3. Replays the action path from the root to the selected node

This guarantees consistent state reconstruction between simulations.

---

## Rollout Policy

- Actions during rollout are selected uniformly at random
- Rollouts terminate when:
  - A life is lost
  - Game over is reached
  - Maximum rollout depth is exceeded

Rollout rewards are computed as the normalized score difference.

---

## Tree Management

- The tree root advances after each selected action
- Subtrees are preserved between steps
- The entire tree is reset when the maze index changes

---

## Parameters and Flags

### Configuration Constants

Defined at the top of `agent_mcts.py`:

| Parameter | Type | Description |
|---------|------|-------------|
| `UCB_C` | float | Exploration constant used in UCB1 |
| `ROLLOUT_MAX_SCORE` | float | Maximum absolute score difference for normalization |
| `DEATH_PENALTY` | float | Rollout value returned when a life is lost |
| `OPPOSITE` | dict | Mapping of opposite movement directions |

---

### Agent Initialization Parameters

```python
MCTSAgent(
    seed=123,
    simulations=50,
    rollout_depth=30,
    rollout_port=5001
)
```

| Parameter | Type | Description |
|---------|------|-------------|
| `seed` | int | Initial random seed |
| `simulations` | int | Number of MCTS simulations per decision |
| `rollout_depth` | int | Maximum depth of random rollouts |

**Note:** The rollout environment is created automatically on port 5001.

---

## Adaptive Simulation Budget

The number of simulations per step is adjusted based on episode progress:

- Early game: full simulation budget
- Mid game: progressively reduced budget
- Late game: minimum bounded number of simulations

---

## Action Selection

After simulations:

- The action corresponding to the most visited child node is selected
- Direction reversal is avoided when possible
- If no children exist, a random valid action is chosen

---

## BaseAgent Interface

This agent extends `BaseAgent` and implements:

```python
def act(self, obs, env=None) -> int
```

- `obs`: observation dictionary
- `env`: required for headless mode; ignored in visual mode
- return value: one of `{-1, 0, 1, 2, 3}`