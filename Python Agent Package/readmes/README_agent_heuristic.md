# Heuristic Agent

**File:** `agent_heuristic.py`  
**Agent Type:** Heuristic-Based  
**Interface:** `BaseAgent.act(obs[, env]) -> int`

---

## Description

This agent implements a heuristic-based decision algorithm.

Decisions are made using predefined rules based on:
- Local environment observations
- Ghost states and distances
- Pill and power pill locations
- A graph incrementally built from observed neighbourhood information

The agent does not use learning, planning, or forward simulation.

---

## Environment Compatibility

The agent is compatible with both execution modes provided by the framework:

### Headless Mode (EnvServer)
- `act(obs, env)` is called
- Additional information is read from `env.last_info`

### Visual Mode (PythonPolicyPacMan)
- `act(obs)` is called
- All required fields are expected to be present directly in `obs`

---

## Behaviour Modes

At each step, the agent selects one behaviour mode according to a fixed priority order.

### 1. FLEE

Activated when a non-edible ghost is within `DANGER_RADIUS`.

- Computes distances from dangerous ghosts using multi-source BFS
- Selects actions that increase distance from danger
- Applies penalties for direction reversal and revisiting locations
- Considers corridor-specific behaviour
- Gives preference to junctions and power pill locations when applicable

---

### 2. CHASE_EDIBLE

Activated when:
- At least one ghost is edible, and
- No dangerous ghost is closer than `CHASE_SAFETY_RADIUS`

- Estimates distances to edible ghosts using BFS
- Considers remaining edible time
- Selects an edible ghost based on distance and time feasibility
- Falls back to pill collection if no edible ghost is suitable

---

### 3. PILLS

Default behaviour when not fleeing or chasing.

- Prioritizes power pills when danger is nearby
- Otherwise prioritizes regular pills
- Penalizes revisiting locations and reversing direction
- Uses BFS to move toward the nearest known pill when needed

---

### 4. EXPLORE

Exploration is considered only when:
- Pac-Man is at a junction, and
- No pill has been collected for a predefined number of steps, or
- A probabilistic condition is met early in a life

Exploration selects actions using visit-count–based weighting.

---

## Graph Construction

The agent maintains an internal undirected graph:

- Nodes represent observed tile locations
- Edges represent observed neighbour connections

Graph updates are performed using:
- Pac-Man neighbour information
- Ghost neighbour information (when available)

The graph is partial and reflects only observed areas of the maze.  
It is reset when the maze index changes.

---

## Distance Computation

Two BFS-based methods are used:

- **Single-target BFS**  
  Used to locate the nearest pill or edible ghost.

- **Multi-source BFS**  
  Used in FLEE mode to compute distances from one or more dangerous ghosts.

Distances are computed only over the currently constructed graph.

---

## Parameters and Flags

### Configuration Constants

Defined at the top of `agent_heuristic.py`:

| Parameter | Type | Description |
|---------|------|-------------|
| `DANGER_RADIUS` | int | Distance threshold to trigger flee behaviour |
| `CHASE_SAFETY_RADIUS` | int | Minimum safe distance required to chase edible ghosts |
| `CRITICAL_DANGER_DISTANCE` | int | Distance considered immediately dangerous |
| `MAX_BFS_DEPTH` | int | Maximum BFS expansion depth |
| `REVERSE_PENALTY` | float | Penalty applied when reversing direction |
| `REVISIT_PENALTY` | float | Penalty applied based on visit count |
| `UNKNOWN_DISTANCE` | int | Default distance value for unknown or unreachable targets |
| `EXPLORATION_CHANCE` | float | Base probability of exploration |
| `EARLY_EXPLORATION_BOOST` | float | Additional exploration probability early in a life |
| `STEPS_WITHOUT_PILL_THRESHOLD` | int | Forces exploration if exceeded |
| `CHASE_DISTANCE_FACTOR` | float | Multiplier used when estimating chase feasibility |

---

### Agent Initialization Parameters

```python
HeuristicAgent(debug=True, debug_every=1)
```

| Parameter | Type | Description |
|---------|------|-------------|
| `debug` | bool | Enables debug logging |
| `debug_every` | int | Number of steps between debug prints |

---

## BaseAgent Interface

This agent extends `BaseAgent` and implements:

```python
def act(self, obs, env=None) -> int
```

- `obs`: observation dictionary
- `env`: optional environment instance (headless mode)
- return value: one of `{-1, 0, 1, 2, 3}`
