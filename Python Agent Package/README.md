# Ms Pac-Man Python Package

This codebase lets you work with the **MsPacman-vs-Ghosts** Java engine from Python
in two main ways:

1. **Headless training / debugging (no graphics)**  
   - Java runs `EnvServer` (your instructor provides this).  
   - Python connects with `MsPacmanEnv` (`env/ms_pacman_env.py`).  
   - You can roll out many episodes, implement RL, MCTS, heuristic agents, etc.

2. **Visual evaluation (with the classic Ms. Pac-Man window)**  
   - Java runs `PythonPolicyPacMan` (visual game).  
   - Python runs `servers/policy_server.py`, which exposes your agent over TCP.  
   - The Java visualizer queries your Python agent every frame.

Your **agent code** lives in `agents/` and is completely independent from the
communication / environment logic, so the *same agent* can be used in both modes.

---

## Rolling out games with `MsPacmanEnv` (headless)

The core pattern is:

```python
from env.ms_pacman_env import MsPacmanEnv
from agents.random_agent import RandomAgent

env = MsPacmanEnv() # Connect to the Java Environment Game Server. 
agent = RandomAgent() # The Agent you want to run.

obs = env.reset(seed=0) # Reset the Game with the Seed in Question
done = False # Has the game finished?
total_reward = 0 # Reward obtained after each game step.

while not done:
    action = agent.act(obs)      # integer in {0,1,2,3,4}
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("Final score:", obs["score"], "Total reward:", total_reward)
env.close()
```

To roll out **many** episodes and compute an average score:

```python
from env.ms_pacman_env import MsPacmanEnv
from agents.my_agent import MyAgent

num_episodes = 10
scores = []

for ep in range(num_episodes):
    env = MsPacmanEnv()
    agent = MyAgent()
    obs = env.reset(seed=ep)   # different seed per episode (optional)
    done = False
    ep_reward = 0

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward

    scores.append(obs["score"])
    env.close()

print("Average score over", num_episodes, "episodes:", sum(scores) / len(scores))
```

`scripts/test_headless.py` contains a small example like this.

---

## Folder structure

- `agents/`
  - `base_agent.py` – agent interface.
  - `random_agent.py` – random baseline.
  - `greedy_safe_agent.py` – simple heuristic baseline.
  - `mcts_agent.py` – starter MCTS skeleton (feel free to edit!).
  - `my_agent.py` – **your file**; implement your own agent here.

- `env/`
  - `action_utils.py` – direction constants (0..4).
  - `ms_pacman_env.py` – Gym-style wrapper around the Java `EnvServer`.
  - `observation_utils.py` – a set of helper functions to extract useful
    information and simple feature representations from observations.

- `servers/`
  - `policy_server.py` – wraps any agent into a TCP policy server
    for the visual Java game.

- `scripts/`
  - `test_headless.py` – quick one-episode rollout example.
  - `train_headless.py` – **Example** of multiple game runs, which can be used for 
  Reinforcement Learning training.
  - `run_visual.py` – instructions for visual mode (No code, just information).

---

## Visual mode recap

To see your agent in the classic Ms. Pac-Man window:

1. In `servers/policy_server.py`, edit `make_agent()` to construct your agent
   (and load any trained parameters, if applicable).  
2. Run:

   ```bash
   python servers/policy_server.py
   ```

3. On the Java side, run the visual game with `PythonPolicyPacMan`.

The Java visualizer will query your Python agent every frame using a JSON-based
protocol. Your agent decides the action; the Java side shows the result.
