import os
import sys
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.base_agent import BaseAgent
from env.action_utils import MOVE_DIRECTIONS, UP, RIGHT, DOWN, LEFT, NEUTRAL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration - Network and Learning
GAMMA = 0.99 # Discount factor
LR = 1e-3 # Learning rate
BATCH_SIZE = 64 # Training batch size
BUFFER_SIZE = 200000 # Replay buffer capacity
TARGET_UPDATE = 2000 # Steps between target updates
LEARN_EVERY = 4 # Learning frequency
WARMUP_STEPS = 5000 # Steps before learning
MAX_GRAD_NORM = 5.0 # Gradient clipping

# Configuration - Exploration
EPS_START = 1.0 # Initial epsilon
EPS_END = 0.05 # Final epsilon
EPS_DECAY_STEPS = 500000 # Decay duration

# Configuration - Reward Shaping
STEP_PENALTY = -0.01 # Per-step penalty
DEATH_PENALTY = -10.0 # Death penalty
PILL_BONUS = 0.5 # Pill collection bonus
POWER_BONUS = 1.5 # Power pill bonus
SCORE_SCALE = 0.02 # Score delta scale

# Configuration - Safety
DANGER_RADIUS = 20 # Danger distance threshold
DANGER_DECREASE_PENALTY = -0.15 # Moving closer penalty
DANGER_INCREASE_BONUS = 0.05 # Moving away bonus

ACTIONS = list(MOVE_DIRECTIONS)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class DQN(nn.Module):
    """Feedforward Neural Network for DQN"""
    
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), # Hidden layer 1
            nn.Linear(128, 128), nn.ReLU(), # Hidden layer 2
            nn.Linear(128, output_dim), # Output layer
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Add experience tuple to buffer
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Sample a batch of experiences
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class AgentRL(BaseAgent):
    """
    Deep Q-Learning Agent
    
    Uses Double DQN with experience replay to learn optimal policies
    
    Features:
    - Action masking for valid moves only
    - Graph-based danger distance computation
    - Reward shaping for faster learning
    - Epsilon-greedy exploration with decay
    """
    
    def __init__(self):
        # Graph tracking for BFS-based features
        self.graph = {}
        self.graph_initialized = False
        self.current_maze = None
        
        # Training state
        self.training_mode = True
        self.steps = 0
        self.epsilon = EPS_START
        
        # Previous transition storage for experience replay
        self.prev_obs = None
        self.prev_info = None
        self.prev_state = None
        self.prev_action = None
        self.last_action = None
        
        # Neural networks (policy + target)
        self.input_dim = 36
        self.policy_net = DQN(self.input_dim, 4).to(DEVICE) # Policy network
        self.target_net = DQN(self.input_dim, 4).to(DEVICE) # Target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync weights
        self.target_net.eval() # Target network in eval mode
        
        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR) # Adam optimizer
        self.memory = ReplayBuffer(BUFFER_SIZE) # Experience replay buffer

    # -----------------------------
    # Mode Control and Persistence
    # -----------------------------
    
    # Toggle between training and inference modes
    def set_training_mode(self, mode):
        self.training_mode = mode
        self.policy_net.train() if mode else self.policy_net.eval()

    # Set epsilon value for exploration
    def set_epsilon(self, epsilon):
        self.epsilon = float(max(0.0, min(1.0, epsilon)))

    # Save model checkpoint to disk
    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "steps": self.steps,
            "epsilon": self.epsilon,
            "input_dim": self.input_dim,
        }, path)

    # Load model checkpoint from disk
    @staticmethod
    def load(path, verbose=True):
        agent = AgentRL()
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        # Validate feature dimensions match
        if int(checkpoint.get("input_dim", agent.input_dim)) != agent.input_dim:
            raise ValueError(f"Model input_dim mismatch: checkpoint={checkpoint.get('input_dim')} vs agent={agent.input_dim}")
        
        # Load network weights
        agent.policy_net.load_state_dict(checkpoint["policy"])
        agent.target_net.load_state_dict(checkpoint.get("target", checkpoint["policy"]))
        if "optim" in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint["optim"])
            except Exception:
                pass
        
        # Restore training state
        agent.steps = int(checkpoint.get("steps", 0))
        agent.epsilon = float(checkpoint.get("epsilon", EPS_END))
        agent.set_training_mode(False)
        agent.epsilon = 0.0
        
        if verbose:
            print(f"[AgentRL] Loaded from {path} (steps={agent.steps})")
        return agent

    # -----------------------------
    # Helper Methods
    # -----------------------------
    
    # Safely convert value to integer with default fallback
    def _safe_int(self, value, default=-1):
        try:
            return default if value is None else int(value)
        except Exception:
            return default

    # Safely convert value to boolean with default fallback
    def _safe_bool(self, value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "t")
        return default

    # Safely convert list elements to integers
    def _safe_list_int(self, value):
        return [self._safe_int(v) for v in value] if isinstance(value, list) else []

    # Clamp value to [0, 1] range for normalization
    def _clamp01(self, value):
        return max(0.0, min(1.0, value))

    # -----------------------------
    # Graph Building
    # -----------------------------
    
    # Add bidirectional edge between two nodes in graph
    def _add_edge(self, node_a, node_b):
        if node_a >= 0 and node_b >= 0 and node_a != node_b:
            self.graph.setdefault(node_a, set()).add(node_b)
            self.graph.setdefault(node_b, set()).add(node_a)

    # Build graph from observation and info data
    def _build_graph(self, obs, info):
        # Update graph with pacman neighbors
        pac_loc = self._safe_int(obs.get("pac_loc"))
        pac_neighbors = self._safe_list_int(info.get("pac_neighbours_loc") or obs.get("pac_neighbours_loc"))
        
        # Add edges for pacman neighbors
        if pac_loc >= 0:
            for neighbor in pac_neighbors[:4]:
                if neighbor >= 0:
                    self._add_edge(pac_loc, neighbor)
        
        # Update graph with ghost neighbors
        ghosts_raw = obs.get("ghosts") or []
        ghost_neighbors_all = info.get("ghost_neighbours") or obs.get("ghost_neighbours")
        
        # Add edges for ghost neighbors
        for i, ghost in enumerate(ghosts_raw[:4] if isinstance(ghosts_raw, list) else []):
            if not isinstance(ghost, dict):
                continue
            ghost_loc = self._safe_int(ghost.get("loc"))
            ghost_neighbors = self._safe_list_int(ghost.get("ghost_neighbour"))
            if not ghost_neighbors and isinstance(ghost_neighbors_all, list) and i < len(ghost_neighbors_all):
                ghost_neighbors = self._safe_list_int(ghost_neighbors_all[i])
            if ghost_loc >= 0:
                for neighbor in ghost_neighbors[:4]:
                    if neighbor >= 0:
                        self._add_edge(ghost_loc, neighbor)
        
        if self.graph:
            self.graph_initialized = True

    # Multi-source BFS to compute distances from dangerous ghosts
    def _multi_source_bfs_dist(self, sources, max_depth=40):
        distance = {}
        queue = deque()
        
        # Initialize queue with all source nodes
        for source in sources:
            if source in self.graph:
                distance[source] = 0
                queue.append(source)
        
        # BFS traversal with depth limit
        while queue:
            current = queue.popleft()
            current_dist = distance[current]
            if current_dist >= max_depth:
                continue
            for neighbor in self.graph.get(current, ()):
                if neighbor not in distance:
                    distance[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    
        return distance

    # -----------------------------
    # Observation Parsing
    # -----------------------------
    
    # Extract info dict from environment or observation
    def _get_info(self, obs, env):
        if env is not None and hasattr(env, "last_info") and isinstance(env.last_info, dict):
            return env.last_info
        
        return {}

    # Get list of valid actions from observation
    def _get_valid_actions(self, obs, info):
        valid = self._safe_list_int(info.get("possible_dirs") or obs.get("possible_dirs"))
        valid = [a for a in (valid or ACTIONS) if a in ACTIONS]
        
        return valid or list(ACTIONS)

    # Get pacman's neighbor locations in all four directions
    def _get_pac_neighbors(self, obs, info):
        neighbors = self._safe_list_int(info.get("pac_neighbours_loc") or obs.get("pac_neighbours_loc"))
        
        return (neighbors + [-1, -1, -1, -1])[:4]

    # Get sets of active pill and power pill locations
    def _get_active_pills(self, obs, info):
        pills = self._safe_list_int(info.get("active_pill_loc") or obs.get("active_pill_loc"))
        power = self._safe_list_int(info.get("active_power_loc") or obs.get("active_power_loc"))
        
        return set(p for p in pills if p >= 0), set(p for p in power if p >= 0)

    # Parse ghost information from observation
    def _parse_ghosts(self, obs, info):
        ghosts_raw = obs.get("ghosts") or []
        ghost_distances = info.get("ghostDistances") or obs.get("ghostDistances")
        parsed = []
        
        # Parse each ghost's data
        for i in range(4):
            ghost = ghosts_raw[i] if isinstance(ghosts_raw, list) and i < len(ghosts_raw) else None
            if not isinstance(ghost, dict):
                parsed.append({"id": i, "loc": -1, "edible": False, "dist": 999})
                continue
            
            ghost_loc = self._safe_int(ghost.get("loc"))
            edible = self._safe_bool(ghost.get("edible", False))
            
            # Determine distance using multiple sources
            dist = (
                self._safe_int(ghost_distances[i], 999)
                if isinstance(ghost_distances, list) and i < len(ghost_distances)
                else 999
            )
            
            parsed.append({"id": i, "loc": ghost_loc, "edible": edible, "dist": dist})
        return parsed

    # -----------------------------
    # Feature Extraction (36 dimensions)
    # -----------------------------
    
    # Extract feature vector from observation for neural network input
    def _extract_features(self, obs, info):
        # Basic state information
        pac_loc = self._safe_int(obs.get("pac_loc"))
        pac_dir = self._safe_int(obs.get("pac_dir"), NEUTRAL)
        lives = self._safe_int(obs.get("lives"), 3)
        num_pills = self._safe_int(obs.get("num_active_pills"))
        num_power = self._safe_int(obs.get("num_active_power_pills") or obs.get("num_active_power"))
        
        # Junction detection from info or obs
        is_junc = self._safe_bool(info.get("pac_loc_isjunction"))
        if is_junc is None:
            is_junc = self._safe_bool(obs.get("pac_loc_isjunction"))
        
        # Get neighbors and pill locations
        pac_neighbors = self._get_pac_neighbors(obs, info)
        pills_set, power_set = self._get_active_pills(obs, info)
        
        # Parse ghost data and compute distances
        ghosts = self._parse_ghosts(obs, info)
        min_danger = min((g["dist"] for g in ghosts if not g["edible"]), default=999)
        min_edible = min((g["dist"] for g in ghosts if g["edible"]), default=999)
        
        # Compute per-direction danger distance using graph BFS
        danger_next = [1.0, 1.0, 1.0, 1.0]
        if self.graph_initialized:
            danger_sources = set(g["loc"] for g in ghosts if (not g["edible"]) and g["loc"] >= 0)
            if danger_sources:
                distance_map = self._multi_source_bfs_dist(danger_sources, max_depth=40)
                for direction in ACTIONS:
                    neighbor = pac_neighbors[direction]
                    if neighbor >= 0:
                        danger_next[direction] = self._clamp01(distance_map.get(neighbor, 40) / 40.0)
        
        # Check per-direction pill and power pill adjacency
        pill_next = [1.0 if pac_neighbors[d] in pills_set else 0.0 for d in ACTIONS]
        power_next = [1.0 if pac_neighbors[d] in power_set else 0.0 for d in ACTIONS]
        
        # Pac direction one-hot encoding
        pac_dir_oh = [1.0 if pac_dir == d else 0.0 for d in ACTIONS]
        
        # Assemble all features into vector
        features = []
        features.extend(pac_dir_oh) # 4 - direction one-hot
        features.append(1.0 if is_junc else 0.0) # 1 - junction flag
        features.append(self._clamp01(lives / 3.0)) # 1 - normalized lives
        features.append(self._clamp01(num_pills / 220.0)) # 1 - normalized pills
        features.append(self._clamp01(num_power / 4.0)) # 1 - normalized power
        features.append(self._clamp01(min_danger / 40.0)) # 1 - min danger distance
        features.append(self._clamp01(min_edible / 40.0)) # 1 - min edible distance
        features.append(1.0 if min_edible < 999 else 0.0) # 1 - edible ghost flag
        features.append(1.0 if min_danger <= DANGER_RADIUS else 0.0) # 1 - danger flag
        features.extend(danger_next) # 4 - per-direction danger
        features.extend(pill_next) # 4 - per-direction pills
        features.extend(power_next) # 4 - per-direction power
        features.extend([1.0 if pac_neighbors[d] >= 0 else 0.0 for d in ACTIONS]) # 4 - neighbor exists
        
        # Per-direction valid action flags
        valid_actions = self._get_valid_actions(obs, info)
        features.extend([1.0 if d in valid_actions else 0.0 for d in ACTIONS]) # 4 - valid actions
        features.append(self._clamp01(len(valid_actions) / 4.0)) # 1 - valid action ratio
        features.append(1.0 if pills_set else 0.0) # 1 - has pills
        features.append(1.0 if power_set else 0.0) # 1 - has power
        features.append(1.0) # 1 - bias term
        
        return np.array(features, dtype=np.float32)

    # -----------------------------
    # Reward Shaping
    # -----------------------------
    
    # Compute shaped reward from state transition
    def _compute_reward(self, prev_obs, obs, prev_info, info):
        # Raw score delta scaled down
        reward = float(self._safe_int(obs.get("score")) - self._safe_int(prev_obs.get("score"))) * SCORE_SCALE
        
        # Small per-step penalty to encourage efficiency
        reward += STEP_PENALTY
        
        # Bonus for collecting pills
        pills_prev = self._safe_int(prev_obs.get("num_active_pills") or prev_obs.get("num_pills"))
        pills_curr = self._safe_int(obs.get("num_active_pills") or obs.get("num_pills"))
        if pills_curr < pills_prev:
            reward += PILL_BONUS
        
        # Bonus for collecting power pills
        power_prev = self._safe_int(prev_obs.get("num_active_power_pills") or prev_obs.get("num_active_power"))
        power_curr = self._safe_int(obs.get("num_active_power_pills") or obs.get("num_active_power"))
        if power_curr < power_prev:
            reward += POWER_BONUS
        
        # Heavy penalty for death
        if self._safe_int(obs.get("lives"), 3) < self._safe_int(prev_obs.get("lives"), 3):
            reward += DEATH_PENALTY
        
        # Danger-based shaping: penalize moving closer, reward moving away
        ghosts_prev = self._parse_ghosts(prev_obs, prev_info)
        ghosts_curr = self._parse_ghosts(obs, info)
        danger_prev = min((g["dist"] for g in ghosts_prev if not g["edible"]), default=999)
        danger_curr = min((g["dist"] for g in ghosts_curr if not g["edible"]), default=999)
        
        # Only apply danger shaping when within danger radius
        if danger_prev <= DANGER_RADIUS or danger_curr <= DANGER_RADIUS:
            if danger_curr < danger_prev:
                reward += DANGER_DECREASE_PENALTY
            elif danger_curr > danger_prev:
                reward += DANGER_INCREASE_BONUS
        
        return float(reward)

    # -----------------------------
    # Learning (Double DQN)
    # -----------------------------
    
    # Perform one gradient descent step using sampled batch
    def _learn_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample random batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states_t = torch.from_numpy(states).float().to(DEVICE)
        actions_t = torch.from_numpy(actions).long().to(DEVICE)
        rewards_t = torch.from_numpy(rewards).float().to(DEVICE)
        next_states_t = torch.from_numpy(next_states).float().to(DEVICE)
        dones_t = torch.from_numpy(dones).float().to(DEVICE)
        
        # Compute current Q(s,a) values
        q_values = self.policy_net(states_t)
        q_values_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double DQN approach
        with torch.no_grad():
            # Select best action using policy network
            next_q_policy = self.policy_net(next_states_t)
            next_actions = torch.argmax(next_q_policy, dim=1)
            # Evaluate action using target network
            next_q_target = self.target_net(next_states_t)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Bellman target with done mask
            target_values = rewards_t + (1.0 - dones_t) * GAMMA * next_q_values
        
        # Compute loss and optimize
        loss = nn.SmoothL1Loss()(q_values_action, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

    # -----------------------------
    # Action Selection
    # -----------------------------
    
    # Select action using epsilon-greedy policy with action masking
    def _select_action(self, state, valid_actions):
        # Epsilon-greedy exploration (training mode only)
        if self.training_mode and random.random() < self.epsilon:
            # Small anti-reversal bias during exploration
            if self.last_action is not None and len(valid_actions) > 1:
                opposite = OPPOSITE.get(self.last_action, None)
                if opposite in valid_actions and random.random() < 0.7:
                    # Avoid reversing 70% of the time
                    choices = [a for a in valid_actions if a != opposite]
                    return random.choice(choices) if choices else random.choice(valid_actions)
            return random.choice(valid_actions)
        
        # Greedy action selection based on Q-values
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Mask out invalid actions with very negative values
        masked_q = np.full((4,), -1e9, dtype=np.float32)
        for action in valid_actions:
            masked_q[action] = q_values[action]
        best_action = int(np.argmax(masked_q))
        
        # Small anti-reversal bias in greedy selection too
        if self.last_action is not None and best_action == OPPOSITE.get(self.last_action, -999) and len(valid_actions) > 1:
            sorted_actions = np.argsort(masked_q)[::-1]
            for candidate in sorted_actions:
                candidate = int(candidate)
                if candidate in valid_actions and candidate != best_action:
                    # Pick second-best if close in value
                    if masked_q[candidate] >= masked_q[best_action] - 0.05:
                        return candidate
                    break
        
        return best_action

    # Update epsilon using linear decay schedule
    def _update_epsilon(self):
        if not self.training_mode or self.steps <= 0:
            self.epsilon = 0.0 if not self.training_mode else EPS_START
            return
        # Linear decay from EPS_START to EPS_END
        fraction = min(1.0, self.steps / float(EPS_DECAY_STEPS))
        self.epsilon = max(EPS_END, EPS_START + fraction * (EPS_END - EPS_START))

    # -----------------------------
    # Main Action Method
    # -----------------------------
    
    # Select action based on current observation
    def act(self, obs, env=None):
        self.steps += 1
        info = self._get_info(obs, env)
        
        # Detect maze change and reset graph
        current_maze = self._safe_int(obs.get("cur_maze"))
        if self.current_maze is None or current_maze != self.current_maze:
            self.current_maze = current_maze
            self.graph.clear()
            self.graph_initialized = False
            # Reset transition data to avoid cross-maze learning
            self.prev_obs = self.prev_info = self.prev_state = self.prev_action = self.last_action = None
        
        # Build graph incrementally
        self._build_graph(obs, info)
        
        # Extract current state features
        state = self._extract_features(obs, info)
        
        # Store transition and learn (training mode only)
        if self.training_mode and all([self.prev_obs, self.prev_state is not None, self.prev_action is not None]):
            # Check if episode is done
            done = self._safe_bool(obs.get("game_over"))
            
            # Compute shaped reward and add to replay buffer
            reward = self._compute_reward(self.prev_obs, obs, self.prev_info or {}, info)
            self.memory.push(self.prev_state, int(self.prev_action), float(reward), state, bool(done))
            
            # Learn periodically after warmup period
            if self.steps > WARMUP_STEPS and (self.steps % LEARN_EVERY == 0):
                self._learn_step()
            
            # Update target network periodically
            if self.steps % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Clear transition data on episode end
            if done:
                self.prev_obs = self.prev_info = self.prev_state = self.prev_action = self.last_action = None
        
        # Get valid actions
        valid_actions = self._get_valid_actions(obs, info)
        if not valid_actions:
            return NEUTRAL
        
        # Update epsilon and select action
        self._update_epsilon()
        action = self._select_action(state, valid_actions)
        
        # Store transition data for next step (training only)
        if self.training_mode:
            self.prev_obs, self.prev_info, self.prev_state, self.prev_action = obs, info, state, action
        
        self.last_action = action
        
        return int(action)