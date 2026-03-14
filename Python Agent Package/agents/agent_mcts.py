from __future__ import annotations
import os
import sys
import math
import random
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.base_agent import BaseAgent
from env.action_utils import MOVE_DIRECTIONS, UP, DOWN, LEFT, RIGHT
from env.ms_pacman_env import MsPacmanEnv

# Configuration
UCB_C = 1.4 # Exploration constant for UCB1
ROLLOUT_MAX_SCORE = 300.0 # Max score delta for normalization
DEATH_PENALTY = -10.0 # Penalty for losing a life
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT} # Opposite directions


class MCTSNode:
    """Node in the MCTS tree representing a game state."""
    
    def __init__(self, parent: Optional["MCTSNode"], action: Optional[int]):
        self.parent = parent
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.value = 0.0

    # Calculate UCB1 value: exploitation + exploration
    def ucb1(self, c: float = UCB_C) -> float:
        # Avoid division by zero
        if self.visits == 0:
            return float("inf")
        
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration

    # Return the child with highest UCB1 value
    def best_child(self, c: float = UCB_C) -> Optional["MCTSNode"]:
        return max(self.children.values(), key=lambda n: n.ucb1(c)) if self.children else None

    # Expand this node by adding a child for an untried action
    def expand(self, valid_actions: List[int], rng: random.Random) -> "MCTSNode":
        # Find untried actions
        untried = [a for a in valid_actions if a not in self.children]
        
        if not untried:
            return self
        
        # Randomly select an untried action to expand
        action = rng.choice(untried)
        child = MCTSNode(parent=self, action=action)
        self.children[action] = child
        
        return child

    # Check if all valid actions have corresponding child nodes
    def is_fully_expanded(self, valid_actions: List[int]) -> bool:
        return all(a in self.children for a in valid_actions)


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search Agent.
    Uses UCB1 selection with random rollouts.
    
    Args:
        seed: Random seed for reproducibility.
        simulations: Number of MCTS simulations per move.
        rollout_depth: Maximum depth of rollout simulations.
    """

    def __init__(self, seed: int = 123, simulations: int = 50, rollout_depth: int = 30):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.history: List[int] = []
        self.episode_seed: Optional[int] = None
        self.root = MCTSNode(parent=None, action=None)
        self.rng = random.Random(seed)
        
        # Rollout environment on port 5001
        self.rollout_env = MsPacmanEnv(port=5001)
        
        # Track maze changes
        self.prev_maze: Optional[int] = None
        self.step_count = 0

    # -----------------------------
    # Episode Management
    # -----------------------------
    
    # Set episode seed (must match main game for state reconstruction)
    def set_episode_seed(self, seed: int):
        self.episode_seed = seed
        self.history.clear()
        self.root = MCTSNode(None, None)
        self.rng.seed(seed)
        self.prev_maze = None
        self.cache_valid = False

    # -----------------------------
    # Helper Methods
    # -----------------------------

    # Extract valid actions from environment info
    def _get_valid_actions(self, info: Optional[Dict[str, Any]]) -> List[int]:
        if not info:
            return MOVE_DIRECTIONS[:]
        
        # Different env versions may use different keys
        valid = info.get("possible_dirs") or info.get("pacAvailableMoves") or []
        
        # Ensure valid actions are a list of known directions
        if not isinstance(valid, list):
            return MOVE_DIRECTIONS[:]
        
        # Filter to known move directions
        filtered = [a for a in valid if a in MOVE_DIRECTIONS]
        
        return filtered if filtered else MOVE_DIRECTIONS[:]
    
    # Reconstruct game state for a given node
    def _reconstruct_state(self, node: MCTSNode) -> tuple[MsPacmanEnv, Dict[str, Any], Dict[str, Any]]:
        env = self.rollout_env
        
        # Reset environment to episode start
        obs = env.reset(self.episode_seed)
        info = env.last_info

        # Replay full history to reach current node
        for action in self.history:
            obs, _, done, _ = env.step(action)
            info = env.last_info
            if done:
                break
        
        # If at root, return current state
        if node.parent is None:
            return env, obs, info
        
        # Replay actions to reach target node
        path = []
        current = node
        
        # Build action path from root to target node
        while current and current.parent:
            path.append(current.action)
            current = current.parent
            
        path.reverse()

        # Execute actions to reach target node state
        for action in path:
            obs, _, done, _ = env.step(action)
            info = env.last_info
            if done:
                break

        return env, obs, info

    # Rollout simulation
    def _rollout(self, env: MsPacmanEnv, obs: Dict[str, Any], info: Dict[str, Any]) -> float:
        # Perform a random rollout from the given state.
        if obs is None or obs.get("game_over", False):
            return DEATH_PENALTY

        #  Extract starting conditions
        start_score = obs.get("score", 0)
        start_lives = obs.get("lives", 0)
        depth = 0
        done = False
        current_obs = obs
        current_info = info

        # Randomly play until depth limit or game over
        while not done and depth < self.rollout_depth:
            valid_actions = self._get_valid_actions(current_info)
            action = self.rng.choice(valid_actions)
            current_obs, _, done, _ = env.step(action)
            current_info = env.last_info

            if current_obs.get("game_over", False) or current_obs.get("lives", 0) < start_lives:
                return DEATH_PENALTY
            depth += 1
        
        # Compute normalized score delta
        delta_score = current_obs.get("score", 0) - start_score
        delta_score = max(-ROLLOUT_MAX_SCORE, min(ROLLOUT_MAX_SCORE, float(delta_score)))
        
        return delta_score / ROLLOUT_MAX_SCORE

    # -----------------------------
    # Main MCTS Algorithm
    # -----------------------------

    def act(self, obs: Dict[str, Any], env=None) -> int:
        self.step_count += 1
        
        if self.episode_seed is None:
            self.set_episode_seed(123)

        # Reset tree if maze changed
        current_maze = obs.get("cur_maze")
        if self.prev_maze is not None and current_maze != self.prev_maze:
            self.root = MCTSNode(None, None)
            self.cache_valid = False # Invalidate cache on maze change
            if self.step_count % 10 == 0:
                print(f"[MCTS] Maze changed: {self.prev_maze} -> {current_maze}, resetting tree")
        self.prev_maze = current_maze
        
        # Invalidate cache at start of new decision (will be rebuilt on first _reconstruct_state call)
        self.cache_valid = False

        # Adaptive simulations based on game progress to maintain speed
        # Adjusted to maintain better decision quality late game
        if len(self.history) < 150:
            num_sims = self.simulations # Full simulations early
        elif len(self.history) < 400:
            num_sims = max(self.simulations // 2, 8) # Half, minimum 8
        elif len(self.history) < 700:
            num_sims = max(self.simulations // 3, 6) # 1/3, minimum 6
        else:
            num_sims = max(self.simulations // 4, 5) # 1/4, minimum 5
        
        # Run MCTS simulations
        MAX_TREE_DEPTH = 3
        
        for sim in range(num_sims):
            # PHASE 1: SELECTION
            node = self.root
            env, current_obs, info = self._reconstruct_state(node)
            
            if current_obs.get("game_over", False):
                break
            
            tree_depth = 0
            
            # Traverse tree using UCB1
            while tree_depth < MAX_TREE_DEPTH:
                valid_actions = self._get_valid_actions(info)
                
                # PHASE 2: EXPANSION
                if not node.is_fully_expanded(valid_actions):
                    node = node.expand(valid_actions, self.rng)
                    if node.action is not None:
                        current_obs, _, done, _ = env.step(node.action)
                        info = env.last_info
                    break
                
                # Select best child
                best = node.best_child()
                if best is None:
                    node = node.expand(valid_actions, self.rng)
                    if node.action is not None:
                        current_obs, _, done, _ = env.step(node.action)
                        info = env.last_info
                    break
                    
                node = best
                current_obs, _, done, _ = env.step(node.action)
                info = env.last_info
                tree_depth += 1
                
                if done or current_obs.get("game_over", False):
                    break

            # PHASE 3: SIMULATION (Rollout)
            value = DEATH_PENALTY if current_obs.get("game_over", False) else self._rollout(env, current_obs, info)

            # PHASE 4: BACKPROPAGATION
            current = node
            while current is not None:
                current.visits += 1
                current.value += value
                current = current.parent

        # Debug output
        if self.step_count % 50 == 0:
            children_info = ""
            if self.root.children:
                children_info = " | children: " + ", ".join(
                    f"{a}:{n.visits}({n.value/n.visits:.2f})" if n.visits > 0 else f"{a}:0"
                    for a, n in sorted(self.root.children.items())
                )
            print(f"[MCTS] step={self.step_count} | sims_used={num_sims}/{self.simulations} | history={len(self.history)}{children_info}")

        # ACTION SELECTION: Choose most-visited child
        if not self.root.children:
            action = self.rng.choice(MOVE_DIRECTIONS)
        else:
            ranked = sorted(self.root.children.values(), key=lambda n: n.visits, reverse=True)
            pac_dir = obs.get("pac_dir", None)
            action = None
            
            # Avoid reversal if possible
            for candidate in ranked:
                if pac_dir in MOVE_DIRECTIONS and candidate.action == OPPOSITE.get(pac_dir):
                    continue
                action = candidate.action
                break

            if action is None:
                action = ranked[0].action

        # Update history and advance tree root
        self.history.append(action)
        
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode(None, None)

        return action

    # Clean up resources
    def close(self):
        if self.rollout_env:
            self.rollout_env.close()
            self.rollout_env = None