import os
import sys
import random
from collections import defaultdict, deque

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.base_agent import BaseAgent
from env.action_utils import MOVE_DIRECTIONS, NEUTRAL, UP, RIGHT, DOWN, LEFT

# Configuration
DANGER_RADIUS = 25 # Distance to consider a ghost "dangerous"
CHASE_SAFETY_RADIUS = 40 # Min distance from danger to consider chasing edibles
MAX_BFS_DEPTH = 6000 # Max depth for BFS searches
REVERSE_PENALTY = 5.0 # Penalty for reversing direction
REVISIT_PENALTY = 0.5 # Penalty for revisiting locations
UNKNOWN_DISTANCE = 9999 # Large distance value for unknowns
EXPLORATION_CHANCE = 0.15 # Base chance to explore at junctions
EARLY_EXPLORATION_BOOST = 0.25 # Extra exploration chance in early lives
CRITICAL_DANGER_DISTANCE = 4 # Distance at which ghost is critically close
STEPS_WITHOUT_PILL_THRESHOLD = 100 # Force exploration if no pills collected
CHASE_DISTANCE_FACTOR = 1.3 # Conservative time estimation multiplier


class HeuristicAgent(BaseAgent):
    """
    Heuristic Agent with graph building and distance calculations.

    Behaviors:
    1) FLEE - escape from CLOSEST dangerous ghost within DANGER_RADIUS
    2) CHASE - pursue edible ghosts
    3) PILLS - collect nearest pill using graph distance
    4) EXPLORE - random exploration at junctions only

    """

    def __init__(self, debug=True, debug_every=1):
        # Debugging options
        self.debug = debug
        self.debug_every = max(1, int(debug_every))
        
        # Internal State
        self.step_count = 0
        self.graph = {}
        self.visit_counts = defaultdict(int)
        self.graph_initialized = False
        self.current_lives = None
        self.current_maze = None
        self.steps_this_life = 0
        self.steps_since_last_pill = 0
        self.last_pill_count = 0
        self._previous_mode = None
        self._previous_target = None
        self._mode_counter = defaultdict(int) # Track mode duration for hysteresis

    # -----------------------------
    # Helper Methods
    # -----------------------------
    
    # Get next location given action
    def _get_next_loc(self, action, pac_neighbors, pac_location):
        return int(pac_neighbors[action]) if 0 <= action < len(pac_neighbors) else pac_location

    # Check if action reverses current direction
    def _is_reverse(self, action, pac_direction):
        reverse_map = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        return reverse_map.get(pac_direction) == action

    # Pick action leading to least visited location
    def _pick_least_visited(self, available_actions, pac_neighbors, pac_location):
        best_visits = 10**9 # Large initial value
        best_action = available_actions[0] if available_actions else NEUTRAL # Default action
        
        # Evaluate each action
        for action in available_actions:
            next_loc = self._get_next_loc(action, pac_neighbors, pac_location)
            visits = self.visit_counts.get(next_loc, 0)
            if visits < best_visits:
                best_visits = visits
                best_action = action
                
        return best_action

    # Decide if we should explore randomly at junctions
    def _should_explore(self, is_junction, available_actions):
        # Only explore at junctions with multiple options
        if not is_junction or len(available_actions) < 3:
            return False
        
        # Force exploration if no pills collected recently
        if self.steps_since_last_pill > STEPS_WITHOUT_PILL_THRESHOLD:
            return True
        
        # Probabilistic exploration
        chance = EXPLORATION_CHANCE + (EARLY_EXPLORATION_BOOST if self.steps_this_life < 200 else 0)
        return random.random() < chance
    
    # Pick exploration action favoring least-visited areas
    def _pick_exploration_action(self, available_actions, pac_neighbors, pac_location):
        # Safety check
        if not available_actions:
            return NEUTRAL
        
        # Weighted random choice inversely proportional to visit counts
        weights = [1.0 / (self.visit_counts.get(self._get_next_loc(a, pac_neighbors, pac_location), 0) + 1) 
                   for a in available_actions]
        
        total = sum(weights)
        if total == 0:
            return random.choice(available_actions)
        
        # Randomly select based on weights
        rand_val = random.random() * total
        cumulative = 0
        for action, weight in zip(available_actions, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return action
            
        return available_actions[-1]

    # Add bidirectional edge in the graph
    def _add_graph_edge(self, node_a, node_b):
        if node_a >= 0 and node_b >= 0 and node_a != node_b:
            self.graph.setdefault(node_a, set()).add(node_b)
            self.graph.setdefault(node_b, set()).add(node_a)

    # Update graph from neighbor list
    def _update_graph_from_neighbors(self, node, neighbors):
        # Validity check
        if node < 0 or not isinstance(neighbors, list):
            return
        
        # Add edges to neighbors
        for neighbor in neighbors[:4]:
            try:
                neighbor_int = int(neighbor)
                if neighbor_int >= 0:
                    self._add_graph_edge(node, neighbor_int)
            except:
                pass

    # Build the complete graph from observation data
    def _build_full_graph(self, obs, info):
        # Update graph with pacman neighbors
        pac_loc = int(obs.get("pac_loc", -1))
        pac_neighbors = info.get("pac_neighbours_loc") or obs.get("pac_neighbours_loc") or []
        
        # Update graph with pacman and ghost neighbors
        if pac_loc >= 0:
            self._update_graph_from_neighbors(pac_loc, pac_neighbors)
        
        # Update graph with ghost neighbors
        for g in obs.get("ghosts") or []:
            if isinstance(g, dict):
                ghost_loc = int(g.get("loc", -1))
                # In visual mode, ghost_neighbour might be in the ghost dict directly
                ghost_neighbors = g.get("ghost_neighbour") or []
                if ghost_loc >= 0:
                    self._update_graph_from_neighbors(ghost_loc, ghost_neighbors)
        
        # Log graph initialization
        if self.debug and not self.graph_initialized:
            print(f"[Graph Init] Built graph with {len(self.graph)} nodes")
        
        self.graph_initialized = True

    # BFS to find nearest target with TRUE distance
    def _find_nearest_target(self, start, targets):
        # Validity check
        if start < 0 or not targets or start not in self.graph:
            return None, None, UNKNOWN_DISTANCE

        # BFS initialization
        queue = deque([start])
        parent = {start: None}
        distance = {start: 0}
        best_target, best_first_step, best_distance = None, None, UNKNOWN_DISTANCE

        # BFS loop
        while queue:
            current = queue.popleft()
            
            if current in targets and distance[current] < best_distance:
                best_distance = distance[current]
                best_target = current
                node = current
                while parent.get(node) is not None and parent[node] != start:
                    node = parent[node]
                best_first_step = node

            if distance[current] < MAX_BFS_DEPTH:
                for neighbor in self.graph.get(current, ()):
                    if neighbor not in distance:
                        distance[neighbor] = distance[current] + 1
                        parent[neighbor] = current
                        queue.append(neighbor)

        return best_target, best_first_step, best_distance

    # Multi-source BFS to compute distances from any source node
    def _compute_distances_from_sources(self, sources):
        distances = {}
        queue = deque()
        
        # Initialize queue with all source nodes
        for source in sources:
            if source >= 0 and source in self.graph:
                distances[source] = 0
                queue.append(source)
                
        # BFS loop
        while queue:
            current = queue.popleft()
            for neighbor in self.graph.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        return distances

    # Find the best edible ghost to chase
    def _find_best_edible_ghost(self, pac_location, edible_ghosts, edible_ghost_times, ghosts):
        # Initialize best tracking variables
        best_ghost, best_first_step, best_distance, best_time, best_score = None, None, UNKNOWN_DISTANCE, 0, -10**12
        
        # Evaluate each edible ghost
        for ghost_loc in edible_ghosts:
            target, first_step, dist = self._find_nearest_target(pac_location, {ghost_loc})
            if target is None or dist >= UNKNOWN_DISTANCE:
                continue
            
            # Find edible time
            ghost_time = next((edible_ghost_times[i] for i, g in enumerate(ghosts) 
                              if isinstance(g, dict) and int(g.get("loc", -1)) == ghost_loc 
                              and i < len(edible_ghost_times)), 0)
            
            # More aggressive time check - only skip if DEFINITELY can't reach
            time_buffer = ghost_time - int(dist * 1.0) - 5  # Less conservative!
            if time_buffer < -10:  # Allow some negative buffer
                continue
            
            # Score: prioritize close ghosts with good time buffers
            score = 1000.0 / (dist + 1) + time_buffer * 2
            if score > best_score:
                best_score, best_ghost, best_first_step, best_distance, best_time = score, ghost_loc, first_step, dist, ghost_time
        
        return best_ghost, best_first_step, best_distance, best_time

    # Extract and parse all relevant data from observation
    def _parse_observation(self, obs, info):
        return {
            'pac_location': int(obs.get("pac_loc", -1)),
            'pac_direction': int(obs.get("pac_dir", NEUTRAL)),
            'available_actions': info.get("possible_dirs") or obs.get("possible_dirs") or list(MOVE_DIRECTIONS),
            'pac_neighbors': info.get("pac_neighbours_loc") or obs.get("pac_neighbours_loc") or [],
            'is_junction': info.get("pac_loc_isjunction") or info.get("isJunction") or obs.get("pac_loc_isjunction") or obs.get("isJunction") or False,
            'ghosts': obs.get("ghosts") or [],
            'active_pills': {int(p) for p in (info.get("active_pill_loc") or obs.get("active_pill_loc") or []) if int(p) >= 0},
            'active_power': {int(p) for p in (info.get("active_power_loc") or obs.get("active_power_loc") or []) if int(p) >= 0},
            'num_pills': int(obs.get("num_active_pills", 0)),
            'num_power': int(obs.get("num_active_power_pills", 0)),
            'closest_pill_dist': info.get("closestPillDist", -1) if info.get("closestPillDist") is not None else -1,
            'edible_times': info.get("edibleTimes") or obs.get("edibleTimes") or [0, 0, 0, 0],
            'junction_locs': {int(j) for j in (info.get("junction_location") or obs.get("junction_location") or []) if int(j) >= 0},
        }

    # -----------------------------
    # Main Action Method
    # -----------------------------
    
    # Select action based on current observation
    def act(self, obs, env=None):
        """
        Select action - works with or without env parameter.
        
        Args:
            obs: Observation dictionary
            env: Optional environment (for headless mode with env.last_info)
        
        Returns:
            int: Action to take
        """
        self.step_count += 1

        # Track maze changes and life changes
        current_lives = int(obs.get("lives", 3))
        current_maze = int(obs.get("cur_maze", 0))
        
        # Reset graph on maze change
        if self.current_maze is not None and current_maze != self.current_maze:
            self.graph, self.visit_counts, self.graph_initialized = {}, defaultdict(int), False
            if self.debug:
                print(f"\n[MAZE CHANGE] Maze {self.current_maze} -> {current_maze}, resetting graph")
        
        self.current_maze = current_maze
        
        # Reset counters on life loss
        if self.current_lives is None or current_lives < self.current_lives:
            self.steps_this_life, self.steps_since_last_pill = 0, 0
            if current_lives < (self.current_lives or 3) and self.debug:
                print(f"\n[NEW LIFE] Lives: {current_lives}, boosting exploration")
            self.current_lives = current_lives
        else:
            self.steps_this_life += 1

        # Extract info - compatible with both modes
        # Priority 1: env.last_info (headless mode)
        # Priority 2: obs itself (visual mode - policy_server sends everything in obs)
        if env is not None and hasattr(env, 'last_info'):
            info = env.last_info
        else:
            # Visual mode: obs already contains all the info fields
            info = obs

        # Parse observation
        parsed = self._parse_observation(obs, info)
        pac_location = parsed['pac_location']
        pac_direction = parsed['pac_direction']
        available_actions = parsed['available_actions']
        pac_neighbors = parsed['pac_neighbors']
        is_junction = parsed['is_junction']
        ghosts = parsed['ghosts']
        
        if not available_actions:
            return NEUTRAL

        # Build graph and track visits
        self._build_full_graph(obs, info)
        self.visit_counts[pac_location] += 1

        # Get pill data
        pill_targets = parsed['active_pills']
        power_targets = parsed['active_power']
        all_pill_targets = pill_targets | power_targets
        total_pills = parsed['num_pills'] + parsed['num_power']
        closest_pill_dist = parsed['closest_pill_dist']
        
        # Track pill collection
        if self.last_pill_count > total_pills:
            self.steps_since_last_pill = 0
        else:
            self.steps_since_last_pill += 1
        self.last_pill_count = total_pills

        # Classify ghosts
        edible_ghosts = set()
        dangerous_ghosts = set()
        min_danger_distance = UNKNOWN_DISTANCE
        closest_dangerous_ghost = None
        
        for ghost in ghosts:
            if not isinstance(ghost, dict):
                continue
            ghost_loc = int(ghost.get("loc", -1))
            if ghost_loc < 0:
                continue

            if ghost.get("edible", False):
                edible_ghosts.add(ghost_loc)
            else:
                dangerous_ghosts.add(ghost_loc)
                danger_distances = self._compute_distances_from_sources(dangerous_ghosts)
                min_danger_distance = danger_distances.get(pac_location, UNKNOWN_DISTANCE)

                closest_dangerous_ghost = min(
                    dangerous_ghosts,
                    key=lambda g: danger_distances.get(g, UNKNOWN_DISTANCE),
                    default=None
                )


        # Determine strategy mode
        best_ghost, chase_dist, chase_time = None, UNKNOWN_DISTANCE, 0
        
        # PRIORITY 1: FLEE if in immediate danger
        if dangerous_ghosts and min_danger_distance <= DANGER_RADIUS:
            mode = "FLEE"
            
        # PRIORITY 2: CHASE edible ghosts ONLY if safe distance from danger
        elif edible_ghosts and min_danger_distance > CHASE_SAFETY_RADIUS:
            best_ghost, _, chase_dist, chase_time = self._find_best_edible_ghost(
                pac_location, edible_ghosts, parsed['edible_times'], ghosts)
            
            if best_ghost is not None:
                mode = "CHASE_EDIBLE"
                if self.debug and self.step_count % 20 == 0:
                    print(f"  [CHASE SAFE] target={best_ghost} dist={chase_dist} time={chase_time} | danger={min_danger_distance}")
            else:
                mode = "PILLS"
                
        # If edible ghosts exist but danger too close, skip chase
        elif edible_ghosts:
            mode = "PILLS"
            if self.debug and self.step_count % 50 == 0:
                print(f"  [CHASE UNSAFE] {len(edible_ghosts)} edible but danger too close ({min_danger_distance} < {CHASE_SAFETY_RADIUS})")
                
        # PRIORITY 3: Collect pills
        else:
            mode = "PILLS"

        self._mode_counter[mode] += 1
        chosen_action, chosen_target, chosen_distance = None, None, None

        # FLEE MODE
        if mode == "FLEE":
            danger_distances = self._compute_distances_from_sources(
                [closest_dangerous_ghost] if closest_dangerous_ghost else list(dangerous_ghosts))
            
            best_action, best_score = None, -10**12
            current_distance = danger_distances.get(pac_location, 0)
            in_corridor = len(available_actions) <= 2

            for action in available_actions:
                next_loc = self._get_next_loc(action, pac_neighbors, pac_location)
                next_distance = danger_distances.get(next_loc, 0)
                score = float(next_distance) * 10
                
                if next_distance <= CRITICAL_DANGER_DISTANCE:
                    score -= 10000
                elif next_distance < current_distance and not in_corridor:
                    score -= 500
                
                is_reverse = self._is_reverse(action, pac_direction)
                if in_corridor and is_reverse:
                    score += 50 if next_distance > current_distance + 5 else -2000
                elif is_reverse and len(available_actions) > 2:
                    score -= REVERSE_PENALTY
                
                if next_loc in parsed['junction_locs']:
                    score += 100
                if next_loc in power_targets:
                    score += 500
                score -= REVISIT_PENALTY * 0.1 * self.visit_counts.get(next_loc, 0)

                if score > best_score:
                    best_score, best_action = score, action

            chosen_action = best_action if best_action is not None else available_actions[0]
            chosen_target, chosen_distance = closest_dangerous_ghost, min_danger_distance

        # CHASE MODE
        elif mode == "CHASE_EDIBLE":
            # Use already computed values from mode determination
            target, first_step, dist, ghost_time = best_ghost, None, chase_dist, chase_time
            
            # Re-find first step (need this for direction)
            _, first_step, _ = self._find_nearest_target(pac_location, {target})
            
            chosen_target, chosen_distance = target, dist
            
            if target and first_step is not None and first_step >= 0:
                direction = next((i for i in range(4) if i < len(pac_neighbors) and int(pac_neighbors[i]) == first_step), None)
                if direction in available_actions:
                    chosen_action = direction
                    if self.debug and self.step_count % 20 == 0:
                        print(f"  [CHASE] target={target} dist={dist} time={ghost_time} action={direction}")
                else:
                    mode = "PILLS"
            else:
                mode = "PILLS"

        # PILLS MODE
        if mode == "PILLS":
            if self._should_explore(is_junction, available_actions):
                mode = "EXPLORE"
                chosen_action = self._pick_exploration_action(available_actions, pac_neighbors, pac_location)
            else:
                prioritize_power = dangerous_ghosts and min_danger_distance < 50
                best_action, best_score = None, -10**12
                
                for action in available_actions:
                    next_loc = self._get_next_loc(action, pac_neighbors, pac_location)
                    score = 2000 if next_loc in power_targets and prioritize_power else 1500 if next_loc in power_targets else 1000 if next_loc in pill_targets else 0
                    score -= REVISIT_PENALTY * self.visit_counts.get(next_loc, 0)
                    if self._is_reverse(action, pac_direction) and len(available_actions) > 1:
                        score -= REVERSE_PENALTY
                    if score > best_score:
                        best_score, best_action = score, action
                
                if best_score < 500 and all_pill_targets:
                    target, first_step, dist = self._find_nearest_target(pac_location, all_pill_targets)
                    if first_step and first_step >= 0 and len(pac_neighbors) >= 4:
                        direction = next((i for i in range(4) if int(pac_neighbors[i]) == first_step), None)
                        chosen_action = direction if direction in available_actions else (best_action or available_actions[0])
                        if direction in available_actions:
                            chosen_target, chosen_distance = target, dist
                    else:
                        chosen_action = best_action or available_actions[0]
                else:
                    chosen_action = best_action or available_actions[0]
                    chosen_distance = closest_pill_dist
        
        # Final safety check
        if chosen_action not in available_actions:
            chosen_action = available_actions[0] if available_actions else NEUTRAL

        # Debug logging
        if self.debug and self.step_count % self.debug_every == 0:
            if mode != self._previous_mode or chosen_target != self._previous_target:
                danger_str = str(min_danger_distance) if min_danger_distance < UNKNOWN_DISTANCE else "safe"
                edible_count = len(edible_ghosts)
                print(f"[{self.step_count}] {mode} | pac={pac_location} | danger={danger_str} | edible={edible_count} | target={chosen_target} (dist={chosen_distance}) | -> {chosen_action}")

        self._previous_mode = mode
        self._previous_target = chosen_target
        
        return int(chosen_action)