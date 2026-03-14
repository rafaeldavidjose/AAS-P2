import os
import sys
import json
import socket
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.action_utils import UP, RIGHT, DOWN, LEFT, NEUTRAL  # keep imports for compatibility


def _i(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        return int(x)
    except Exception:
        return default


def _b(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    return default


def _li(x: Any) -> List[int]:
    if isinstance(x, list):
        return [_i(v, 0) for v in x]
    return []


class MsPacmanEnv:
    """
    JSON-based EnvServer client.

    Protocol:
      RESET [seed] -> one JSON line (state)
      STEP <a>     -> one JSON line (step result + state)
      QUIT         -> (optional) one JSON line

    IMPORTANT: We do NOT branch on message type. We always parse JSON and
    take reward/done if present; otherwise default to 0/False.
    """

    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.fin = None
        self.fout = None
        self.last_info: Dict[str, Any] = {}
        self._connect()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        # wrap as text files for easy line I/O
        self.fin = s.makefile("r", encoding="utf-8")
        self.fout = s.makefile("w", encoding="utf-8")
        self.sock = s
        # print(f"[MsPacmanEnv] Connected to {self.host}:{self.port}")

    def _send_line(self, line: str) -> str:
        self.fout.write(line + "\n")
        self.fout.flush()
        resp = self.fin.readline()
        if not resp:
            raise RuntimeError("EnvServer closed the connection")
        return resp.strip()

    # ------------------------------------------------------------------ API

    def reset(self, seed=None):
        if seed is None:
            resp = self._send_line("RESET")
        else:
            resp = self._send_line(f"RESET {int(seed)}")

        obs, reward, done, info = self._parse_json_line(resp)
        # reward/done ignored on reset; returned obs is what callers expect
        self.last_info = info
        return obs

    def step(self, action: int):
        resp = self._send_line(f"STEP {int(action)}")
        obs, reward, done, info = self._parse_json_line(resp)
        self.last_info = info
        # keep signature identical to your old env
        return obs, reward, done, {}

    def close(self):
        try:
            self._send_line("QUIT")
        except Exception:
            pass
        try:
            if self.fin:
                self.fin.close()
            if self.fout:
                self.fout.close()
            if self.sock:
                self.sock.close()
        finally:
            self.fin = self.fout = self.sock = None

    # ------------------------------------------------------------------ parsing (JSON)

    def _parse_json_line(self, line: str) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """
        Parse one JSON object per line.
        Does NOT check type/prefix. Works for both reset and step replies.

        Returns:
          obs (legacy-compatible dict),
          reward (int),
          done (bool),
          info (rich dict: arrays/topology/raw json)
        """
        obj = json.loads(line)

        # reward/done may be absent on reset -> defaults
        reward = _i(obj.get("reward", 0), 0)
        done = _b(obj.get("done", False), False)

        # Core fields (normalize to your existing obs keys)
        score = _i(obj.get("score", 0), 0)
        level_time = _i(obj.get("level_time", 0), 0)
        total_time = _i(obj.get("total_time", 0), 0)
        lives = _i(obj.get("lives", 0), 0)
        cur_level = _i(obj.get("cur_level", 0), 0)
        cur_maze = _i(obj.get("cur_maze", 0), 0)
        pac_loc = _i(obj.get("pac_loc", -1), -1)
        pac_dir = _i(obj.get("pac_dir", NEUTRAL), NEUTRAL)

        # Pills (your JSON uses num_active_power; old env used num_active_power_pills)
        num_pills = _i(obj.get("num_active_pills", 0), 0)
        num_power = _i(obj.get("num_active_power", obj.get("num_active_power_pills", 0)), 0)

        # Game over flag (may not be present; infer from done)
        game_over = _b(obj.get("gameOver", obj.get("game_over", done)), done)

        # Ghosts
        ghosts_in = obj.get("ghosts") or []
        ghosts: List[Dict[str, Any]] = []
        for g in range(4):
            if g < len(ghosts_in) and isinstance(ghosts_in[g], dict):
                gi = ghosts_in[g]
                ghosts.append({
                    "id": g,
                    "loc": _i(gi.get("loc", -1), -1),
                    "dir": _i(gi.get("dir", NEUTRAL), NEUTRAL),
                    "edible": _b(gi.get("edible", False), False),
                })
            else:
                ghosts.append({"id": g, "loc": -1, "dir": NEUTRAL, "edible": False})

        # Legacy-compatible observation dict (matches your previous shape)
        obs = {
            "score": score,
            "level_time": level_time,
            "total_time": total_time,
            "lives": lives,
            "cur_level": cur_level,
            "cur_maze": cur_maze,
            "pac_loc": pac_loc,
            "pac_dir": pac_dir,
            "ghosts": ghosts,
            "num_active_pills": num_pills,
            "num_active_power_pills": num_power,
            "game_over": game_over,
        }

        # Rich info (new fields, raw json, topology)
        info: Dict[str, Any] = {
            "raw": obj,

            # coords
            "pac_x": _i(obj.get("pac_getX", 0), 0),
            "pac_y": _i(obj.get("pac_getY", 0), 0),

            # arrays you already send
            "active_pill_loc": _li(obj.get("active_pill_loc")),
            "active_power_loc": _li(obj.get("active_power_loc")),
            "possible_dirs": _li(obj.get("possible_dirs")),
            "pac_neighbours_loc": _li(obj.get("pac_neighbours_loc")),
            "pac_loc_isjunction": _b(obj.get("pac_loc_isjunction", False), False),
            "junction_location": _li(obj.get("junction_location")),
            "pill_location": _li(obj.get("pill_location")),
            "power_location": _li(obj.get("power_location")),

            # convenience aliases (names you wanted earlier)
            "pacAvailableMoves": _li(obj.get("possible_dirs")),
            "isJunction": _b(obj.get("pac_loc_isjunction", False), False),

            # derived from per-ghost dist_to_pac if present
            "ghostDistances": [
                _i(ghosts_in[g].get("dist_to_pac", 999), 999)
                if g < len(ghosts_in) and isinstance(ghosts_in[g], dict)
                else 999
                for g in range(4)
            ],

            # edibleTimes / closestPillDist (only if your Java sends them)
            "edibleTimes": _li(obj.get("edibleTimes")),
            "closestPillDist": _i(obj.get("closestPillDist", -1), -1),

            # ghost neighbours per ghost
            "ghost_neighbours": [
                _li(ghosts_in[g].get("ghost_neighbour"))
                if g < len(ghosts_in) and isinstance(ghosts_in[g], dict)
                else []
                for g in range(4)
            ],
        }

        return obs, reward, done, info
