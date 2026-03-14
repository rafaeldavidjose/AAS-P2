"""
Helper functions for working with observations.
"""
import os
import sys

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from typing import Dict, Any


def is_any_ghost_nearby(obs: Dict[str, Any], radius: int = 20) -> bool:
    """Crude danger heuristic based on |loc difference|."""
    pac_loc = obs.get("pac_loc", 0)
    for g in obs.get("ghosts", []):
        if not g.get("edible", False):
            if abs(g.get("loc", 0) - pac_loc) <= radius:
                return True
    return False


def count_edible_ghosts(obs: Dict[str, Any]) -> int:
    """Counts how many ghosts are currently edible."""
    return sum(1 for g in obs.get("ghosts", []) if g.get("edible", False))