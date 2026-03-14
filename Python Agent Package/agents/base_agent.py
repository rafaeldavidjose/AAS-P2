import os
import sys

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# This is the Base Agent. All agents must extend this specific class, and 
# implement the act(self, obs) for it to work within the Java Environment. 
class BaseAgent:
    """
    Every agent must implement act(obs) -> action_int.
    obs: dict
    returns: one of {-1,0,1,2,3} - See action_utils.py
    """
    def act(self, obs):
        raise NotImplementedError()
