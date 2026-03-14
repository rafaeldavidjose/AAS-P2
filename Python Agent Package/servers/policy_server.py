"""
policy_server.py
Serves any agent implementing BaseAgent over TCP to the Java visualizer.
"""

import os
import sys

# Ensure the project root (the folder that contains 'env', 'agents', 'servers')
# is on sys.path, even if this script is run from an IDE that sets cwd=script dir.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import socket
import threading
from agents.random_agent import RandomAgent   # or student agent
from agents.greedy_safe_agent import GreedySafeAgent
from agents.base_agent import BaseAgent

class PolicyServer:
    def __init__(self, host, port, agent: BaseAgent):
        self.host = host
        self.port = port
        self.agent = agent

    def handle(self, conn, addr):
        print(f"[PolicyServer] {addr} connected")
        fin = conn.makefile("r")
        fout = conn.makefile("w")

        for line in fin:
            obs = json.loads(line)
            action = self.agent.act(obs)
            fout.write(str(int(action)) + "\n")
            fout.flush()

        conn.close()

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, self.port))
        srv.listen()
        print(f"[PolicyServer] Running on {self.host}:{self.port}")

        while True:
            conn, addr = srv.accept()
            threading.Thread(target=self.handle, args=(conn, addr), daemon=True).start()


from agents.agent_heuristic import HeuristicAgent
from agents.agent_mcts import MCTSAgent

def make_agent():
    from agents.agent_rl import AgentRL
    return MCTSAgent(simulations=1, rollout_depth=10)

if __name__ == "__main__":
    server = PolicyServer("127.0.0.1", 6001, make_agent())
    server.run()
