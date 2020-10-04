import time
from typing import List
from gym_mapf.envs.mapf_env import MapfEnv

from research.solvers.utils import CrossedPolicy
from research.solvers.rtdp import RtdpPolicy


class MultiagentRtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super(MultiagentRtdpPolicy, self).__init__(env, gamma, heuristic)
        self.agent_to_q = {i: {} for i in range(env.n_agents)}

    def _get_agent_action_from_state(self, s, agent):

    def get_q(self, s, a, agent):
        if s in self.agent_to_q[agent]:
            return self.agent_to_q[agent][s][a]

        agent_to_action = self.agent_to_q[agent][s]


def select_action(policy: MultiagentRtdpPolicy, joint_state: int):
    pass


def update(policy: MultiagentRtdpPolicy, joint_state: int, joint_action: int):
    pass
