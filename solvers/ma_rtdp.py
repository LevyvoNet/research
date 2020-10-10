import time
from typing import List
from collections import defaultdict
import numpy as np
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    STAY,
                                    ACTIONS,
                                    vector_action_to_integer)

from research.solvers.utils import CrossedPolicy
from research.solvers.rtdp import RtdpPolicy


class MultiagentRtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super(MultiagentRtdpPolicy, self).__init__(env, gamma, heuristic)
        self.q_partial_table = {i: defaultdict(dict) for i in range(env.n_agents)}

    def get_q(self, agent, joint_state, local_action):
        if joint_state in self.q_partial_table[agent]:
            return self.q_partial_table[agent][joint_state][local_action]

        all_stay = (STAY,) * self.env.n_agents
        joint_action_vector = all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:]
        joint_action = vector_action_to_integer(joint_action_vector)

        q_value = sum([[prob * (reward + self.gamma * self.v[next_state])
                        for prob, next_state, reward, done in self.env.P[joint_state][joint_action]]])

        self.q_partial_table[agent][joint_state][local_action] = q_value
        return q_value

    def q_update(self, agent, joint_state, local_action, joint_next_state):
        self.q_partial_table[agent][joint_state][local_action] = self.v[joint_next_state]


def best_response(policy: MultiagentRtdpPolicy, joint_state: int, agent: int):
    action_values = [policy.get_q(agent, joint_state, local_action)
                     for local_action in range(len(ACTIONS))]

    max_value = np.max(action_values)
    return np.random.choice(np.argwhere(action_values == max_value).flatten())


def multiagent_turn_based_rtdp_single_iteration(policy: MultiagentRtdpPolicy):
    s = policy.env.reset()
    done = False
    start = time.time()
    path = []
    total_reward = 0
    all_stay = (STAY,) * policy.env.n_agents

    while not done:
        trajectory_states = [s]
        trajectory_actions = []
        for agent in policy.env.n_agents:
            local_action = best_response(policy, s, agent)
            trajectory_actions.append(local_action)
            fake_joint_action_vector = all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:]
            fake_joint_action = vector_action_to_integer(fake_joint_action)

            s = policy.env.step(fake_joint_action)
            trajectory_states.append(s)

        for agent in policy.env.n_agents:
            # update q(s, agent, action) based on the last state
            policy.q_update(agent, trajectory_states[agent], trajectory_actions[agent], s)
            policy.v_update(trajectory_states[agent])
