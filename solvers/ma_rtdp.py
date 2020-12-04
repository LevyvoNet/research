import time
import math
from typing import Dict, Callable
from collections import defaultdict
import numpy as np
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    STAY,
                                    ACTIONS,
                                    vector_action_to_integer)

from solvers.utils import evaluate_policy
from solvers.rtdp import RtdpPolicy


class MultiagentRtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super(MultiagentRtdpPolicy, self).__init__(env, gamma, heuristic)
        self.q_partial_table = {i: defaultdict(dict) for i in range(env.n_agents)}

    def get_q(self, agent, joint_state, local_action):
        if joint_state in self.q_partial_table[agent]:
            return self.q_partial_table[agent][joint_state][local_action]

        # Calculate Q[s][a] for each possible local action
        for a in range(len(ACTIONS)):
            all_stay = (STAY,) * self.env.n_agents
            joint_action_vector = all_stay[:agent] + (ACTIONS[a],) + all_stay[agent + 1:]
            joint_action = vector_action_to_integer(joint_action_vector)

            # Compute Q[s][a]. In case of a possible clash set the reward to -infinity
            q_value = 0
            for prob, next_state, reward, done in self.env.P[joint_state][joint_action]:
                if reward == self.env.reward_of_clash and done:
                    q_value = -math.inf

                q_value += prob * (reward + (self.gamma * self.v[next_state]))

            self.q_partial_table[agent][joint_state][a] = q_value

        return self.q_partial_table[agent][joint_state][local_action]

    def q_update(self, agent, joint_state, local_action):
        all_stay = (STAY,) * self.env.n_agents

        fake_joint_action = vector_action_to_integer(all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:])

        self.q_partial_table[agent][joint_state][local_action] = sum([prob * (reward + self.gamma * self.v[next_state])
                                                                      for prob, next_state, reward, done in
                                                                      self.env.P[joint_state][fake_joint_action]])

    def v_update(self, joint_state):
        self.v_partial_table[joint_state] = max([max([self.get_q(agent_idx, joint_state, a)
                                                      for a in range(len(ACTIONS))])
                                                 for agent_idx in range(self.env.n_agents)])


def best_response(policy: MultiagentRtdpPolicy, joint_state: int, agent: int):
    action_values = [policy.get_q(agent, joint_state, local_action)
                     for local_action in range(len(ACTIONS))]

    max_value = np.max(action_values)
    return np.random.choice(np.argwhere(action_values == max_value).flatten())


def multi_agent_turn_based_rtdp_single_iteration(policy: MultiagentRtdpPolicy, info: Dict):
    s = policy.env.reset()
    done = False
    start = time.time()
    path = [s]
    total_reward = 0
    all_stay = (STAY,) * policy.env.n_agents

    # # debug
    # print('--------start iteration---------------')

    while not done:
        # import ipdb
        # ipdb.set_trace()
        trajectory_states = [s]
        trajectory_actions = []
        for agent in range(policy.env.n_agents):
            local_action = best_response(policy, s, agent)
            trajectory_actions.append(local_action)
            fake_joint_action_vector = all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:]
            fake_joint_action = vector_action_to_integer(fake_joint_action_vector)

            # # debug
            # policy.env.render()
            # print(f'selected action: {fake_joint_action_vector}')
            # time.sleep(1)

            s, r, done, _ = policy.env.step(fake_joint_action)
            trajectory_states.append(s)
            path.append(s)

        for agent in reversed(range(policy.env.n_agents)):
            # update q(s, agent, action) based on the last state
            policy.q_update(agent, trajectory_states[agent], trajectory_actions[agent])
            policy.v_update(trajectory_states[agent])

    # Backward update
    while path:
        s = path.pop()
        policy.v_update(s)

    # TODO: There is a bug here(total_reward is always 0), will think about it later

    # # debug
    # print('--------end iteration---------------')
    return total_reward


def multi_agent_turn_based_rtdp_iterations_generator(policy, info: Dict):
    info['iterations'] = []

    while True:
        info['iterations'].append({})
        iter_reward = multi_agent_turn_based_rtdp_single_iteration(policy, info['iterations'][-1])
        yield iter_reward


def ma_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
            gamma: float,
            iterations_batch_size: int,
            max_iterations: int,
            env: MapfEnv,
            info: Dict):
    def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int):
        if iter_count % iterations_batch_size != 0:
            return False

        policy.policy_cache.clear()
        reward, _ = evaluate_policy(policy, 100, 1000)
        if reward == policy.env.reward_of_living * 1000:
            return False

        if not hasattr(policy, 'last_eval'):
            policy.last_eval = reward
            return False
        else:
            prev_eval = policy.last_eval
            policy.last_eval = reward
            return abs(policy.last_eval - prev_eval) / abs(prev_eval) <= 0.01

    # initialize V to an upper bound
    start = time.time()
    policy = MultiagentRtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = time.time() - start

    # Run RTDP iterations
    for iter_count, reward in enumerate(multi_agent_turn_based_rtdp_iterations_generator(policy, info),
                                        start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        if no_improvement_from_last_batch(policy, iter_count) or iter_count >= max_iterations:
            break

    return policy
