import functools
import time
import math
from typing import Dict, Callable
from collections import defaultdict
import numpy as np

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    STAY,
                                    ACTIONS,
                                    vector_action_to_integer,
                                    integer_action_to_vector)
from gym_mapf.envs.utils import get_local_view

from solvers.utils import evaluate_policy, Policy
from solvers.rtdp import RtdpPolicy, no_improvement_from_last_batch


class MultiagentRtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super(MultiagentRtdpPolicy, self).__init__(env, gamma, heuristic)
        self.q_partial_table = {i: defaultdict(dict) for i in range(env.n_agents)}
        self.local_env_aux = get_local_view(self.env, [0])

    def get_q(self, agent, joint_state, local_action):
        if joint_state in self.q_partial_table[agent]:
            if local_action in self.q_partial_table[agent][joint_state]:
                return self.q_partial_table[agent][joint_state][local_action]

        # Calculate Q[s][a] for each possible local action
        all_stay = (STAY,) * self.env.n_agents
        joint_action_vector = all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:]
        joint_action = vector_action_to_integer(joint_action_vector)

        # Compute Q[s][a]. In case of a possible clash set the reward to -infinity
        q_value = 0
        for (prob, collision), next_state, reward, done in self.env.P[joint_state][joint_action]:
            if collision:
                q_value = -math.inf

            q_value += prob * (reward + (self.gamma * self.v[next_state]))

        self.q_partial_table[agent][joint_state][local_action] = q_value

        return self.q_partial_table[agent][joint_state][local_action]

    def q_update(self, agent, joint_state, local_action, joint_action):
        # TODO: figure out which way is right - considering the joint action or the local one.
        # TODO: Maybe each agent should have a different heuristic
        # all_stay = (STAY,) * self.env.n_agents
        # fake_joint_action = vector_action_to_integer(all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:])

        self.q_partial_table[agent][joint_state][local_action] = sum([prob * (reward + self.gamma * self.v[next_state])
                                                                      for (prob, collision), next_state, reward, done in
                                                                      self.env.P[joint_state][joint_action]])

    def v_update(self, joint_state):
        self.v_partial_table[joint_state] = max([max([self.get_q(agent_idx, joint_state, a)
                                                      for a in range(len(ACTIONS))])
                                                 for agent_idx in range(self.env.n_agents)])

    def act(self, joint_state):
        if joint_state in self.policy_cache:
            return self.policy_cache[joint_state]

        joint_action = ()
        forbidden_states = set()
        for agent in range(self.env.n_agents):
            # TODO: the problem is that the best response is according to joint state even though we are in state s.
            # TODO: we shouldn't actually step in this part...
            local_action = best_response(self, joint_state, agent, forbidden_states, False)
            joint_action = joint_action + (ACTIONS[local_action],)

        best_action = vector_action_to_integer(joint_action)
        self.policy_cache[joint_state] = best_action
        return best_action


def best_response(policy: MultiagentRtdpPolicy, joint_state: int, agent: int, forbidden_states, stochastic=True):
    action_values = [policy.get_q(agent, joint_state, local_action)
                     for local_action in range(len(ACTIONS))]

    # Make sure the chosen local action is not forbidden
    locations = policy.env.state_to_locations(joint_state)
    local_state = policy.local_env_aux.locations_to_state((locations[agent],))
    for local_action in range(len(action_values)):
        for _, next_state, prob in policy.env.single_agent_movements(local_state, local_action):
            if next_state in forbidden_states and prob > 0:
                action_values[local_action] = -math.inf

    max_value = np.max(action_values)
    if stochastic:
        best_action = np.random.choice(np.argwhere(action_values == max_value).flatten())
    else:
        best_action = np.argmax(action_values)

    # Forbid the possible states from the chosen action to the next agents
    for _, next_state, prob in policy.env.single_agent_movements(local_state, best_action):
        if prob > 0:
            forbidden_states.add(next_state)

    return best_action


def multi_agent_turn_based_rtdp_single_iteration(policy: MultiagentRtdpPolicy,
                                                 info: Dict):
    s = policy.env.reset()
    done = False
    start = time.time()
    path = []
    total_reward = 0

    # # debug
    # print('--------start iteration---------------')

    steps = 0
    while not done and steps < 1000:
        steps += 1
        trajectory_actions = []
        forbidden_states = set()
        joint_action_vector = (STAY,) * policy.env.n_agents

        # Calculate local action
        for agent in range(policy.env.n_agents):
            local_action = best_response(policy, s, agent, forbidden_states, False)
            trajectory_actions.append(local_action)
            joint_action_vector = joint_action_vector[:agent] + (ACTIONS[local_action],) + joint_action_vector[
                                                                                           agent + 1:]

        # # debug
        # policy.env.render()
        # print(f'selected action: {joint_action_vector}')
        # time.sleep(0.2)

        # Compose the joint action
        joint_action = vector_action_to_integer(joint_action_vector)
        path.append((s, joint_action))

        # update the current state
        for agent in reversed(range(policy.env.n_agents)):
            # update q(s, agent, action) based on the last state
            policy.v_update(s)
            policy.q_update(agent, s, trajectory_actions[agent], joint_action)

        policy.visited_states[s] = policy.visited_states[s] + 1

        # step
        s, r, done, _ = policy.env.step(joint_action)
        total_reward += r

    # # debug
    # policy.env.render()

    # # Backward update
    # while path:
    #     s, joint_action = path.pop()
    #     policy.v_update(s)
    #     joint_action_vector = integer_action_to_vector(joint_action, policy.env.n_agents)
    #     for agent in reversed(range(policy.env.n_agents)):
    #         local_action = vector_action_to_integer((joint_action_vector[agent],))
    #         policy.q_update(agent, s, local_action, joint_action)

    # # debug
    # print('--------end iteration---------------')

    return total_reward


def multi_agent_turn_based_rtdp_iterations_generator(policy,
                                                     info: Dict):
    info['iterations'] = []

    while True:
        iter_info = {}
        info['iterations'].append(iter_info)
        start = time.time()
        iter_reward = multi_agent_turn_based_rtdp_single_iteration(policy, {})
        iter_info['time'] = time.time() - start
        iter_info['reward'] = iter_reward
        yield iter_reward


def ma_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
            gamma: float,
            iterations_batch_size: int,
            max_iterations: int,
            env: MapfEnv,
            info: Dict):
    max_eval_steps = 1000
    n_episodes_eval = 100

    # initialize V to an upper bound
    start = time.time()
    policy = MultiagentRtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = time.time() - start
    info['total_evaluation_time'] = 0

    # Run RTDP iterations
    for iter_count, reward in enumerate(multi_agent_turn_based_rtdp_iterations_generator(policy, info),
                                        start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        eval_start = time.time()
        no_improvement = no_improvement_from_last_batch(policy,
                                                        iter_count,
                                                        iterations_batch_size,
                                                        n_episodes_eval,
                                                        max_eval_steps)
        info['total_evaluation_time'] += time.time() - eval_start
        if no_improvement or iter_count >= max_iterations:
            break

    info['n_iterations'] = iter_count
    info['total_time'] = time.time() - start

    return policy


def ma_rtdp_merge(
        heuristic_function: Callable[[Policy, Policy, MapfEnv], Callable[[int], float]],
        gamma,
        iterations_batch_size,
        max_iterations,
        env,
        old_groups,
        old_group_i_idx,
        old_group_j_idx,
        policy_i,
        policy_j,
        info):
    return ma_rtdp(functools.partial(heuristic_function,
                                     policy_i,
                                     policy_j,
                                     old_groups,
                                     old_group_i_idx,
                                     old_group_j_idx),
                   gamma,
                   iterations_batch_size,
                   max_iterations,
                   env,
                   info)

