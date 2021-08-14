import functools

import numpy as np
import time
import math
from typing import Callable, Dict, Iterable
from collections import defaultdict

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    MultiAgentState,
                                    MultiAgentAction)
from gym_mapf.envs.grid import SingleAgentState, SingleAgentAction
from solvers.vi import prioritized_value_iteration
from solvers.utils import Policy, ValueFunctionPolicy, get_local_view, evaluate_policy


class RtdpPolicy(ValueFunctionPolicy):
    def __init__(self, env, gamma, heuristic):
        super().__init__(env, gamma)
        self.v_partial_table = {}
        # Now this v behaves like a full numpy array
        self.v = function_to_get_item_of_object(self._get_value)
        self.heuristic = heuristic
        self.visited_states = defaultdict(lambda: 0)

    def _get_value(self, s):
        if s.hash_value in self.v_partial_table:
            return self.v_partial_table[s.hash_value]

        value = self.heuristic(s)
        self.v_partial_table[s.hash_value] = value
        return value


def greedy_action(policy: RtdpPolicy, s: MultiAgentState):
    return policy.act(s)


def local_views_prioritized_value_iteration_min_heuristic(gamma: float, env: MapfEnv) -> Callable[
    [MultiAgentState], float]:
    local_envs = [get_local_view(env, [agent]) for agent in env.agents]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s: MultiAgentState):
        relevant_values = [
            local_v[env.agents.index(agent)][MultiAgentState({agent: s[agent]}, env.grid).hash_value] for agent in
            env.agents
            if env.goal_state[agent] != s[agent]
        ]

        if not relevant_values:
            return 0
        return min(relevant_values)

    return heuristic_function


def local_views_prioritized_value_iteration_sum_heuristic(gamma: float, env: MapfEnv) \
        -> Callable[[MultiAgentState], float]:
    local_envs = [get_local_view(env, [i]) for i in env.agents]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s: MultiAgentState):
        relevant_values = [
            local_v[env.agents.index(agent)][MultiAgentState({agent: s[agent]}, env.grid).hash_value] for agent in
            env.agents
            if env.goal_state[agent] != s[agent]
        ]

        if not relevant_values:
            return 0

        # Each relevant value is composed of the reward of living until goal, and the reward for reaching the goal.
        # We only want to count the reward of the goal once.
        return sum(relevant_values) - (len(relevant_values) - 1) * env.reward_of_goal

    return heuristic_function


def _dijkstra_distance_single_env(env):
    distance = {s[env.agents[0]]: math.inf for s in env.observation_space}
    visited = {s[env.agents[0]]: False for s in env.observation_space}
    n_visited_true = 0

    # Initialize the distance from goal state to 0
    distance[env.goal_state[env.agents[0]]] = 0

    while n_visited_true < env.nS:
        # Fetch the cheapest unvisited state
        current_state = min(filter(lambda s: not visited[s], distance), key=distance.get)
        current_distance = distance[current_state]

        # Update the distance for each of the neighbors
        for prev_state in env.predecessors(MultiAgentState({env.agents[0]: current_state}, env.grid)):
            distance[prev_state[env.agents[0]]] = min(distance[prev_state[env.agents[0]]], current_distance + 1)

        # Mark the current state as visited
        visited[current_state] = True
        n_visited_true += 1

    return distance


def dijkstra_min_heuristic(env: MapfEnv, *args, **kwargs):
    local_envs = [get_local_view(env, [agent]) for agent in env.agents]
    local_distance = [(_dijkstra_distance_single_env(local_env)) for local_env in local_envs]

    def f(s):
        relevant_distances = [
            local_distance[i][s[i]] for i in env.agents
            if env.goal_state[i] != s[i]
        ]

        if not relevant_distances:
            return 0

        return max(relevant_distances) * env.reward_of_living + env.reward_of_goal

    return f


def dijkstra_sum_heuristic(env: MapfEnv, *args, **kwargs):
    local_envs = [get_local_view(env, [agent]) for agent in env.agents]
    local_distance = [_dijkstra_distance_single_env(local_env) for local_env in local_envs]

    def f(s):
        relevant_distances = [
            local_distance[i][s[i]] for i in env.agents
            if env.goal_state[i] != s[i]
        ]

        if not relevant_distances:
            return 0

        return sum(relevant_distances) * env.reward_of_living + env.reward_of_goal

    return f


def bellman_update(policy: RtdpPolicy, s: MultiAgentState):
    vs = -math.inf
    best_action = None
    for a in policy.env.action_space:
        qsa = 0
        for prob, next_state, reward, done in policy.env.P[s][a]:
            if policy.env.is_collision_transition(s, next_state):
                qsa = -math.inf
                break

            qsa += prob * (reward + (policy.gamma * policy.v[next_state]))

        if qsa > vs:
            vs = qsa
            best_action = a

    policy.v_partial_table[s.hash_value] = vs
    policy.policy_cache[s] = best_action


def rtdp_single_iteration(policy: RtdpPolicy,
                          select_action: Callable[[RtdpPolicy, MultiAgentState], MultiAgentAction],
                          update: Callable[[RtdpPolicy, MultiAgentState], None],
                          info: Dict):
    """Run a single iteration of RTDP.

    Args:
        policy (RtdpPolicy): the current policy (RTDP is an on-policy algorithm)
        select_action ((RtdpPolicy, state) -> action): Action selection function
        update ((RtdpPolicy, state, action) -> None): Update function
        info (Dict): optional for gathering information about the iteration - time, reward, special events, etc.

    Returns:
        float. The total reward of the episode.
    """
    s = policy.env.reset()
    done = False
    start = time.time()
    path = []
    total_reward = 0

    steps = 0
    while not done and steps < 1000:
        steps += 1

        # Choose action action for current state
        a = select_action(policy, s)

        # Do a bellman update
        update(policy, s)

        # Simulate the step and sample a new state
        s, r, done, _ = policy.env.step(a)
        total_reward += r

        # Add next state to path
        path.append(s)

    # Backward update
    while path:
        s = path.pop()
        update(policy, s)

    # Write measures about that information
    info['time'] = round(time.time() - start, 2)
    info['n_moves'] = len(path)

    # Reset again just for safety
    policy.env.reset()

    return total_reward


def rtdp_iterations_generator(policy: RtdpPolicy, select_action, update, info: Dict) -> Iterable:
    # info['iterations'] = []

    while True:
        # info['iterations'].append({})
        iter_reward = rtdp_single_iteration(policy, select_action, update, {})
        yield iter_reward


def fixed_iterations_count_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                                gamma: float,
                                n_iterations: int,
                                env: MapfEnv,
                                info: Dict) -> Policy:
    # initialize V to an upper bound
    policy = RtdpPolicy(env, gamma, heuristic_function(env))

    for iter_count, reward in enumerate(rtdp_iterations_generator(policy, greedy_action, bellman_update, info),
                                        start=1):
        if iter_count >= n_iterations:
            break

    return policy


def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int, iterations_batch_size: int, n_episodes: int,
                                   max_eval_steps: int):
    if iter_count % iterations_batch_size != 0:
        return False

    policy.policy_cache.clear()
    reward, _, _ = evaluate_policy(policy, n_episodes, max_eval_steps)
    if reward == policy.env.reward_of_living * max_eval_steps:
        return False

    if not hasattr(policy, 'last_eval'):
        policy.last_eval = reward
        return False
    else:
        prev_eval = policy.last_eval
        policy.last_eval = reward
        return abs(policy.last_eval - prev_eval) / abs(prev_eval) <= 0.01


def stop_when_no_improvement_between_batches_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                                                  gamma: float,
                                                  iterations_batch_size: int,
                                                  max_iterations: int,
                                                  env: MapfEnv,
                                                  info: Dict):
    max_eval_steps = 1000
    n_episodes_eval = 100

    # initialize V to an upper bound
    start = time.time()
    policy = RtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = time.time() - start
    info['total_evaluation_time'] = 0

    # Run RTDP iterations
    for iter_count, reward in enumerate(rtdp_iterations_generator(policy, greedy_action, bellman_update, info),
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


def fixed_iterations_rtdp_merge(heuristic_function: Callable[[Policy, Policy, MapfEnv], Callable[[int], float]],
                                gamma: float,
                                n_iterations: int,
                                env: MapfEnv,
                                old_groups,
                                old_group_i_idx,
                                old_group_j_idx,
                                policy_i,
                                policy_j,
                                info: Dict):
    return fixed_iterations_count_rtdp(functools.partial(heuristic_function,
                                                         policy_i,
                                                         policy_j,
                                                         old_groups,
                                                         old_group_i_idx,
                                                         old_group_j_idx),
                                       gamma,
                                       n_iterations,
                                       env, info)


def stop_when_no_improvement_between_batches_rtdp_merge(
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
    return stop_when_no_improvement_between_batches_rtdp(functools.partial(heuristic_function,
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


def solution_heuristic_sum(policy1: ValueFunctionPolicy,
                           policy2: ValueFunctionPolicy,
                           old_groups,
                           old_group_1_idx,
                           old_group_2_idx,
                           env: MapfEnv):
    group1 = old_groups[old_group_1_idx]
    group2 = old_groups[old_group_2_idx]

    def func(s: MultiAgentState):
        s1 = MultiAgentState({agent: s[agent] for agent in group1}, policy1.env.grid)
        s2 = MultiAgentState({agent: s[agent] for agent in group2}, policy2.env.grid)

        v1 = policy1.v[s1]
        v2 = policy2.v[s2]

        return v1 + v2

    return func


def solution_heuristic_min(policy1: ValueFunctionPolicy,
                           policy2: ValueFunctionPolicy,
                           old_groups,
                           old_group_1_idx,
                           old_group_2_idx,
                           env: MapfEnv):
    group1 = old_groups[old_group_1_idx]
    group2 = old_groups[old_group_2_idx]

    def func(s: MultiAgentState):
        s1 = MultiAgentState({agent: s[agent] for agent in group1}, policy1.env.grid)
        s2 = MultiAgentState({agent: s[agent] for agent in group2}, policy2.env.grid)

        v1 = policy1.v[s1]
        v2 = policy2.v[s2]

        return min(v1, v2)

    return func
