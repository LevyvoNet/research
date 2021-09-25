import functools

import numpy as np
import time
import math
from typing import Callable, Dict, Iterable
from collections import defaultdict

from gym_mapf.envs.mapf_env import MapfEnv, function_to_get_item_of_object, vector_action_to_integer, STAY
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
        if s in self.v_partial_table:
            return self.v_partial_table[s]

        value = self.heuristic(s)
        self.v_partial_table[s] = value
        return value

    def _act_in_unfamiliar_state(self, s: int):
        all_stay_action_vector = (STAY,) * self.env.n_agents
        return vector_action_to_integer(all_stay_action_vector)


def greedy_action(policy: RtdpPolicy, s):
    q_s_a = calc_q_s_no_clash_possible(policy, s)

    # # for debug
    # for i in range(env.nA):
    #     print(f'{integer_action_to_vector(i, env.n_agents)}: {action_values[i]}')

    return np.argmax(q_s_a)
    # max_value = np.max(q_s_a)
    # return np.random.choice(np.argwhere(q_s_a == max_value).flatten())


# Heuristics #######################################################################################################

def local_views_prioritized_value_iteration_min_heuristic(gamma: float, env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented, ignore agents who are currently in their goal

        relevant_values = [
            local_v[i][local_states[i]] for i in range(env.n_agents)
            if local_envs[i].loc_to_int[local_envs[i].agents_goals[0]] != local_states[i]
        ]

        if not relevant_values:
            return 0
        return min(relevant_values)

    return heuristic_function


def local_views_prioritized_value_iteration_sum_heuristic(gamma: float, env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # try something which is more SoC oriented

        relevant_values = [
            local_v[i][local_states[i]] for i in range(env.n_agents)
            if local_envs[i].loc_to_int[local_envs[i].agents_goals[0]] != local_states[i]
        ]

        if not relevant_values:
            return 0

        return sum(relevant_values) - (len(relevant_values) - 1) * env.reward_of_goal

    return heuristic_function


def deterministic_relaxation_prioritized_value_iteration_heuristic(gamma: float,
                                                                   env: MapfEnv) -> Callable[[int], float]:
    deterministic_env = MapfEnv(env.grid,
                                env.n_agents,
                                env.agents_starts,
                                env.agents_goals,
                                0,
                                env.reward_of_clash,
                                env.reward_of_goal,
                                env.reward_of_living,
                                env.optimization_criteria)
    # TODO: consider using RTDP instead of PVI here, this is theoretically bad but practically may give better results
    policy = prioritized_value_iteration(gamma, deterministic_env, {})

    def heuristic_function(s):
        return policy.v[s]

    return heuristic_function


def _dijkstra_distance_single_env(env):
    goal_state = env.locations_to_state(env.agents_goals)
    distance = np.full((env.nS,), math.inf)
    visited = np.full((env.nS,), False)

    # Initialize the distance from goal state to 0
    distance[goal_state] = 0

    while not visited.all():
        # Fetch the cheapest unvisited state
        masked_distance = np.ma.masked_array(distance, mask=visited)
        current_state = masked_distance.argmin()
        current_distance = distance[current_state]

        # Update the distance for each of the neighbors
        for n in env.predecessors(current_state):
            distance[n] = min(distance[n], current_distance + 1)

        # Mark the current state as visited
        visited[current_state] = True

    return distance


def dijkstra_min_heuristic(env: MapfEnv, *args, **kwargs):
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_distance = [(_dijkstra_distance_single_env(local_env)) for local_env in local_envs]

    def f(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented, ignore agents who are currently in their goal

        relevant_distances = [
            local_distance[i][local_states[i]] for i in range(env.n_agents)
            if local_envs[i].loc_to_int[local_envs[i].agents_goals[0]] != local_states[i]
        ]

        if not relevant_distances:
            return 0

        return max(relevant_distances) * env.reward_of_living + env.reward_of_goal

    return f


def dijkstra_sum_heuristic(env: MapfEnv, *args, **kwargs):
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_distance = [_dijkstra_distance_single_env(local_env) for local_env in local_envs]

    def f(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented, ignore agents who are currently in their goal

        relevant_distances = [
            local_distance[i][local_states[i]] for i in range(env.n_agents)
            if local_envs[i].loc_to_int[local_envs[i].agents_goals[0]] != local_states[i]
        ]

        if not relevant_distances:
            return 0

        return sum(relevant_distances) * env.reward_of_living + env.reward_of_goal

    return f


# RTDP #############################################################################################################

def calc_q_s_no_clash_possible(policy: RtdpPolicy, s: int):
    q_s_a = np.zeros(policy.env.nA)
    for a in range(policy.env.nA):
        for (prob, collision), next_state, reward, done in policy.env.P[s][a]:
            if collision:
                q_s_a[a] = -math.inf
                break

            q_s_a[a] += prob * (reward + (policy.gamma * policy.v[next_state]))

    return q_s_a


def bellman_update(policy: RtdpPolicy, s: int):
    q_s_a = calc_q_s_no_clash_possible(policy, s)
    policy.v_partial_table[s] = max(q_s_a)

    # TODO: this is wrong, when a state change there might be more states which depend on this state's value and need
    #   to be updated in the policy cache as well
    policy.policy_cache[s] = np.argmax(q_s_a)


def rtdp_single_iteration(policy: RtdpPolicy,
                          select_action: Callable[[RtdpPolicy, int], int],
                          update: Callable[[RtdpPolicy, int], None],
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
    start = time.time()
    policy = RtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = round(time.time() - start, 1)
    info['total_evaluation_time'] = 0

    for iter_count, reward in enumerate(rtdp_iterations_generator(policy, greedy_action, bellman_update, info),
                                        start=1):
        if iter_count >= n_iterations:
            break

    return policy


def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int, iterations_batch_size: int, n_episodes: int,
                                   max_eval_steps: int):
    if iter_count % iterations_batch_size != 0:
        return False

    info = evaluate_policy(policy, n_episodes, max_eval_steps)
    if info['success_rate'] == 0:
        return False

    if not hasattr(policy, 'last_eval'):
        policy.last_eval = info['MDR']
        return False
    else:
        prev_eval = policy.last_eval
        policy.last_eval = info['MDR']
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
    info['initialization_time'] = round(time.time() - start, 1)
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

    info['total_evaluation_time'] = round(info['total_evaluation_time'], 1)
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

    def func(s: int):
        # Get the locations of each agent in the given state
        locations = env.state_to_locations(s)

        # Compose the local states of each of the policies
        loc1 = tuple([locations[agent_idx] for agent_idx in range(len(group1))])
        loc2 = tuple([locations[agent_idx + len(group1)] for agent_idx in range(len(group2))])

        s1 = policy1.env.locations_to_state(loc1)
        s2 = policy2.env.locations_to_state(loc2)

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

    def func(s: int):
        # Get the locations of each agent in the given state
        locations = env.state_to_locations(s)

        # Compose the local states of each of the policies
        loc1 = tuple([locations[agent_idx] for agent_idx in range(len(group1))])
        loc2 = tuple([locations[agent_idx + len(group1)] for agent_idx in range(len(group2))])

        s1 = policy1.env.locations_to_state(loc1)
        s2 = policy2.env.locations_to_state(loc2)

        v1 = policy1.v[s1]
        v2 = policy2.v[s2]

        return min(v1, v2)

    return func
