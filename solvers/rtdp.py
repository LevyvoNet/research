import functools
import math
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List

import numpy as np

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    ALL_STAY_JOINT_ACTION)
from solvers.utils import Policy, ValueFunctionPolicy, get_local_view, evaluate_policy, dijkstra_distance_single_env
from solvers.vi import PrioritizedValueIterationPolicy

MDR_EPSILON = 0.1
MIN_SUCCESS_RATE = 0.5


class RtdpPolicy(ValueFunctionPolicy):
    def __init__(self, env, gamma, heuristic, batch_size, max_iters, name: str = ''):
        super().__init__(env, gamma, name)
        self.v_partial_table = {}
        # Now this v behaves like a full numpy array
        self.v = function_to_get_item_of_object(self._get_value)
        self.heuristic_function = heuristic
        self.heuristic = None
        self.visited_states = defaultdict(lambda: 0)
        self.in_train = True
        self.max_iters = max_iters
        self.batch_size = batch_size

    def _get_value(self, s):
        if s in self.v_partial_table:
            return self.v_partial_table[s]

        value = self.heuristic(s)

        # Add this states to 'seen states' only during training
        if self.in_train:
            self.v_partial_table[s] = value

        return value

    def _act_in_unfamiliar_state(self, s: int):
        if self.in_train or s in self.v_partial_table:
            a = greedy_action(self, s)
            self.policy_cache[s] = a
            return a

        return ALL_STAY_JOINT_ACTION
        # all_stay_action_vector = (STAY,) * self.env.n_agents
        # return vector_action_to_integer(all_stay_action_vector)

    def train(self, *args, **kwargs):
        start = time.time()
        self.heuristic = self.heuristic_function(self.env)
        self.info['initialization_time'] = round(time.time() - start, 1)
        iterations_generator = rtdp_iterations_generator(self, greedy_action, bellman_update, self.info)

        _stop_when_no_improvement_between_batches_rtdp(self.heuristic,
                                                       self.gamma,
                                                       self.batch_size,
                                                       self.max_iters,
                                                       self.env,
                                                       self,
                                                       iterations_generator,
                                                       self.info)
        self.info['total_time'] = round(time.time() - start)

        return self

    def train_info(self):
        train_info_dict = {}

        # Set initialization time
        train_info_dict['solver_init_time'] = round(self.info['initialization_time'], 1)

        # Set evaluation time
        train_info_dict['total_evaluation_time'] = round(self.info['total_evaluation_time'], 1)

        # Set number of iterations
        train_info_dict['n_visited_states'] = self.info['n_visited_states']

        # Set visited states
        train_info_dict['n_iterations'] = self.info['n_iterations']

        # Set last MDR
        train_info_dict['last_MDR'] = self.info['last_MDR']

        return train_info_dict


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
    local_v = [(PrioritizedValueIterationPolicy(local_env, gamma).train()).v for local_env in local_envs]

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
    local_v = [(PrioritizedValueIterationPolicy(local_env, gamma).train()).v for local_env in local_envs]

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
    policy = PrioritizedValueIterationPolicy(deterministic_env, gamma).train()

    def heuristic_function(s):
        return policy.v[s]

    return heuristic_function


def dijkstra_min_heuristic(env: MapfEnv, *args, **kwargs):
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_distance = [(dijkstra_distance_single_env(local_env)) for local_env in local_envs]

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
    local_distance = [dijkstra_distance_single_env(local_env) for local_env in local_envs]

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


def rtdp_dijkstra_sum_heuristic(gamma, max_iters, env: MapfEnv):
    local_envs = {
        agent: get_local_view(env, [agent])
        for agent in range(env.n_agents)
    }

    local_policies = {
        agent: RtdpPolicy(local_envs[agent], gamma, dijkstra_sum_heuristic, 100, max_iters).train()
        for agent in range(env.n_agents)
    }

    local_v = {agent: local_policies[agent].v for agent in range(env.n_agents)}

    def f(s: int):
        locations = env.state_to_locations(s)
        local_states = tuple([env.loc_to_int[loc] for loc in locations])
        relevant_values = [
            local_v[agent][local_states[agent]]
            for agent in range(env.n_agents) if env.loc_to_int[env.agents_goals[agent]] != local_states[agent]
        ]

        if not relevant_values:
            return 0

        # Each relevant value is composed of the reward of living until goal, and the reward for reaching the goal.
        # We only want to count the reward of the goal once.
        return sum(relevant_values) - (len(relevant_values) - 1) * env.reward_of_goal

    return f


def rtdp_dijkstra_min_heuristic(gamma, max_iters, env: MapfEnv):
    local_envs = {
        agent: get_local_view(env, [agent])
        for agent in range(env.n_agents)
    }

    local_policies = {
        agent: RtdpPolicy(local_envs[agent], gamma, dijkstra_min_heuristic, 100, max_iters).train()
        for agent in range(env.n_agents)
    }

    local_v = {agent: local_policies[agent].v for agent in range(env.n_agents)}

    def f(s: int):
        locations = env.state_to_locations(s)
        local_states = tuple([env.loc_to_int[loc] for loc in locations])
        relevant_values = [
            local_v[agent][local_states[agent]]
            for agent in range(env.n_agents) if env.loc_to_int[env.agents_goals[agent]] != local_states[agent]
        ]

        if not relevant_values:
            return 0

        return min(relevant_values)

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


def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int, iterations_batch_size: int, n_episodes: int,
                                   max_eval_steps: int, info: Dict):
    if iter_count % iterations_batch_size != 0:
        return False

    policy.in_train = False
    policy.policy_cache.clear()
    eval_info = evaluate_policy(policy, n_episodes, max_eval_steps, min_success_rate=MIN_SUCCESS_RATE)
    policy.in_train = True
    info['last_MDR'] = eval_info['MDR']
    info['last_success_rate'] = eval_info['success_rate']

    if eval_info['success_rate'] < 100 * MIN_SUCCESS_RATE:
        return False

    if not hasattr(policy, 'last_eval'):
        policy.last_eval = eval_info
        return False
    else:
        prev_eval = policy.last_eval
        policy.last_eval = eval_info

        return abs(policy.last_eval['MDR'] - prev_eval['MDR']) / abs(prev_eval['MDR']) <= MDR_EPSILON


def _stop_when_no_improvement_between_batches_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
                                                   gamma: float,
                                                   iterations_batch_size: int,
                                                   max_iterations: int,
                                                   env: MapfEnv,
                                                   policy: RtdpPolicy,
                                                   iterations_generator: Iterable,
                                                   info: Dict):
    max_eval_steps = 1000
    n_episodes_eval = 100

    info['total_evaluation_time'] = 0

    # Run RTDP iterations
    for iter_count, reward in enumerate(iterations_generator, start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        eval_start = time.time()
        no_improvement = no_improvement_from_last_batch(policy,
                                                        iter_count,
                                                        iterations_batch_size,
                                                        n_episodes_eval,
                                                        max_eval_steps,
                                                        info)

        # Update information
        info['total_evaluation_time'] += time.time() - eval_start
        info['n_iterations'] = iter_count
        info['n_visited_states'] = len(policy.v_partial_table)

        if no_improvement or iter_count >= max_iterations:
            break

    policy.in_train = False
    return policy


def fixed_iterations_rtdp_merge(heuristic_function: Callable[[Policy, Policy, MapfEnv], Callable[[int], float]],
                                n_iterations: int,
                                env: MapfEnv,
                                gamma: float,
                                old_groups,
                                old_group_i_idx,
                                old_group_j_idx,
                                policy_i,
                                policy_j,
                                info: Dict):
    heursitic_func = functools.partial(heuristic_function,
                                       policy_i,
                                       policy_j,
                                       old_groups,
                                       old_group_i_idx,
                                       old_group_j_idx)

    policy = RtdpPolicy(env, gamma, heursitic_func, n_iterations, n_iterations).train()
    return policy


def stop_when_no_improvement_between_batches_rtdp_merge(
        heuristic_function: Callable[[Policy, Policy, List, int, int, MapfEnv], Callable[[int], float]],
        iterations_batch_size,
        max_iterations,
        env,
        gamma,
        old_groups,
        old_group_i_idx,
        old_group_j_idx,
        policy_i,
        policy_j):
    heuristic_func = functools.partial(heuristic_function,
                                       policy_i,
                                       policy_j,
                                       old_groups,
                                       old_group_i_idx,
                                       old_group_j_idx)

    policy = RtdpPolicy(env, gamma, heuristic_func, iterations_batch_size, max_iterations).train()
    return policy


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

        g1 = policy1.env.locations_to_state(policy1.env.agents_goals)
        g2 = policy2.env.locations_to_state(policy2.env.agents_goals)

        relevant_values = []
        if s1 != g1:
            relevant_values.append(v1)
        if s2 != g2:
            relevant_values.append(v2)

        if not relevant_values:
            return 0

        return sum(relevant_values) - (len(relevant_values) - 1) * env.reward_of_goal

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

        g1 = policy1.env.locations_to_state(policy1.env.agents_goals)
        g2 = policy2.env.locations_to_state(policy2.env.agents_goals)

        relevant_values = []
        if s1 != g1:
            relevant_values.append(v1)
        if s2 != g2:
            relevant_values.append(v2)

        if not relevant_values:
            return 0

        return min(relevant_values)

    return func
