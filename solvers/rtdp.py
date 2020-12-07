import numpy as np
import time
import math
from typing import Callable, Dict, Iterable

from gym_mapf.envs.mapf_env import MapfEnv, function_to_get_item_of_object, integer_action_to_vector
from solvers.vi import prioritized_value_iteration
from solvers.utils import Policy, ValueFunctionPolicy, get_local_view, evaluate_policy


class RtdpPolicy(ValueFunctionPolicy):
    def __init__(self, env, gamma, heuristic):
        super().__init__(env, gamma)
        self.v_partial_table = {}
        # Now this v behaves like a full numpy array
        self.v = function_to_get_item_of_object(self._get_value)
        self.heuristic = heuristic

    def _get_value(self, s):
        if s in self.v_partial_table:
            return self.v_partial_table[s]

        value = self.heuristic(s)
        self.v_partial_table[s] = value
        return value


# TODO: Is really important to get a random greedy action (instead of just the first index?).
#  I wish I could delete this function and just use `policy.act(s)` instead
def greedy_action(policy: RtdpPolicy, s):
    q_s_a = calc_q_s_no_clash_possible(policy, s)

    # # for debug
    # for i in range(env.nA):
    #     print(f'{integer_action_to_vector(i, env.n_agents)}: {action_values[i]}')

    max_value = np.max(q_s_a)
    return np.random.choice(np.argwhere(q_s_a == max_value).flatten())


def local_views_prioritized_value_iteration_min_heuristic(gamma: float, env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # at the current, the MapfEnv reward is makespan oriented
        return min([local_v[i][local_states[i]] for i in range(env.n_agents)])

    return heuristic_function


def local_views_prioritized_value_iteration_sum_heuristic(gamma: float, env: MapfEnv) -> Callable[[int], float]:
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(prioritized_value_iteration(gamma, local_env, {})).v for local_env in local_envs]

    def heuristic_function(s):
        locations = env.state_to_locations(s)
        local_states = [local_envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
        # try something which is more SoC oriented
        return sum([local_v[i][local_states[i]] for i in range(env.n_agents)])

    return heuristic_function


def deterministic_relaxation_prioritized_value_iteration_heuristic(gamma: float,
                                                                   env: MapfEnv) -> Callable[[int], float]:
    deterministic_env = MapfEnv(env.grid,
                                env.n_agents,
                                env.agents_starts,
                                env.agents_goals,
                                0,
                                0,
                                env.reward_of_clash,
                                env.reward_of_goal,
                                env.reward_of_living)
    # TODO: consider using RTDP instead of PVI here, this is theoretically bad but practically may give better results
    policy = prioritized_value_iteration(gamma, deterministic_env, {})

    def heuristic_function(s):
        return policy.v[s]

    return heuristic_function


def calc_q_s_no_clash_possible(policy: RtdpPolicy, s: int):
    q_s_a = np.zeros(policy.env.nA)
    for a in range(policy.env.nA):
        for prob, next_state, reward, done in policy.env.P[s][a]:
            if reward == policy.env.reward_of_clash and done:
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

    while not done:
        # Choose action action for current state
        a = select_action(policy, s)

        # Simulate the step and sample a new state
        s, r, done, _ = policy.env.step(a)
        total_reward += r

        # Do a bellman update
        update(policy, s)

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
    info['iterations'] = []

    while True:
        info['iterations'].append({})
        iter_reward = rtdp_single_iteration(policy, select_action, update, info['iterations'][-1])
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


def stop_when_no_improvement_between_batches_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
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
    policy = RtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = time.time() - start

    # Run RTDP iterations
    for iter_count, reward in enumerate(rtdp_iterations_generator(policy, greedy_action, bellman_update, info),
                                        start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        if no_improvement_from_last_batch(policy, iter_count) or iter_count >= max_iterations:
            break

    return policy
