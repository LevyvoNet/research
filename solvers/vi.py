import time
import numpy as np
import math
import copy
from collections import defaultdict
from typing import Dict

from solvers import V_TYPE, V_TYPE_SIZE, MAXIMUM_RAM
from solvers.utils import safe_actions, ValueFunctionPolicy, Policy
from gym_mapf.envs.mapf_env import MapfEnv, MultiAgentState, MultiAgentAction

from gym_mapf.envs.grid import *


def get_layers(env):
    layers = []
    visited_states = set()
    iter_states = set(env.predecessors(env.goal_state))
    next_iter_states = set(iter_states)
    while len(visited_states) < env.nS:
        iter_states = set(next_iter_states)
        next_iter_states = set()
        for s in iter_states:
            visited_states.add(s)
            next_iter_states = next_iter_states.union(env.predecessors(s))

        next_iter_states = next_iter_states.difference(visited_states)

        layers.append(iter_states)

    return layers


def value_iteration(gamma: float, env: MapfEnv, info: Dict, **kwargs) -> ValueFunctionPolicy:
    """ Value-iteration algorithm"""
    info['converged'] = False
    info['n_iterations'] = 0
    info['iterations'] = {}
    gamma = kwargs.get('gamma', 1.0)
    if V_TYPE_SIZE * env.nS > MAXIMUM_RAM:
        info['end_reason'] = "out_of_memory"
        return None

    policy = ValueFunctionPolicy(env, gamma)
    v = {s.hash_value: 0 for s in env.observation_space}  # initialize value-function
    max_iterations = 1000
    eps = 1e-2
    q_sa = 0
    real_start = time.time()
    for i in range(max_iterations):
        prev_v = copy.copy(v)
        start = time.time()
        diff = 0
        info['iterations'][info['n_iterations']] = {}
        for s in env.observation_space:
            q_s = {}
            max_value = -math.inf

            for a in env.action_space:
                q_sa = 0
                for p, s_, r, done in env.P[s][a]:
                    if env.is_collision_transition(s, s_) and done:
                        # This is a dangerous action which might get to conflict
                        q_sa = -math.inf
                        break
                    q_sa += p * (r + prev_v[s_.hash_value])

                # Check if the current action is the best. If so, save it in the policy.
                if q_sa > max_value:
                    max_value = q_sa
                    policy.policy_cache[s] = a
                    v[s.hash_value] = q_sa

                q_s[a] = q_sa

            diff += math.fabs(
                0 if math.isnan(prev_v[s.hash_value] - v[s.hash_value]) else prev_v[s.hash_value] - v[s.hash_value])

        # # debug print
        # if i % 10 == 0:
        #     print(v.values())

        # print(f'VI: iteration {i + 1} took {time.time() - start} seconds')
        info['iterations'][info['n_iterations']]['total_time'] = time.time() - start
        info['n_iterations'] = i + 1
        if diff <= eps:
            # debug print
            # print('value iteration converged at iteration# %d.' % (i + 1))
            info['converged'] = True
            break

    policy.v = v

    end = time.time()
    info['VI_time'] = round(end - real_start, 2)

    return policy


def prioritized_value_iteration(gamma: float, env: MapfEnv, info: Dict, **kwargs) -> ValueFunctionPolicy:
    info['converged'] = False
    info['n_iterations'] = 0
    info['iterations'] = {}
    gamma = kwargs.get('gamma', 1.0)
    real_start = time.time()

    if V_TYPE_SIZE * env.nS > MAXIMUM_RAM:
        info['end_reason'] = "out_of_memory"
        return None

    policy = ValueFunctionPolicy(env, gamma)
    v = {s.hash_value: 0 for s in env.observation_space}  # initialize value-function
    max_iterations = 1000
    eps = 1e-2
    q_sa_a = 0
    layers = get_layers(env)
    for i in range(max_iterations):
        prev_v = copy.copy(v)
        start = time.time()
        diff = 0
        info['iterations'][info['n_iterations']] = {}
        for layer in layers:
            for s in layer:
                q_s = {}
                max_value = -math.inf
                for a in env.action_space:
                    q_sa = 0
                    for p, s_, r, done in env.P[s][a]:
                        if env.is_collision_transition(s, s_) and done:
                            # This is a dangerous action which might get to conflict
                            q_sa = -math.inf
                            break
                        q_sa += p * (r + prev_v[s_.hash_value])

                    # Check if the current action is the best. If so, save it in the policy.
                    if q_sa > max_value:
                        max_value = q_sa
                        policy.policy_cache[s] = a
                        v[s.hash_value] = q_sa

                    q_s[a] = q_sa

                diff += math.fabs(prev_v[s.hash_value] - v[s.hash_value])

        # # debug print
        # if i % 10 == 0:
        #     print(v)
        # print(f'PVI: iteration {i + 1} took {time.time() - start} seconds')
        info['iterations'][info['n_iterations']]['total_time'] = time.time() - start
        info['n_iterations'] = i + 1
        if diff <= eps:
            # debug print
            # print('prioritized value iteration converged at iteration# %d.' % (i + 1))
            info['converged'] = True
            break

    policy.v = v
    end = time.time()
    info['prioritized_VI_time'] = round(end - real_start, 2)
    return policy
