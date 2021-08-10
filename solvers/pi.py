import time
import numpy as np
import math
import copy
from typing import Dict
from collections import defaultdict

from solvers import V_TYPE_SIZE, V_TYPE, MAXIMUM_RAM
from gym_mapf.envs.mapf_env import MapfEnv
from solvers.utils import Policy, ValueFunctionPolicy


def one_step_lookahead(env, state, v, discount_factor=1.0):
    """
    Helper function to  calculate state-value function

    Arguments:
        env: openAI GYM Enviorment object
        state: state to consider
        V: Estimated Value for each state. Vector of length nS
        discount_factor: MDP discount factor

    Return:
        action_values: Expected value of each action in a state. Vector of length nA
    """
    # initialize vector of action values
    action_values = defaultdict(lambda: 0.0)

    # loop over the actions we can take in an environment
    for action in env.action_space:
        # loop over the P_sa distribution.
        for probability, next_state, reward, done in env.P[state][action]:
            # if we are in state s and take action a. then sum over all the possible states we can land into.
            if env.is_collision_transition(state, next_state):
                action_values[action] = -math.inf
                break

            action_values[action] += probability * (reward + (discount_factor * v[next_state]))

    return action_values


def update_policy(env, policy, V, discount_factor):
    """
    Helper function to update a given policy based on given value function.

    Arguments:
        env: openAI GYM Environment object.
        policy: policy to update.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy: Updated policy based on the given state-Value function 'V'.
    """
    for state in env.observation_space:
        # for a given state compute state-action value.
        action_values = one_step_lookahead(env, state, V, discount_factor)

        # choose the action which maximize the state-action value.
        policy[state] = max(action_values, key=action_values.get)

    return policy


def policy_eval(env: MapfEnv, policy: Dict, v: Dict, discount_factor: float):
    """
    Helper function to evaluate a policy.

    Arguments:
        env: openAI gym env object.
        policy: policy to evaluate.
        V: Estimated Value for each state. Vector of length nS.
        discount_factor: MDP discount factor.
    Return:
        policy_value: Estimated value of each state following a given policy and state-value 'V'.

    """
    policy_value = np.zeros(env.nS, dtype=V_TYPE)
    new_v = {s: 0 for s in env.observation_space}
    for state, action in policy.items():
        for probablity, next_state, reward, info in env.P[state][action]:
            new_v[state] += probablity * (reward + (discount_factor * v[next_state]))

    return new_v


def policy_iteration(gamma: float, env: MapfEnv, info: Dict, **kwargs) -> Policy:
    gamma = kwargs.get('gamma', 1.0)
    max_iteration = 1000

    # intialize the state-Value function
    if V_TYPE_SIZE * env.nS > MAXIMUM_RAM:
        info['end_reason'] = "out_of_memory"
        return None

    v = {s: 0 for s in env.observation_space}
    policy = ValueFunctionPolicy(env, 1.0)

    # initialize a random policy
    policy_curr = {s: env.action_space.sample() for s in env.observation_space}
    policy_prev = copy.copy(policy_curr)

    for i in range(max_iteration):
        # evaluate given policy
        start = time.time()
        v = policy_eval(env, policy_curr, v, gamma)

        # improve policy
        policy_curr = update_policy(env, policy_curr, v, gamma)

        # if policy not changed over 10 iterations it converged.
        if i % 10 == 0:
            if policy_curr == policy_prev:
                # print('policy iteration converged at iteration %d' % (i + 1))
                break
            policy_prev = copy.copy(policy_curr)

        # print(f'PI: iteration {i + 1} took {time.time() - start} seconds')

    policy.policy_cache = policy_curr
    policy.v = policy_eval(env, policy_curr, v, gamma)

    return policy
