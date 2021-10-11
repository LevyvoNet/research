"""New Independence Detection Algorithm which is kind of like CBS"""
import collections
import functools
import time
import numpy as np
from typing import Dict, Callable, List

from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria
from solvers.utils import (detect_conflict,
                           solve_independently_and_cross,
                           get_local_view,
                           Policy,
                           group_of_agent,
                           merge_agent_groups,
                           CrossedPolicy,
                           dijkstra_distance_single_env,
                           get_reachable_states)


def problem_factorization(joint_policy, env, info):
    state_to_agents = collections.defaultdict(set)

    for agent in range(env.n_agents):
        info[f'reachable_{agent}'] = {}
        reachable_states = get_reachable_states(env, joint_policy, agent, info[f'reachable_{agent}'])
        for s in reachable_states:
            state_to_agents[s].add(agent)

    return state_to_agents


Factor = collections.namedtuple("Factor", ['agents', 'env'])


def solve_factors(factors: List[Factor], low_level_planner):
    raise NotImplementedError()


class FactorizationPolicy(Policy):
    def __init__(self, env: MapfEnv, gamma: float):
        super().__init__(env, gamma)

    def _act_in_unfamiliar_state(self, s: int):
        local_states = [self.env.loc_to_int[loc] for loc in self.env.state_to_locations(s)]


def new_id(
        low_level_planner: Callable[[MapfEnv, Dict], Policy],
        low_level_merger: Callable[[MapfEnv, List, int, int, Policy, Policy, Dict], Policy],
        env: MapfEnv,
        info: Dict, **kwargs
) -> Policy:
    # First, solve for each agent independently
    info['independent_policy'] = {}
    curr_joint_policy = solve_independently_and_cross(env,
                                                      [[i] for i in range(env.n_agents)],
                                                      low_level_planner,
                                                      info['independent_policy'])

    # Factorize the problem according to conflicts
    info['factorization'] = {}
    factors = problem_factorization(curr_joint_policy, env, info['factorization'])

    # Re-plan according to the found conflicts
    policy = solve_factors(factors)
