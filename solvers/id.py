"""Independence Detection Algorithm"""
import functools
import time
import copy
from typing import Dict, Callable, List

from gym_mapf.envs.mapf_env import MapfEnv
from solvers.utils import (detect_conflict,
                           solve_independently_and_cross,
                           get_local_view,
                           Policy,
                           group_of_agent,
                           merge_agent_groups,
                           CrossedPolicy)


class IdPolicy(Policy):
    def __init__(self, env, gamma, low_level_policy_creator: Callable[[MapfEnv, float], Policy], low_level_merger,
                 name: str = ''):
        super(IdPolicy, self).__init__(env, gamma, name)
        self.low_level_policy_creator = low_level_policy_creator
        self.low_level_merger = low_level_merger
        self.joint_policy = None

    def train(self, *args, **kwargs):
        self.joint_policy = id(self.low_level_policy_creator, self.low_level_merger, self.env, self.gamma, self.info)
        return self

    def _act_in_unfamiliar_state(self, s: int):
        return self.joint_policy._act_in_unfamiliar_state(s)

    def train_info(self):
        ret = {}

        # Number of found conflicts
        ret['n_conflicts'] = len(self.info['iterations']) - 1

        # Total time for conflicts detection,
        # In case of all agents merged, the last iteration might not have a detect_conflict_time and therefore the 'get'.
        ret['conflict_detection_time'] = round(sum([self.info['iterations'][i].get('detect_conflict_time', 0)
                                                    for i in range(len(self.info['iterations']))]), 1)

        # This time mostly matters for heuristics calculation (on RTDP for example)
        ret['solver_init_time'] = round(sum(
            [sum([self.info['iterations'][j]['joint_policy'][key]['initialization_time']
                  for key in self.info['iterations'][j]['joint_policy'].keys() if all([key.startswith('['),
                                                                                       key.endswith(']')])
                  ])
             for j in range(len(self.info['iterations']))]), 1)

        # Set train time
        ret['train_time'] = self.info['train_time']

        return ret


def merge_agents(low_level_merger: Callable[[MapfEnv, float, List, int, int, Policy, Policy], Policy],
                 env: MapfEnv,
                 gamma: float,
                 agents_groups: List,
                 i: int,
                 j: int,
                 conflict_details: tuple,
                 joint_policy: CrossedPolicy,
                 info):
    # merge groups of i and j
    old_groups = agents_groups[:]
    new_agents_groups = merge_agent_groups(agents_groups,
                                           group_of_agent(agents_groups, i),
                                           group_of_agent(agents_groups, j))

    policies = []
    for group in new_agents_groups:
        info[f'{group}'] = {}
        if group in old_groups:
            # We already have a solution for this group, don't calculate it again.
            policy = joint_policy.policies[old_groups.index(group)]
            info[f'{group}']['total_time'] = 0
            info[f'{group}']['initialization_time'] = 0
        else:
            # We need to solve a new environment, use previous solutions as a heuristic
            old_group_i_idx = group_of_agent(old_groups, i)
            old_group_j_idx = group_of_agent(old_groups, j)
            policy = low_level_merger(get_local_view(env, group),
                                      gamma,
                                      old_groups,
                                      old_group_i_idx,
                                      old_group_j_idx,
                                      joint_policy.policies[old_group_i_idx],
                                      joint_policy.policies[old_group_j_idx])
            info[f'{group}'] = policy.info

        policies.append(policy)

    return CrossedPolicy(env, gamma, policies, new_agents_groups)


def _default_low_level_merger_abstract(low_level_policy_creator,
                                       env,
                                       gamma,
                                       agents_groups,
                                       group1,
                                       group2,
                                       policy1,
                                       policy2):
    """This will casue ID to behave the old way - just solve from the beginning"""
    policy = copy.copy(low_level_policy_creator)
    policy = low_level_policy_creator(env, gamma)
    policy.train()
    return policy


def id(
        low_level_policy_creator: Callable[[MapfEnv, float], Policy],
        low_level_merger: Callable[[MapfEnv, float, List, int, int, Policy, Policy, Dict], Policy],
        env: MapfEnv,
        gamma: float,
        info: Dict, **kwargs
) -> Policy:
    """Solve MAPF gym environment with ID algorithm.
    Args:
        low_level_merger: (MapfEnv, List, int, int, Policy, Policy, info): curried function which merges two groups
            and return a new joint policy where the the received groups are planned together.
        low_level_planner ((MapfEnv)->Policy)): curried function which receives an env and returns a policy
        env (MapfEnv): mapf env
        info (dict): information about the run. For ID it will return information about conflicts
            detected during the solving.
    Returns:
          function int->int. The optimal policy, function from state to action.
    """
    # TODO: delete eventually
    low_level_merger = functools.partial(_default_low_level_merger_abstract,
                                         low_level_policy_creator) if low_level_merger is None else low_level_merger

    start = time.time()  # TODO: use a decorator for updating info with time measurement
    agents_groups = [[i] for i in range(env.n_agents)]
    info['iterations'] = []
    curr_iter_info = {}
    info['iterations'].append(curr_iter_info)
    curr_iter_info['agent_groups'] = agents_groups
    curr_iter_info['joint_policy'] = {}
    curr_joint_policy = solve_independently_and_cross(env,
                                                      agents_groups,
                                                      low_level_policy_creator,
                                                      gamma,
                                                      curr_iter_info['joint_policy'])

    conflict = detect_conflict(env, curr_joint_policy, curr_iter_info)
    while conflict:
        i, j, conflict_details = conflict
        # merge groups of i and j
        curr_iter_info = {}
        info['iterations'].append(curr_iter_info)
        curr_iter_info['joint_policy'] = {}
        curr_joint_policy = merge_agents(low_level_merger,
                                         env,
                                         gamma,
                                         agents_groups,
                                         i,
                                         j,
                                         conflict_details,
                                         curr_joint_policy,
                                         curr_iter_info['joint_policy'])

        if len(agents_groups) == 1:
            # we have merged all of the agents and conflict is not possible
            break

        # find a new conflict
        conflict = detect_conflict(env, curr_joint_policy, **{'info': curr_iter_info})

    end = time.time()
    info['train_time'] = round(end - start, 2)
    return curr_joint_policy
