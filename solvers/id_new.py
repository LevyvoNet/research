"""New Independence Detection Algorithm which is kind of like CBS"""
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
                           dijkstra_distance_single_env)


class Conflict:
    def __init__(self, agents, in_conflict):
        self.agents = agents
        self.in_conflict = in_conflict


class ConflictOvercomePolicy(Policy):
    def __init__(self, env: MapfEnv, gamma: float, conflicts: List):
        super().__init__(env, gamma)
        self.conflicts = conflicts

        raise NotImplementedError()

    def _act_in_unfamiliar_state(self, s: int):
        local_locations = self.env.state_to_locations(s)
        local_states = [self.env.loc_to_int[loc] for loc in local_locations]

        # Check if the state is inside some conflict zone
        for conflict in self.conflicts:
            if conflict.in_conflict(local_states):

            # Use the in-conflict policy if both of the agents are in conflict
            if conflict.in_conflict[s_i] and conflict.in_conflict[s_j]:
                pass


def conflict_zone_frame(agent: int, in_conflict: np.array, env: MapfEnv):
    start_location = (env.agents_starts[agent],)

    # Calculate distance from conflict zone to start
    env_for_distance_calc = MapfEnv(env.grid,
                                    1,
                                    start_location,
                                    start_location,
                                    env.fail_prob,
                                    env.reward_of_clash,
                                    env.reward_of_goal,
                                    env.reward_of_living,
                                    env.optimization_criteria)
    distance = dijkstra_distance_single_env(env_for_distance_calc)
    distance_from_conflict_zone = np.ma.masked_array(distance, mask=in_conflict)
    # Get the conflict entrance location
    conflict_entrance_state = distance_from_conflict_zone.argmin()
    conflict_entrance_location = (env.valid_locations[conflict_entrance_state])

    # Calculate the distance from conflict entrance to conflict exit.
    # This is the location which is the closest to the entrance - the motivation is to be as less as possible
    # inside the conflict zone
    env_for_distance_calc = MapfEnv(env.grid,
                                    1,
                                    conflict_entrance_location,
                                    conflict_entrance_location,
                                    env.fail_prob,
                                    env.reward_of_clash,
                                    env.reward_of_goal,
                                    env.reward_of_living,
                                    env.optimization_criteria)
    distance = dijkstra_distance_single_env(env_for_distance_calc)
    in_conflict_not_entrance = in_conflict.copy()
    in_conflict_not_entrance[conflict_entrance_state] = False
    distance_from_conflict_zone = np.ma.masked_array(distance, mask=in_conflict_not_entrance)
    # Get the conflict exit location
    conflict_exit_state = distance_from_conflict_zone.argmin()
    conflict_exit_location = (env.valid_locations[conflict_exit_state])

    return conflict_entrance_location, conflict_exit_location


def plan_replaced_starts_goals(agents: List[int],
                               env: MapfEnv,
                               new_start_locations: List[tuple],
                               new_goal_locations: List[tuple],
                               planner: Callable[[MapfEnv, Dict], Policy],
                               info: Dict):
    # Calculate the new start and goal locations for all of the replaced agents
    new_agents_starts = env.agents_starts
    new_agents_goals = env.agents_goals
    for agent, new_start, new_goal in zip(agents, new_start_locations, new_goal_locations):
        new_agents_starts = new_agents_starts[:agent] + new_start + new_agents_starts[agent + 1:]
        new_agents_goals = new_agents_goals[:agent] + new_goal + new_agents_goals[agent + 1:]

    # Create the new env
    new_env = MapfEnv(env.grid,
                      env.n_agents,
                      new_agents_starts,
                      new_agents_goals,
                      env.fail_prob,
                      env.reward_of_clash,
                      env.reward_of_goal,
                      env.reward_of_living,
                      env.optimization_criteria)

    policy = planner(new_env, info)

    return policy


def merge_agents(low_level_planner: Callable[[MapfEnv, Dict], Policy],
                 low_level_merger: Callable[[MapfEnv, List, int, int, Policy, Policy, Dict], Policy],
                 env: MapfEnv,
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

            # Calculate the in_conflict matrix
            conflict_zone = conflict_details[0]
            in_conflict = np.full((len(env.valid_locations)), False)
            for s in conflict_zone[0]:
                in_conflict[s] = True

            # Find the conflict frame for each agent
            conflict_entrance_location_i, conflict_exit_location_i = conflict_zone_frame(i, in_conflict, env)
            conflict_entrance_location_j, conflict_exit_location_j = conflict_zone_frame(j, in_conflict, env)

            # Solve agent i reach to conflict zone
            info[f'{i}_start2conflict'] = {}
            policy_i2conflict = plan_replaced_starts_goals([i],
                                                           env,
                                                           [(env.agents_starts[i],)],
                                                           [conflict_entrance_location_i],
                                                           low_level_planner,
                                                           info[f'{i}2conflict'])

            # Solve agent j reach to conflict zone
            info[f'{j}_start2conflict'] = {}
            policy_j2conflict = plan_replaced_starts_goals([j],
                                                           env,
                                                           [(env.agents_starts[j],)],
                                                           [conflict_entrance_location_j],
                                                           low_level_planner,
                                                           info[f'{j}2conflict'])

            # Solve both agents getting out of conflict zone
            # TODO: this is not good enough, this kind of assumes that the agents are getting out of the conflict together.
            #   This is not that important, agent i could be out while j is in and they don't need to execute together.
            #   Maybe just change to optimization criteria for SoC is enough?
            info[f'{i}_{j}_conflict_zone'] = {}
            policy_in_conflict = plan_replaced_starts_goals(
                [i, j],
                env,
                [conflict_entrance_location_i, conflict_entrance_location_j],
                [conflict_exit_location_i, conflict_exit_location_j],
                low_level_planner,
                info[f'{i}_{j}_conflict_zone'])

            # Solve agent i reach from conflict zone exit to goal
            info[f'{i}_conflict2goal'] = {}
            policy_i2goal = plan_replaced_starts_goals([i],
                                                       env,
                                                       [conflict_exit_location_i],
                                                       [(env.agents_goals[i],)],
                                                       low_level_planner,
                                                       info[f'{i}_conflict2goal'])

            # Solve agent j reach from conflict zone exit to goal
            info[f'{j}_conflict2goal'] = {}
            policy_j2goal = plan_replaced_starts_goals([j],
                                                       env,
                                                       [conflict_exit_location_j],
                                                       [(env.agents_goals[j],)],
                                                       low_level_planner,
                                                       info[f'{j}_conflict2goal'])

            # Merge all of the partial policies to a joint one which depends on the state
            policy = ConflictOvercomePolicy()

        policies.append(policy)

    return CrossedPolicy(env, policies, new_agents_groups)


def _default_low_level_merger_abstract(low_level_planner,
                                       env,
                                       agents_groups,
                                       group1,
                                       group2,
                                       policy1,
                                       policy2,
                                       info):
    """This will casue ID to behave the old way - just solve from the beginning"""

    return low_level_planner(env, info)


def new_id(
        low_level_planner: Callable[[MapfEnv, Dict], Policy],
        low_level_merger: Callable[[MapfEnv, List, int, int, Policy, Policy, Dict], Policy],
        env: MapfEnv,
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
                                         low_level_planner) if low_level_merger is None else low_level_merger

    start = time.time()  # TODO: use a decorator for updating info with time measurement
    agents_groups = [[i] for i in range(env.n_agents)]
    info['iterations'] = []
    curr_iter_info = {}
    info['iterations'].append(curr_iter_info)
    curr_iter_info['agent_groups'] = agents_groups
    curr_iter_info['joint_policy'] = {}
    curr_joint_policy = solve_independently_and_cross(env,
                                                      agents_groups,
                                                      low_level_planner,
                                                      curr_iter_info['joint_policy'])

    conflict = detect_conflict(env, curr_joint_policy, curr_iter_info)
    while conflict:
        i, j, conflict_details = conflict
        # merge groups of i and j
        curr_iter_info = {}
        info['iterations'].append(curr_iter_info)
        curr_iter_info['joint_policy'] = {}
        curr_joint_policy = merge_agents(low_level_planner,
                                         low_level_merger,
                                         env,
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
    info['ID_time'] = round(end - start, 2)
    return curr_joint_policy
