import collections
import time
import itertools
import math

from gym_mapf.envs.grid import MapfGrid, ObstacleCell
from gym_mapf.envs.utils import create_mapf_env, get_local_view
from gym_mapf.envs.mapf_env import (OptimizationCriteria,
                                    MapfEnv,
                                    integer_action_to_vector,
                                    vector_action_to_integer,
                                    ACTIONS)

from available_solvers import *
from solvers.utils import (solve_independently_and_cross,
                           get_reachable_states,
                           couple_detect_conflict,
                           Policy,
                           CrossedPolicy,
                           evaluate_policy)

ConflictArea = collections.namedtuple('ConflictArea', ['top_row', 'bottom_row', 'left_col', 'right_col'])


def distance(loc1: tuple, loc2: tuple):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


def cast_location(top_left: tuple, loc: tuple):
    return (loc[0] - top_left[0], loc[1] - top_left[1])


def inside(loc: tuple, area: ConflictArea):
    return all([loc[0] <= area.bottom_row,
                loc[0] >= area.top_row,
                loc[1] <= area.right_col,
                loc[1] >= area.left_col])


def divide_to_groups(locations, k):
    n_agents = len(locations)
    neighbours = {}
    # For each agent, calculate its neighbours
    for agent in range(n_agents):
        neighbours[agent] = []
        for other_agent in range(n_agents):
            if agent == other_agent:
                continue

            if distance(locations[agent], locations[other_agent]) <= k:
                neighbours[agent].append(other_agent)

    # Now run a DFS-like search in order to determine the connectivity components of the graph.
    groups = []
    for agent in range(n_agents):
        # If the agent is already part of a group (black), continue to the next one
        if [x for x in filter(lambda g: agent in g, groups)]:
            continue

        # Calculate the connectivity component of the current agent
        new_group = []
        stack = [agent]
        while stack:
            curr_agent = stack.pop(0)
            # All of the neighbours are white because if one was black the current agent will be black as well
            for neighbour in neighbours[curr_agent]:
                if neighbour not in new_group:
                    stack.insert(0, neighbour)

            new_group.append(curr_agent)

        groups.append(sorted(new_group))

    return groups


class OnlineReplanPolicy(Policy):
    def __init__(self, env: MapfEnv, gamma: float, k: int, self_policy: CrossedPolicy, low_level_planner, info):
        super().__init__(env, gamma)
        self.k = k
        self.online_policies = []
        self.self_policy = self_policy
        self.single_env = self.self_policy.policies[0].env
        self.low_level_planner = low_level_planner
        self.info = info
        self.replans = collections.defaultdict(partial(collections.defaultdict, dict))

    def reset(self):
        super().reset()
        self.replans = collections.defaultdict(partial(collections.defaultdict, dict))

    def replan_for_group(self, group, locations, info):
        # Determine the borders
        top_row = min([locations[agent][0] for agent in group])
        bottom_row = max([locations[agent][0] for agent in group])
        left_col = min([locations[agent][1] for agent in group])
        right_col = max([locations[agent][1] for agent in group])

        # Pad so the created area will be at least kxk
        extra_rows = max(self.k - (bottom_row - top_row + 1), 0)
        extra_cols = max(self.k - (right_col - left_col + 1), 0)

        top_row = top_row - math.floor(extra_rows / 2)
        left_col = left_col - math.floor(extra_cols / 2)
        bottom_row = bottom_row + math.ceil(extra_rows / 2)
        right_col = right_col + math.ceil(extra_cols / 2)

        # Calculate the conflict area locations
        conflict_area_locations = []
        for row in range(bottom_row, top_row + 1):
            for col in range(left_col, right_col + 1):
                conflict_area_locations.append((row, col))

        # Calculate the candidates for goal states - these are the states which are in the first layer outside of the
        # conflict area
        n_rows = bottom_row - top_row + 1
        n_cols = right_col - left_col + 1
        possible_goal_locations = []
        if top_row > 0:
            n_rows += 1
            for col in range(left_col, right_col + 1):
                possible_goal_locations.append((top_row - 1, col))

        if bottom_row < self.env.grid.max_row:
            n_rows += 1
            for col in range(left_col, right_col + 1):
                possible_goal_locations.append((bottom_row + 1, col))

        if left_col > 0:
            n_cols += 1
            for row in range(bottom_row, top_row + 1):
                possible_goal_locations.append((row, left_col - 1))

        if right_col < self.env.grid.max_col:
            n_cols += 1
            for row in range(bottom_row, top_row + 1):
                possible_goal_locations.append((row, right_col + 1))

        top_left = (max(0, top_row - 1), max(0, left_col - 1))
        goal_locs = [max(possible_goal_locations,
                         key=lambda loc: self.self_policy.policies[agent].v[
                             self.single_env.locations_to_state((loc,))])
                     for agent in group]
        goal_locs = tuple([cast_location(top_left, goal_loc) for goal_loc in goal_locs])
        start_locs = tuple([cast_location(top_left, locations[agent]) for agent in group])
        conflict_area = ConflictArea(top_row, bottom_row, left_col, right_col)

        # Create the grid map
        grid_map = ['.' * n_cols] * n_rows
        for loc in conflict_area_locations:
            if self.env.grid[loc[0]][loc[1]] is ObstacleCell:
                casted_loc = cast_location(top_left, loc)
                grid_map[casted_loc[0]][casted_loc[1]] = '@'

        grid = MapfGrid(grid_map)
        env = MapfEnv(grid,
                      len(group),
                      start_locs,
                      goal_locs,
                      self.env.fail_prob,
                      self.env.reward_of_clash,
                      self.env.reward_of_goal,
                      self.env.reward_of_living,
                      self.env.optimization_criteria)

        self.info[f'{group}_{top_left}'] = {}
        joint_policy = self.low_level_planner(env, self.info[f'{group}_{top_left}'])

        self.replans[tuple(group)][conflict_area] = joint_policy
        return conflict_area, joint_policy

    def select_action_for_group(self, group, locations):
        # If this is an independent agent, just fetch the action from its self policy
        if len(group) == 1:
            agent = group[0]
            s = self.single_env.loc_to_int[locations[agent]]
            return [ACTIONS[self.self_policy.policies[agent].act(s)]]

        # Look for an existing re-plan for the current group according to its location
        found = False
        for conflict_area in self.replans[tuple(group)].keys():
            if all([inside(locations[agent], conflict_area) for agent in group]):
                # Found one, return the actions from it
                joint_policy = self.replans[tuple(group)][conflict_area]
                found = True
                break

        # If there isn't a policy for the current situation, replan and get one.
        if not found:
            conflict_area, joint_policy = self.replan_for_group(group, locations, self.info)

        # Retrieve the actions for the agents from the joint policy in the casted env
        joint_location = [locations[agent] for agent in group]
        # TODO: calculate the top_left in a more generic way (by using a function or something), this is counting
        #   on the goal states being on the first layer after the conflict area
        top_left = (max(0, conflict_area.top_row - 1), max(0, conflict_area.left_col - 1))
        joint_location_casted = [cast_location(top_left, loc) for loc in joint_location]
        joint_location_casted = tuple(joint_location_casted)
        joint_state = joint_policy.env.locations_to_state(joint_location_casted)
        joint_action = joint_policy.act(joint_state)
        joint_action_vector = integer_action_to_vector(joint_action, len(group))
        return joint_action_vector

    def _act_in_unfamiliar_state(self, s: int):
        locations = self.env.state_to_locations(s)
        joint_action_vector = [None] * self.env.n_agents
        groups = divide_to_groups(locations, self.k)
        for group in groups:
            group_joint_action_vector = self.select_action_for_group(group, locations)
            for i, agent in enumerate(group):
                joint_action_vector[agent] = group_joint_action_vector[i]

        return vector_action_to_integer(joint_action_vector)


def online_replan(low_level_planner, k, env, info):
    self_groups = [[i] for i in range(env.n_agents)]
    info['independent_policies'] = {}

    # Plan for each agent independently
    self_policy = solve_independently_and_cross(env,
                                                self_groups,
                                                low_level_planner,
                                                info['independent_policies'])
    info['online_policy'] = {}
    return OnlineReplanPolicy(env,
                              self_policy.gamma,
                              k,
                              self_policy,
                              low_level_planner,
                              info['online_policy'])


def main():
    env = create_mapf_env('empty-48-48', 25, 4, 0.2, -1000, 0, -1, OptimizationCriteria.Makespan)
    low_level_solver_describer = long_rtdp_stop_no_improvement_sum_rtdp_dijkstra_heuristic_describer
    info = {}

    k = 3
    policy = online_replan(low_level_solver_describer.func, k, env, info)

    # import ipdb
    # ipdb.set_trace()

    eval_info = evaluate_policy(policy, 100, 100, 0)

    print(f"MDR: {eval_info['MDR']}")
    print(f"success_rate: {eval_info['success_rate']}")
    print(f"clashed: {eval_info['clashed']}")
    print(f"mean time: {eval_info['mean_time']}")
    print(f"replans: {eval_info['n_replans']}")


if __name__ == '__main__':
    main()
