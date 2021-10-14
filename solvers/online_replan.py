import collections
import time
import itertools

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

    def replan_for_couple(self, a1, a2, locations, info):
        loc1 = locations[a1]
        loc2 = locations[a2]

        # Determine the borders according to loc1
        # TODO: what if they row/col are equal? at the current we arbitrary treat it as the same as larger.
        # Determine the row borders
        if loc2[0] < loc1[0]:
            top_row = max(0, loc1[0] - self.k)
            bottom_row = loc1[0]
        else:
            top_row = loc1[0]
            bottom_row = min(self.env.grid.max_row, loc1[0] + self.k)

        # Determine the column borders
        if loc2[1] > loc1[1]:
            right_col = min(self.env.grid.max_col, loc1[1] + self.k)
            left_col = loc1[1]
        else:
            left_col = max(0, loc1[1] - self.k)
            right_col = loc1[1]

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

        # Calculate the goal location for each of the agents, these are the ones with the largest V according to the
        # self policies (TODO: this is not optimal, think about it)
        # TODO: how to combine exit from the conflict area as fast as possible but also close to the goal (just return back is not good)
        goal_loc1 = max(possible_goal_locations,
                        key=lambda loc: self.self_policy.policies[a1].v[self.single_env.locations_to_state((loc,))])
        goal_loc2 = max(possible_goal_locations,
                        key=lambda loc: self.self_policy.policies[a2].v[self.single_env.locations_to_state((loc,))])
        top_left = (max(0, top_row - 1), max(0, left_col - 1))
        goal_locs = (cast_location(top_left, goal_loc1), cast_location(top_left, goal_loc2))
        start_locs = (cast_location(top_left, locations[a1]), cast_location(top_left, locations[a2]))
        conflict_area = ConflictArea(top_row, bottom_row, left_col, right_col)

        print(f'Re-planned for {a1} and {a2} with conflict {conflict_area}')

        grid_map = ['.' * n_cols] * n_rows
        for loc in conflict_area_locations:
            if self.env.grid[loc[0]][loc[1]] is ObstacleCell:
                casted_loc = cast_location(top_left, loc)
                grid_map[casted_loc[0]][casted_loc[1]] = '@'

        grid = MapfGrid(grid_map)
        env = MapfEnv(grid,
                      2,
                      start_locs,
                      goal_locs,
                      self.env.fail_prob,
                      self.env.reward_of_clash,
                      self.env.reward_of_goal,
                      self.env.reward_of_living,
                      self.env.optimization_criteria)

        self.info[f'{a1}_{a2}_{top_left}'] = {}
        joint_policy = self.low_level_planner(env, self.info[f'{a1}_{a2}_{top_left}'])

        self.replans[tuple(sorted((a1, a2)))][conflict_area] = joint_policy
        return conflict_area, joint_policy

    def select_action_for_agent(self, agent, locations):
        # TODO: make this function support multiple conflicts (with many agents, not just one)
        for other_agent in range(self.env.n_agents):
            if agent == other_agent:
                continue

            if distance(locations[agent], locations[other_agent]) <= self.k:
                # Check if we already have a joint policy for these agents
                found = False
                for conflict_area in self.replans[tuple(sorted((agent, other_agent)))].keys():
                    if inside(locations[agent], conflict_area) and inside(locations[other_agent], conflict_area):
                        joint_policy = self.replans[tuple(sorted((agent, other_agent)))][conflict_area]
                        found = True
                        break

                if not found:
                    # There is no online policy for the current situation, re-plan it!
                    conflict_area, joint_policy = self.replan_for_couple(agent, other_agent, locations, self.info)

                agent_idx = sorted((agent, other_agent)).index(agent)
                other_agent_idx = sorted((agent, other_agent)).index(other_agent)

                joint_location = [None] * 2
                joint_location[agent_idx] = locations[agent]
                joint_location[other_agent_idx] = locations[other_agent]
                top_left = (max(0, conflict_area.top_row - 1), max(0, conflict_area.left_col - 1))
                joint_location_casted = [cast_location(top_left, loc) for loc in joint_location]
                joint_location_casted = tuple(joint_location_casted)
                joint_state = joint_policy.env.locations_to_state(joint_location_casted)
                joint_action = joint_policy.act(joint_state)
                joint_action_vector = integer_action_to_vector(joint_action, 2)
                return joint_action_vector[agent_idx]

            # No conflicts, return from self policy
            s = self.single_env.locations_to_state((locations[agent],))
            return ACTIONS[self.self_policy.policies[agent].act(s)]

    def _act_in_unfamiliar_state(self, s: int):
        locations = self.env.state_to_locations(s)
        joint_action_vector = [None] * self.env.n_agents
        for curr_agent in range(self.env.n_agents):
            local_action = self.select_action_for_agent(curr_agent, locations)
            joint_action_vector[curr_agent] = local_action

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
    env = create_mapf_env('empty-48-48', 25, 2, 0.2, -1000, 0, -1, OptimizationCriteria.Makespan)
    low_level_solver_describer = long_ma_rtdp_sum_rtdp_dijkstra_describer
    info = {}

    k = 3
    policy = online_replan(low_level_solver_describer.func, k, env, info)

    eval_info = evaluate_policy(policy, 1000, 100, 0, True)

    print(f"MDR: {eval_info['MDR']}")
    print(f"success_rate: {eval_info['success_rate']}")
    print(f"clashed: {eval_info['clashed']}")
    print(f"mean time: {eval_info['mean_time']}")
    print(f"replans: {eval_info['n_replans']}")


if __name__ == '__main__':
    main()
