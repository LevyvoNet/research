import functools
import unittest
from functools import partial

from gym_mapf.envs.utils import MapfGrid, create_mapf_env
from gym_mapf.envs.grid import SingleAgentState, SingleAgentAction
from gym_mapf.envs.mapf_env import (MapfEnv, MultiAgentState, MultiAgentAction, OptimizationCriteria)

from solvers import (value_iteration,
                     id,
                     fixed_iterations_count_rtdp)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          fixed_iterations_rtdp_merge,
                          solution_heuristic_min,
                          solution_heuristic_sum)
from solvers.utils import solve_independently_and_cross, evaluate_policy


class IdTests(unittest.TestCase):
    def test_corridor_switch_independent_vs_merged(self):
        grid = MapfGrid(['...',
                         '@.@'])
        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 0)}, grid)
        env = MapfEnv(grid, 2, start_state, goal_state, 0.1, -1, 1, -0.01, OptimizationCriteria.Makespan)

        vi_plan_func = partial(value_iteration, 1.0)
        independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], vi_plan_func, {})
        merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], vi_plan_func, {})

        interesting_state = MultiAgentState({0: SingleAgentState(1, 1), 1: SingleAgentState(0, 1)}, grid)

        expected_possible_actions = [MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.UP}),
                                     MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.UP})]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy.act(interesting_state),
                         MultiAgentAction({0: SingleAgentAction.UP, 1: SingleAgentAction.LEFT}))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)

    def test_two_columns_independent_vs_merged(self):
        grid = MapfGrid(['..',
                         '..',
                         '..'])
        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 1)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 1)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0.1, -1, 1, -0.1, OptimizationCriteria.Makespan)

        vi_plan_func = partial(value_iteration, 1.0)
        independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], vi_plan_func, {})
        merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], vi_plan_func, {})

        interesting_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 1)}, grid)
        expected_possible_actions = [MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.RIGHT})]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy.act(interesting_state),
                         MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.DOWN}))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)

    def test_narrow_empty_grid(self):
        grid = MapfGrid(['....'])

        agents_starts = ((0, 1), (0, 2))
        agents_goals = ((0, 0), (0, 3))

        start_state = MultiAgentState({0: SingleAgentState(0, 1), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 3)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0.1, -1, 1, -0.01, OptimizationCriteria.Makespan)

        vi_plan_func = partial(value_iteration, 1.0)
        joint_policy = id(vi_plan_func, None, env, {})

        expected_action = MultiAgentAction({0: SingleAgentAction.LEFT, 1: SingleAgentAction.RIGHT})
        self.assertEqual(joint_policy.act(env.s), expected_action)

    def test_env_with_switch_conflict_solved_properly(self):
        env = create_mapf_env('room-32-32-4', 9, 2, 0, 0, -1000, 0, -1)
        gamma = 1.0
        n_iterations = 100

        rtdp_plan_func = partial(fixed_iterations_count_rtdp,
                                 partial(local_views_prioritized_value_iteration_min_heuristic, gamma),
                                 gamma,
                                 n_iterations)
        rtdp_merge_func = functools.partial(fixed_iterations_rtdp_merge, solution_heuristic_min, gamma, n_iterations)
        policy = id(rtdp_plan_func, rtdp_merge_func, env, {})

        reward, clashed, _ = evaluate_policy(policy, 1, 1000)

        self.assertFalse(clashed)

        # Assert that the solution is reasonable (actually solving)
        self.assertGreater(reward, -1000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
