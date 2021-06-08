import functools
import unittest
from functools import partial

from gym_mapf.envs.utils import MapfGrid, create_mapf_env
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, RIGHT, LEFT, STAY)

from solvers import (value_iteration,
                     id,
                     fixed_iterations_count_rtdp)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          fixed_iterations_rtdp_merge,
                          solution_heuristic_min,
                          solution_heuristic_sum)
from solvers.utils import solve_independently_and_cross, evaluate_policy


class IdTests(unittest.TestCase):
    # def test_corridor_switch_independent_vs_merged(self):
    #     grid = MapfGrid(['...',
    #                      '@.@'])
    #     agents_starts = ((0, 0), (0, 2))
    #     agents_goals = ((0, 2), (0, 0))
    #
    #     env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)
    #
    #     vi_plan_func = partial(value_iteration, 1.0)
    #     independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], vi_plan_func, {})
    #     merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], vi_plan_func, {})
    #
    #     interesting_state = env.locations_to_state(((1, 1), (0, 1)))
    #
    #     expected_possible_actions = [vector_action_to_integer((STAY, UP)),
    #                                  vector_action_to_integer((DOWN, UP))]
    #
    #     # Assert independent_joint_policy just choose the most efficient action
    #     self.assertEqual(independent_joiont_policy.act(interesting_state), vector_action_to_integer((UP, LEFT)))
    #
    #     # Assert merged_joint_policy avoids collision
    #     self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)
    #
    # def test_two_columns_independent_vs_merged(self):
    #     grid = MapfGrid(['..',
    #                      '..',
    #                      '..'])
    #     agents_starts = ((0, 0), (0, 1))
    #     agents_goals = ((2, 0), (2, 1))
    #
    #     env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.01, -1, 1, -0.1)
    #
    #     vi_plan_func = partial(value_iteration, 1.0)
    #     independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], vi_plan_func, {})
    #     merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], vi_plan_func, {})
    #
    #     interesting_state = env.locations_to_state(((0, 0), (0, 1)))
    #
    #     expected_possible_actions = [vector_action_to_integer((LEFT, RIGHT))]
    #
    #     # Assert independent_joint_policy just choose the most efficient action
    #     self.assertEqual(independent_joiont_policy.act(interesting_state), vector_action_to_integer((DOWN, DOWN)))
    #
    #     # Assert merged_joint_policy avoids collision
    #     self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)
    #
    # def test_narrow_empty_grid(self):
    #     grid = MapfGrid(['....'])
    #
    #     agents_starts = ((0, 1), (0, 2))
    #     agents_goals = ((0, 0), (0, 3))
    #
    #     env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -1, 1, -0.01)
    #
    #     vi_plan_func = partial(value_iteration, 1.0)
    #     joint_policy = id(vi_plan_func, None, env, {})
    #
    #     self.assertEqual(joint_policy.act(env.s), vector_action_to_integer((LEFT, RIGHT)))

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
