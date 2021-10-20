import functools
import unittest
from functools import partial

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, RIGHT, LEFT, STAY,
                                    OptimizationCriteria)
from gym_mapf.envs.utils import MapfGrid, create_mapf_env
from solvers import (ValueIterationPolicy,
                     IdPolicy,
                     RtdpPolicy)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          fixed_iterations_rtdp_merge,
                          solution_heuristic_min)
from solvers.utils import solve_independently_and_cross


def rtdp_policy_creator(env, gamma):
    return RtdpPolicy(env, gamma, partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                      100, 100)


class IdTests(unittest.TestCase):
    def test_corridor_switch_independent_vs_merged(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.2, -1, 1, -0.01, OptimizationCriteria.Makespan)

        independent_joint_policy = solve_independently_and_cross(env, [[0], [1]], ValueIterationPolicy, 1.0, {})
        merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], ValueIterationPolicy, 1.0, {})

        interesting_state = env.locations_to_state(((1, 1), (0, 1)))

        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joint_policy.act(interesting_state), vector_action_to_integer((UP, LEFT)))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)

    def test_two_columns_independent_vs_merged(self):
        grid = MapfGrid(['..',
                         '..',
                         '..'])
        agents_starts = ((0, 0), (0, 1))
        agents_goals = ((2, 0), (2, 1))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, -1, 1, -0.1, OptimizationCriteria.Makespan)

        independent_joint_policy = solve_independently_and_cross(env, [[0], [1]], ValueIterationPolicy, 1.0, {})
        merged_joint_policy = solve_independently_and_cross(env, [[0, 1]], ValueIterationPolicy, 1.0, {})

        interesting_state = env.locations_to_state(((0, 0), (0, 1)))

        expected_possible_actions = [vector_action_to_integer((LEFT, RIGHT))]

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joint_policy.act(interesting_state), vector_action_to_integer((DOWN, DOWN)))

        # Assert merged_joint_policy avoids collision
        self.assertIn(merged_joint_policy.act(interesting_state), expected_possible_actions)

    def test_narrow_empty_grid(self):
        grid = MapfGrid(['....'])

        agents_starts = ((0, 1), (0, 2))
        agents_goals = ((0, 0), (0, 3))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, -1, 1, -0.01, OptimizationCriteria.Makespan)

        joint_policy = IdPolicy(env, 1.0, ValueIterationPolicy, None).train()

        self.assertEqual(joint_policy.act(env.s), vector_action_to_integer((LEFT, RIGHT)))

    def test_env_with_switch_conflict_solved_properly(self):
        env = create_mapf_env('room-32-32-4', 9, 2, 0, -1000, 0, -1, OptimizationCriteria.Makespan)

        rtdp_merge_func = functools.partial(fixed_iterations_rtdp_merge, solution_heuristic_min, 100)
        policy = IdPolicy(env, 1.0, rtdp_policy_creator, rtdp_merge_func).train()

        info = policy.evaluate(1, 1000)

        self.assertFalse(info['clashed'])

        # Assert that the solution is reasonable (actually solving)
        self.assertGreater(info['success_rate'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
