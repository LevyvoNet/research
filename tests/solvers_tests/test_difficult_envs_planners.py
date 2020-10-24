import unittest
from typing import Dict, Callable
from functools import partial

from gym_mapf.tests.utils import measure_time
from gym_mapf.envs.utils import create_mapf_env, MapfEnv, MapfGrid
from research.solvers.utils import evaluate_policy, Policy
from research.solvers import fixed_iterations_count_rtdp, stop_when_no_improvement_between_batches_rtdp, ma_rtdp
from research.solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                                   deterministic_relaxation_prioritized_value_iteration_heuristic,
                                   local_views_prioritized_value_iteration_sum_heuristic)


class DifficultEnvsPlannerTest(unittest.TestCase):
    """
    This test case is for stochastic environments with multiple agents and complicated maps (room for exapmle).

    so far, only RTDP based solvers succeed to solve such environments.
    """

    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def print_white_box_data(self, policy: Policy, info: Dict):
        pass

    @measure_time
    def test_room_scen_13_converges(self):
        """This is a pretty hard scenario (maybe because of the potential conflict).

        Note how the 'smart' RTDP needs only 300-400 iterations and stops afterwards.
        The fixed iterations RTDP however needs to know in advance... (used to be 1000)
        """
        env = create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, -1, -1)

        plan_func = self.get_plan_func()
        info = {}
        policy = plan_func(env, info)

        reward, clashed = evaluate_policy(policy, 1, 1000)

        self.print_white_box_data(policy, info)

        # Assert that the solution is reasonable (actually solving)
        optimal_reward = -48.0
        self.assertGreaterEqual(reward, optimal_reward * 1.05)

    @measure_time
    def test_normal_room_scenario_converges(self):
        env = create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, -1, -1)

        plan_func = self.get_plan_func()
        info = {}
        policy = plan_func(env, info)

        reward, clashed = evaluate_policy(policy, 1, 1000)

        self.print_white_box_data(policy, info)

        # Assert that the solution is reasonable (actually solving)
        self.assertEqual(reward, -9.0)

    @measure_time
    def test_deterministic_room_scenario_1_2_agents(self):
        env = create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        self.print_white_box_data(policy, info)

        reward, _ = evaluate_policy(policy, 1, 1000)
        self.assertGreaterEqual(reward, -43 * 1.05)

    @measure_time
    def test_hand_crafted_env_converges(self):
        grid = MapfGrid([
            '...',
            '@.@',
            '@.@',
            '...'])

        agent_starts = ((0, 0), (0, 2))
        agents_goals = ((3, 0), (3, 2))

        deterministic_env = MapfEnv(grid, 2, agent_starts, agents_goals,
                                    0.0, 0.0, -1000, -1, -1)

        planner = self.get_plan_func()
        policy = planner(deterministic_env, {})
        reward, clashed = evaluate_policy(policy, 1, 20)

        # Make sure this policy is optimal
        self.assertEqual(reward, -6.0)

    @measure_time
    def test_stochastic_room_env(self):
        """Easy room scenario with fail probabilities"""
        env = create_mapf_env('room-32-32-4', 12, 2, 0.1, 0.1, -1000, -1, -1)

        plan_func = self.get_plan_func()
        info = {}
        policy = plan_func(env, info)

        reward, clashed = evaluate_policy(policy, 1, 1000)

        self.print_white_box_data(policy, info)

        # Assert that the solution is reasonable (actually solving)
        self.assertGreaterEqual(reward, -20 * 1.05)


class FixedIterationsCountRtdpPlannerTest(DifficultEnvsPlannerTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(fixed_iterations_count_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                       400)


class StopWhenNoImprovementRtdpMinLocalHeuristicPlannerTest(DifficultEnvsPlannerTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.iter_in_batches = 100
        self.max_iterations = 500

        return partial(stop_when_no_improvement_between_batches_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                       1.0,
                       self.iter_in_batches,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


class StopWhenNoImprovementRtdpSumLocalHeuristicPlannerTest(DifficultEnvsPlannerTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.iter_in_batches = 100
        self.max_iterations = 500

        return partial(stop_when_no_improvement_between_batches_rtdp,
                       partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
                       1.0,
                       self.iter_in_batches,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


class MultiagentSumHeuristicRtdpPlannerTest(DifficultEnvsPlannerTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.iters_in_batch = 100
        self.max_iterations = 500

        return partial(ma_rtdp,
                       partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


if __name__ == '__main__':
    unittest.main(verbosity=2)
