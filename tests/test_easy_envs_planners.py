import unittest
from functools import partial
from typing import Dict, Callable

from gym_mapf.envs.utils import MapfGrid, create_mapf_env
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, STAY)
from solvers.utils import Policy, evaluate_policy
from solvers import (value_iteration,
                              policy_iteration,
                              rtdp_iterations_generator,
                              id,
                              lrtdp,
                              fixed_iterations_count_rtdp,
                              stop_when_no_improvement_between_batches_rtdp,
                              ma_rtdp)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                                   local_views_prioritized_value_iteration_sum_heuristic)


class EasyEnvironmentsPlannersTest(unittest.TestCase):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def print_white_box_data(self, policy: Policy, info: Dict):
        pass

    def test_corridor_switch(self):
        grid = MapfGrid(['...',
                         '@.@'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((0, 2), (0, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Assert no conflict is possible
        interesting_state = env.locations_to_state(((1, 1), (0, 1)))
        expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                     vector_action_to_integer((DOWN, UP))]

        self.assertIn(policy.act(interesting_state), expected_possible_actions)

        # Check the policy performance
        reward, _ = evaluate_policy(policy, 100, 100)

        # print white box data
        self.print_white_box_data(policy, info)


class EasyEnvironmentsValueIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(value_iteration, 1.0)


class EasyEnvironmentsPolicyIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(policy_iteration, 1.0)


class EasyEnvironmentsFixedIterationsCountRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        return partial(fixed_iterations_count_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                       1.0,
                       100)


class EasyEnvironmentsStopWhenNoImprovementLocalMinHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 100
        self.iters_in_batch = 10

        return partial(stop_when_no_improvement_between_batches_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")

# The sum heuristic is not admissible for Makespan, therefore not interesting right now.
# class EasyEnvironmentsStopWhenNoImprovementLocalSumHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
#     def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
#         self.max_iterations = 100
#         self.iters_in_batch = 10
#
#         return partial(stop_when_no_improvement_between_batches_rtdp,
#                        partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
#                        1.0,
#                        10,
#                        100)
#
#     def print_white_box_data(self, policy: Policy, info: Dict):
#         print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


class EasyEnvironmentsIdOverValueIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        low_level_planner = partial(value_iteration, 1.0)
        return partial(id, low_level_planner)

# The sum heuristic is not admissible for Makespan, therefore not interesting right now.
# class EasyEnvironmentsMultiagentSumHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
#     def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
#         self.max_iterations = 100
#         self.iters_in_batch = 10
#
#         return partial(ma_rtdp,
#                        partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
#                        1.0,
#                        10,
#                        100)
#
#     def print_white_box_data(self, policy: Policy, info: Dict):
#         print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


class EasyEnvironmentsMultiagentMinHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 100
        self.iters_in_batch = 10

        return partial(ma_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                       1.0,
                       10,
                       100)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {len(info['iterations'])}/{self.max_iterations} iterations")


if __name__ == '__main__':
    unittest.main(verbosity=2)
