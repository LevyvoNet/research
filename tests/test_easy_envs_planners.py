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
                          local_views_prioritized_value_iteration_sum_heuristic,
                          dijkstra_min_heuristic)


class EasyEnvironmentsPlannersTest(unittest.TestCase):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def print_white_box_data(self, policy: Policy, info: Dict):
        pass

    def test_corridor_switch_no_clash_possible(self):
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
        reward, clashed, _ = evaluate_policy(policy, 100, 100)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Assert the reward is reasonable
        self.assertGreater(reward, 50.0 * env.reward_of_living)

    def test_single_agent_empty_grid(self):
        grid = MapfGrid(['.' * 8] * 8)

        env = MapfEnv(grid, 1, ((7, 0),), ((0, 7),), 0.1, 0.1, -1000, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        reward, clashed, _ = evaluate_policy(policy, 100, 1000)

        # print white box data
        self.print_white_box_data(policy, info)

        self.assertFalse(clashed)

        # solvers get cost of about 17-18 so 30 should be a fair upper bound
        self.assertGreater(reward, -30)

    def test_symmetrical_bottleneck_deterministic(self):
        grid = MapfGrid(['..@...',
                         '..@...',
                         '......',
                         '..@...'
                         '..@...'])

        agents_starts = ((2, 0), (2, 5))
        agents_goals = ((2, 5), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 1, 20)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        optimal_reward = 6.0 * env.reward_of_living + env.reward_of_goal
        # Allow a difference of 1 for MA-RTDP sub optimality
        self.assertGreaterEqual(reward - optimal_reward, env.reward_of_living)

    def test_symmetrical_bottleneck_deterministic_goal_large_reward(self):
        grid = MapfGrid(['..@...',
                         '..@...',
                         '......',
                         '..@...'
                         '..@...'])

        agents_starts = ((2, 0), (2, 5))
        agents_goals = ((2, 5), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -0.001, 100, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 1, 20)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        optimal_reward = 6.0 * env.reward_of_living + env.reward_of_goal
        # Allow a difference of 1 for MA-RTDP sub optimality
        self.assertGreaterEqual(reward - optimal_reward, env.reward_of_living)

    def test_symmetrical_bottleneck_stochastic(self):
        grid = MapfGrid(['..@...',
                         '..@...',
                         '......',
                         '..@...'
                         '..@...'])

        agents_starts = ((2, 0), (2, 5))
        agents_goals = ((2, 5), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 100, 100)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        optimal_reward_expectation = -12.605924
        self.assertGreaterEqual(reward, 30 * env.reward_of_living)

    def test_bottleneck_deterministic(self):
        grid = MapfGrid(['..@..',
                         '..@..',
                         '.....',
                         '..@..'
                         '..@..'])

        agents_starts = ((2, 0), (2, 4))
        agents_goals = ((2, 4), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 1, 100)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        # Value iteration is done in 7 steps
        # Vanilla RTDP also do it in 7 steps
        # MA-RTDP takes 8 steps
        optimal_reward = 6.0 * env.reward_of_living + env.reward_of_goal

        # Allow a difference of 1 for MA-RTDP sub optimality
        self.assertGreaterEqual(reward - optimal_reward, env.reward_of_living)

    def test_bottleneck_deterministic_goal_large_reward(self):
        grid = MapfGrid(['..@..',
                         '..@..',
                         '.....',
                         '..@..'
                         '..@..'])

        agents_starts = ((2, 0), (2, 4))
        agents_goals = ((2, 4), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, 0, -0.001, 100, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 1, 100)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        # Value iteration is done in 7 steps
        # Vanilla RTDP also do it in 7 steps
        # MA-RTDP takes 8 steps
        optimal_reward = 6.0 * env.reward_of_living + env.reward_of_goal
        # Allow a difference of 1 for MA-RTDP sub optimality
        self.assertGreaterEqual(reward - optimal_reward, env.reward_of_living)

    def test_bottleneck_stochastic(self):
        grid = MapfGrid(['..@..',
                         '..@..',
                         '.....',
                         '..@..'
                         '..@..'])

        agents_starts = ((2, 0), (2, 4))
        agents_goals = ((2, 4), (2, 0))

        # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)

        info = {}
        plan_func = self.get_plan_func()
        policy = plan_func(env, info)

        # Check the policy performance
        reward, clashed, _ = evaluate_policy(policy, 100, 100)

        # print white box data
        self.print_white_box_data(policy, info)

        # Make sure no clash happened
        self.assertFalse(clashed)

        # Make sure the policy is near optimal
        # Value iteration is done in 7 steps
        # Vanilla RTDP also do it in 7 steps
        # MA-RTDP takes 8 steps
        optimal_reward_expectation = -12.420579
        self.assertGreaterEqual(reward, 30 * env.reward_of_living)


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
                       500)


class EasyEnvironmentsStopWhenNoImprovementLocalMinHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 500
        self.iters_in_batch = 50

        return partial(stop_when_no_improvement_between_batches_rtdp,
                       partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {info['n_iterations']}/{self.max_iterations} iterations")


class EasyEnvironmentsIdOverValueIterationPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        low_level_planner = partial(value_iteration, 1.0)
        return partial(id, low_level_planner)


class EasyEnvironmentsStopWhenNoImprovementDijkstraHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 500
        self.iters_in_batch = 50

        return partial(stop_when_no_improvement_between_batches_rtdp,
                       dijkstra_heuristic,
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {info['n_iterations']}/{self.max_iterations} iterations")


class EasyEnvironmentsMultiagentSumHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 1000
        self.iters_in_batch = 50

        return partial(ma_rtdp,
                       partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {info['n_iterations']}/{self.max_iterations} iterations")


class EasyEnvironmentsMultiagentDijkstraHeuristicRtdpPlannerTest(EasyEnvironmentsPlannersTest):
    def get_plan_func(self) -> Callable[[MapfEnv, Dict], Policy]:
        self.max_iterations = 1000
        self.iters_in_batch = 50

        return partial(ma_rtdp,
                       dijkstra_min_heuristic,
                       1.0,
                       self.iters_in_batch,
                       self.max_iterations)

    def print_white_box_data(self, policy: Policy, info: Dict):
        print(f"performed {info['n_iterations']}/{self.max_iterations} iterations")


if __name__ == '__main__':
    unittest.main(verbosity=2)
