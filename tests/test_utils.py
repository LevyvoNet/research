import unittest
import json
from functools import partial

from solvers.utils import (CrossedPolicy,
                           detect_conflict,
                           solve_independently_and_cross,
                           Policy,
                           couple_detect_conflict)
from solvers.vi import value_iteration
from gym_mapf.envs.utils import MapfGrid, get_local_view, create_mapf_env
from gym_mapf.envs.grid import SingleAgentState, SingleAgentAction, SingleAgentStateSpace
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    MultiAgentState,
                                    MultiAgentAction,
                                    OptimizationCriteria)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          fixed_iterations_count_rtdp)


class DictPolicy(Policy):
    def __init__(self, env, gamma, dict_policy):
        super().__init__(env, 1.0)
        self.dict_policy = dict_policy

    def act(self, s):
        return self.dict_policy[s]


class SolversUtilsTests(unittest.TestCase):
    def test_detect_conflict_finds_classical_conflict(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 2)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        policy1 = {
            MultiAgentState({0: SingleAgentState(0, 0)}, grid): MultiAgentAction({0: SingleAgentAction.RIGHT}),
            MultiAgentState({0: SingleAgentState(2, 0)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(0, 1)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(1, 1)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(2, 1)}, grid): MultiAgentAction({0: SingleAgentAction.LEFT}),
            MultiAgentState({0: SingleAgentState(0, 2)}, grid): MultiAgentAction({0: SingleAgentAction.RIGHT}),
            MultiAgentState({0: SingleAgentState(2, 2)}, grid): MultiAgentAction({0: SingleAgentAction.LEFT}),
        }

        policy2 = {
            MultiAgentState({1: SingleAgentState(0, 0)}, grid): MultiAgentAction({1: SingleAgentAction.RIGHT}),
            MultiAgentState({1: SingleAgentState(2, 0)}, grid): MultiAgentAction({1: SingleAgentAction.RIGHT}),
            MultiAgentState({1: SingleAgentState(0, 1)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(1, 1)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(2, 1)}, grid): MultiAgentAction({1: SingleAgentAction.RIGHT}),
            MultiAgentState({1: SingleAgentState(0, 2)}, grid): MultiAgentAction({1: SingleAgentAction.LEFT}),
            MultiAgentState({1: SingleAgentState(2, 2)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy2)],
                                     [[0], [1]])

        self.assertEqual(detect_conflict(env, joint_policy),
                         (
                             (0, SingleAgentState(0, 0), SingleAgentState(0, 1)),
                             (1, SingleAgentState(0, 2), SingleAgentState(0, 1))
                         )
                         )

    def test_detect_conflict_return_none_when_no_conflict(self):
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 2)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        policy1 = {
            MultiAgentState({0: SingleAgentState(0, 0)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(1, 0)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(2, 0)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(0, 1)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(1, 1)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(2, 1)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(0, 2)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(1, 2)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
            MultiAgentState({0: SingleAgentState(2, 2)}, grid): MultiAgentAction({0: SingleAgentAction.DOWN}),
        }

        policy2 = {
            MultiAgentState({1: SingleAgentState(0, 0)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(1, 0)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(2, 0)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(0, 1)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(1, 1)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(2, 1)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(0, 2)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(1, 2)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
            MultiAgentState({1: SingleAgentState(2, 2)}, grid): MultiAgentAction({1: SingleAgentAction.DOWN}),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy2)],
                                     [[0], [1]])

        self.assertEqual(detect_conflict(env, joint_policy), None)

    def test_roni_scenario_with_id(self):
        # TODO: this test only pass when the first action in the ACTIONS array is STAY,
        #  fix it to work without the cheating
        grid = MapfGrid(['.@.',
                         '.@.',
                         '...'])

        start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 2)}, grid)

        env = MapfEnv(grid, 2, start_state, goal_state, 0.1, -1, 1, -0.1, OptimizationCriteria.Makespan)

        independent_joint_policy = solve_independently_and_cross(env, [[0], [1]], partial(value_iteration, 1.0), {})

        interesting_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
        expected_action = MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.DOWN})

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joint_policy.act(interesting_state), expected_action)

        # Assert no conflict
        self.assertEqual(detect_conflict(env, independent_joint_policy), None)

    def test_conflict_detected_for_room_scenario_with_crossed_policy(self):
        env = create_mapf_env('room-32-32-4', 1, 2, 0.1, -1000, 0, -1, OptimizationCriteria.Makespan)

        policy1 = fixed_iterations_count_rtdp(partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                              100,
                                              get_local_view(env, [1]), {})

        policy0 = fixed_iterations_count_rtdp(partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                              100,
                                              get_local_view(env, [0]), {})

        crossed_policy = CrossedPolicy(env, [policy0, policy1], [[0], [1]])

        self.assertIsNot(detect_conflict(env, crossed_policy), None)

    def test_policy_crossing_for_continuous_agent_range(self):
        """
        * Solve independently for agent groups [[0, 1]]
        * Cross the policies
        * Make sure the crossed policy behaves right
        """
        env = create_mapf_env('room-32-32-4', 15, 3, 0, -1000, 0, -1, OptimizationCriteria.Makespan)
        interesting_state0 = SingleAgentState(19, 22)
        interesting_state1 = SingleAgentState(18, 24)
        interesting_state2 = SingleAgentState(17, 22)
        interesting_state = MultiAgentState({0: interesting_state0,
                                             1: interesting_state1,
                                             2: interesting_state2}, env.grid)

        plan_func = partial(fixed_iterations_count_rtdp,
                            partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                            100)

        crossed_policy = solve_independently_and_cross(env, [[0, 1], [2]], plan_func, {})

        policy01 = plan_func(get_local_view(env, [0, 1]), {})
        policy2 = plan_func(get_local_view(env, [2]), {})

        expected_action01 = policy01.act(MultiAgentState({0: interesting_state0, 1: interesting_state1},
                                                         policy01.env.grid))
        expected_action2 = policy2.act(MultiAgentState({2: interesting_state2}, policy2.env.grid))

        expected_joint_action = MultiAgentAction({
            0: expected_action01[0],
            1: expected_action01[1],
            2: expected_action2[2],
        })

        joint_action = crossed_policy.act(interesting_state)

        self.assertEqual(expected_joint_action, joint_action)

    def test_policy_crossing_for_non_continuous_agent_range(self):
        """
        * Solve independently for agent groups [[1], [0,2]]
        * Cross the policies
        * Make sure the crossed policy behaves right
        """
        env = create_mapf_env('room-32-32-4', 15, 3, 0, -1000, 0, -1, OptimizationCriteria.Makespan)
        interesting_state0 = SingleAgentState(19, 22)
        interesting_state1 = SingleAgentState(18, 24)
        interesting_state2 = SingleAgentState(17, 22)
        interesting_state = MultiAgentState({0: interesting_state0,
                                             1: interesting_state1,
                                             2: interesting_state2}, env.grid)

        plan_func = partial(fixed_iterations_count_rtdp,
                            partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                            100)
        crossed_policy = solve_independently_and_cross(env, [[1], [0, 2]], plan_func, {})

        policy1 = plan_func(get_local_view(env, [1]), {})
        policy02 = plan_func(get_local_view(env, [0, 2]), {})

        action1 = policy1.act(MultiAgentState({1: interesting_state1}, env.grid))
        action02 = policy02.act(MultiAgentState({0: interesting_state0, 2: interesting_state2}, env.grid))

        expected_joint_action = MultiAgentAction({
            0: action02[0],
            1: action1[1],
            2: action02[2]
        })

        joint_action = crossed_policy.act(interesting_state)

        self.assertEqual(expected_joint_action, joint_action)

    def test_detect_conflict_detects_switching(self):
        """
        * Create an env which its independent optimal policies cause a SWITCHING conflict
        * Solve independently
        * Make sure the conflict is detected
        """
        env = create_mapf_env('room-32-32-4', 9, 2, 0, -1000, 0, -1, OptimizationCriteria.Makespan)

        low_level_plan_func = partial(fixed_iterations_count_rtdp,
                                      partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                      100)

        policy = solve_independently_and_cross(env,
                                               [[0], [1]],
                                               low_level_plan_func,
                                               {})
        conflict = detect_conflict(env, policy)
        # Assert a conflict detected
        self.assertIsNotNone(conflict)

        agent_0_state = SingleAgentState(21, 19)
        agent_1_state = SingleAgentState(21, 20)

        possible_conflicts = [
            ((1, agent_1_state, agent_0_state), (0, agent_0_state, agent_1_state)),
            ((0, agent_0_state, agent_1_state), (1, agent_1_state, agent_0_state))
        ]

        # Assert the conflict parameters are right
        self.assertIn(conflict, possible_conflicts)

    def test_couple_detect_conflict_3_agents(self):
        """This test may sometime be used to test detecting a conflict for only a couple of agents.

        The test will make sure that agent 0 got no conflicts with 1 and 2 while agents 1 and 2 do get a conflict.
        """
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(2, 0),
            2: SingleAgentState(2, 2),

        }, grid)

        goal_state = MultiAgentState({
            0: SingleAgentState(0, 2),
            1: SingleAgentState(2, 2),
            2: SingleAgentState(2, 0),

        }, grid)

        env = MapfEnv(grid, 3, start_state, goal_state, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        # >>S
        # SSS
        # SSS
        policy0 = {
            MultiAgentState({0: SingleAgentState(0, 0)}, grid): MultiAgentAction({0: SingleAgentAction.RIGHT}),
            MultiAgentState({0: SingleAgentState(1, 0)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(2, 0)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(0, 1)}, grid): MultiAgentAction({0: SingleAgentAction.RIGHT}),
            MultiAgentState({0: SingleAgentState(1, 1)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(2, 1)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(0, 2)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(1, 2)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
            MultiAgentState({0: SingleAgentState(2, 2)}, grid): MultiAgentAction({0: SingleAgentAction.STAY}),
        }

        # SSS
        # SSS
        # >>S
        policy1 = {
            MultiAgentState({1: SingleAgentState(0, 0)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(1, 0)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(2, 0)}, grid): MultiAgentAction({1: SingleAgentAction.RIGHT}),
            MultiAgentState({1: SingleAgentState(0, 1)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(1, 1)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(2, 1)}, grid): MultiAgentAction({1: SingleAgentAction.RIGHT}),
            MultiAgentState({1: SingleAgentState(0, 2)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(1, 2)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
            MultiAgentState({1: SingleAgentState(2, 2)}, grid): MultiAgentAction({1: SingleAgentAction.STAY}),
        }

        # SSS
        # SSS
        # S<<
        policy2 = {
            MultiAgentState({2: SingleAgentState(0, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(0, 1)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 1)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 1)}, grid): MultiAgentAction({2: SingleAgentAction.LEFT}),
            MultiAgentState({2: SingleAgentState(0, 2)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 2)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 2)}, grid): MultiAgentAction({2: SingleAgentAction.LEFT}),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy0),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [2]), 1.0, policy2)],
                                     [[0], [1], [2]])

        aux_local_env = get_local_view(env, [0])

        # Assert a conflict is found for agents 1 and 2
        self.assertEqual(couple_detect_conflict(env, joint_policy, 1, 2),
                         (
                             (
                                 1,
                                 SingleAgentState(2, 0),
                                 SingleAgentState(2, 1)
                             ),
                             (
                                 2,
                                 SingleAgentState(2, 2),
                                 SingleAgentState(2, 1)
                             )
                         ))

        # Assert no conflict is found for agents 0 and 1
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 1))

        # Assert no conflict is found for agents 0 and 2
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 2))

    def test_couple_detect_conflict_3_agents_multiple_agents_in_group(self):
        """This test may sometime be used to test detecting a conflict for only a couple of agents.

            The test will make sure that agent 0 got no conflicts with 1 and 2 while agents 1 and 2 do get a conflict.
            Now agent 1 will be a part of a group contains both agent 0 and 1 ([0,1]). This way agent 1 index in its
            group will be 1 and not 0. This case is catching a bug I had previously.
        """
        grid = MapfGrid(['...',
                         '...',
                         '...'])
        start_state = MultiAgentState({
            0: SingleAgentState(0, 0),
            1: SingleAgentState(2, 0),
            2: SingleAgentState(2, 2),

        }, grid)

        goal_state = MultiAgentState({
            0: SingleAgentState(0, 2),
            1: SingleAgentState(2, 2),
            2: SingleAgentState(2, 0),

        }, grid)

        env = MapfEnv(grid, 3, start_state, goal_state, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        # >>S
        # SSS
        # SSS
        policy0 = {
            SingleAgentState(0, 0): SingleAgentAction.RIGHT,
            SingleAgentState(1, 0): SingleAgentAction.STAY,
            SingleAgentState(2, 0): SingleAgentAction.STAY,
            SingleAgentState(0, 1): SingleAgentAction.RIGHT,
            SingleAgentState(1, 1): SingleAgentAction.STAY,
            SingleAgentState(2, 1): SingleAgentAction.STAY,
            SingleAgentState(0, 2): SingleAgentAction.STAY,
            SingleAgentState(1, 2): SingleAgentAction.STAY,
            SingleAgentState(2, 2): SingleAgentAction.STAY,
        }

        # SSS
        # SSS
        # >>S
        policy1 = {
            SingleAgentState(0, 0): SingleAgentAction.STAY,
            SingleAgentState(1, 0): SingleAgentAction.STAY,
            SingleAgentState(2, 0): SingleAgentAction.RIGHT,
            SingleAgentState(0, 1): SingleAgentAction.STAY,
            SingleAgentState(1, 1): SingleAgentAction.STAY,
            SingleAgentState(2, 1): SingleAgentAction.RIGHT,
            SingleAgentState(0, 2): SingleAgentAction.STAY,
            SingleAgentState(1, 2): SingleAgentAction.STAY,
            SingleAgentState(2, 2): SingleAgentAction.STAY,
        }

        # policy01 is a cross between agent 0 and agent 1
        policy01 = {}
        for s0 in SingleAgentStateSpace(grid):
            for s1 in SingleAgentStateSpace(grid):
                joint_state = MultiAgentState({0: s0, 1: s1}, grid)
                policy01[joint_state] = MultiAgentAction({0: policy0[s0], 1: policy1[s1]})

        # SSS
        # SSS
        # S<<
        policy2 = {
            MultiAgentState({2: SingleAgentState(0, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 0)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(0, 1)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 1)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 1)}, grid): MultiAgentAction({2: SingleAgentAction.LEFT}),
            MultiAgentState({2: SingleAgentState(0, 2)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(1, 2)}, grid): MultiAgentAction({2: SingleAgentAction.STAY}),
            MultiAgentState({2: SingleAgentState(2, 2)}, grid): MultiAgentAction({2: SingleAgentAction.LEFT}),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0, 1]), 1.0, policy01),
                                           DictPolicy(get_local_view(env, [2]), 1.0, policy2)],
                                     [[0, 1], [2]])

        # Assert a conflict is found for agents 1 and 2
        self.assertEqual(couple_detect_conflict(env, joint_policy, 2, 1),
                         (
                             (
                                 2,
                                 SingleAgentState(2, 2),
                                 SingleAgentState(2, 1)
                             ),
                             (
                                 1,
                                 SingleAgentState(2, 0),
                                 SingleAgentState(2, 1)
                             )
                         ))

        # Assert no conflict is found for agents 0 and 1
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 1))

        # Assert no conflict is found for agents 0 and 2
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
