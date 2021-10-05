import unittest
from functools import partial

from solvers.utils import (CrossedPolicy,
                           detect_conflict_old,
                           solve_independently_and_cross,
                           Policy,
                           couple_detect_conflict,
                           get_shared_states)
from solvers.vi import value_iteration
from gym_mapf.envs.utils import MapfGrid, get_local_view, create_mapf_env
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    integer_action_to_vector,
                                    DOWN, RIGHT, LEFT, STAY,
                                    ACTIONS,
                                    OptimizationCriteria)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          fixed_iterations_count_rtdp)


class DictPolicy(Policy):
    def __init__(self, env, gamma, dict_policy):
        super().__init__(env, 1.0)
        self.dict_policy = dict_policy

    def _act_in_unfamiliar_state(self, s: int):
        return self.act(s)

    def act(self, s):
        return self.dict_policy[s]


class SolversUtilsTests(unittest.TestCase):
    def test_detect_conflict_finds_classical_conflict(self):
        grid = MapfGrid(['...',
                         '@.@',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        policy1 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(LEFT),
            5: ACTIONS.index(RIGHT),
            6: ACTIONS.index(LEFT),
        }

        policy2 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(RIGHT),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(RIGHT),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy2)],
                                     [[0], [1]])

        aux_local_env = get_local_view(env, [0])

        self.assertEqual(detect_conflict_old(env, joint_policy, {}),
                         (0,
                          1,
                          ((
                               0,
                               aux_local_env.locations_to_state(((0, 0),)),
                               aux_local_env.locations_to_state(((0, 1),))
                           ),
                           (
                               1,
                               aux_local_env.locations_to_state(((0, 2),)),
                               aux_local_env.locations_to_state(((0, 1),))
                           ))
                          ))

    def test_detect_conflict_return_none_when_no_conflict(self):
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        policy1 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        policy2 = {
            0: ACTIONS.index(DOWN),
            1: ACTIONS.index(DOWN),
            2: ACTIONS.index(DOWN),
            3: ACTIONS.index(DOWN),
            4: ACTIONS.index(DOWN),
            5: ACTIONS.index(DOWN),
            6: ACTIONS.index(DOWN),
            7: ACTIONS.index(DOWN),
            8: ACTIONS.index(DOWN),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy2)],
                                     [[0], [1]])

        self.assertEqual(detect_conflict_old(env, joint_policy, {}), None)

    def test_roni_scenario_with_id(self):
        # TODO: this test only pass when the first action in the ACTIONS array is STAY,
        #  fix it to work without the cheating
        grid = MapfGrid(['.@.',
                         '.@.',
                         '...'])
        agents_starts = ((0, 0), (0, 2))
        agents_goals = ((2, 0), (2, 2))

        env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, -1, 1, -0.1, OptimizationCriteria.Makespan)

        independent_joiont_policy = solve_independently_and_cross(env, [[0], [1]], partial(value_iteration, 1.0), {})

        interesting_state = env.locations_to_state(((0, 0), (0, 2)))

        # Assert independent_joint_policy just choose the most efficient action
        self.assertEqual(independent_joiont_policy.act(interesting_state), vector_action_to_integer((DOWN, DOWN)))

        # Assert no conflict
        self.assertEqual(detect_conflict_old(env, independent_joiont_policy, {}), None)

    def test_conflict_detected_for_room_scenario_with_crossed_policy(self):
        env = create_mapf_env('room-32-32-4', 1, 2, 0.2, -1000, 0, -1, OptimizationCriteria.Makespan)

        policy1 = fixed_iterations_count_rtdp(partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                              100,
                                              get_local_view(env, [0]), {})
        policy2 = fixed_iterations_count_rtdp(partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                              100,
                                              get_local_view(env, [1]), {})
        crossed_policy = CrossedPolicy(env, [policy1, policy2], [[0], [1]])

        self.assertIsNot(detect_conflict_old(env, crossed_policy, {}), None)

    def test_policy_crossing_for_continuous_agent_range(self):
        """
        * Solve independently for agent groups [[0, 1]]
        * Cross the policies
        * Make sure the crossed policy behaves right
        """
        env = create_mapf_env('room-32-32-4', 15, 3, 0, -1000, 0, -1, OptimizationCriteria.Makespan)
        interesting_locations = ((19, 22), (18, 24), (17, 22))

        plan_func = partial(fixed_iterations_count_rtdp,
                            partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                            100)

        crossed_policy = solve_independently_and_cross(env, [[0, 1], [2]], plan_func, {})

        policy0 = plan_func(get_local_view(env, [0, 1]), {})
        policy1 = plan_func(get_local_view(env, [2]), {})

        action0 = policy0.act(policy0.env.locations_to_state(interesting_locations[0:2]))
        action1 = policy1.act(policy1.env.locations_to_state((interesting_locations[2],)))

        vector_action_local = integer_action_to_vector(action0, 2) + integer_action_to_vector(action1, 1)

        joint_action = crossed_policy.act(env.locations_to_state(interesting_locations))
        vector_action_joint = integer_action_to_vector(joint_action, 3)

        self.assertEqual(vector_action_local, vector_action_joint)

    def test_policy_crossing_for_non_continuous_agent_range(self):
        """
        * Solve independently for agent groups [[1], [0,2]]
        * Cross the policies
        * Make sure the crossed policy behaves right
        """
        env = create_mapf_env('room-32-32-4', 15, 3, 0, -1000, 0, -1, OptimizationCriteria.Makespan)
        interesting_locations = ((19, 22), (18, 24), (17, 22))

        plan_func = partial(fixed_iterations_count_rtdp,
                            partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                            100)
        crossed_policy = solve_independently_and_cross(env, [[1], [0, 2]], plan_func, {})

        policy0 = plan_func(get_local_view(env, [1]), {})
        policy1 = plan_func(get_local_view(env, [0, 2]), {})

        action0 = policy0.act(policy0.env.locations_to_state((interesting_locations[1],)))
        action1 = policy1.act(
            policy1.env.locations_to_state((interesting_locations[0],) + (interesting_locations[2],)))

        vector_action0 = integer_action_to_vector(action0, 1)
        vector_action1 = integer_action_to_vector(action1, 2)
        vector_action_local = (vector_action1[0], vector_action0[0], vector_action1[1])

        joint_action = crossed_policy.act(env.locations_to_state(interesting_locations))
        vector_action_joint = integer_action_to_vector(joint_action, 3)

        self.assertEqual(vector_action_local, vector_action_joint)

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
        conflict = detect_conflict_old(env, policy, {})
        # Assert a conflict detected
        self.assertIsNotNone(conflict)

        aux_local_env = get_local_view(env, [0])
        agent_1_state = aux_local_env.locations_to_state(((21, 20),))
        agent_0_state = aux_local_env.locations_to_state(((21, 19),))

        possible_conflicts = [
            (0, 1, ((1, agent_1_state, agent_0_state), (0, agent_0_state, agent_1_state))),
            (0, 1, ((0, agent_0_state, agent_1_state), (1, agent_1_state, agent_0_state)))
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

        agents_starts = ((0, 0), (2, 0), (2, 2))
        agents_goals = ((0, 2), (2, 2), (2, 0))
        env = MapfEnv(grid, 3, agents_starts, agents_goals, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)

        # >>S
        # SSS
        # SSS
        policy0 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(STAY),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(STAY),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(STAY),
        }

        # SSS
        # SSS
        # >>S
        policy1 = {
            0: ACTIONS.index(STAY),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(RIGHT),
            3: ACTIONS.index(STAY),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(RIGHT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(STAY),
        }

        # SSS
        # SSS
        # S<<
        policy2 = {
            0: ACTIONS.index(STAY),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(STAY),
            3: ACTIONS.index(STAY),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(LEFT),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(get_local_view(env, [0]), 1.0, policy0),
                                           DictPolicy(get_local_view(env, [1]), 1.0, policy1),
                                           DictPolicy(get_local_view(env, [2]), 1.0, policy2)],
                                     [[0], [1], [2]])

        aux_local_env = get_local_view(env, [0])

        # Assert a conflict is found for agents 1 and 2
        self.assertEqual(couple_detect_conflict(env, joint_policy, 1, 2, {}),
                         (
                             (
                                 1,
                                 aux_local_env.locations_to_state(((2, 0),)),
                                 aux_local_env.locations_to_state(((2, 1),))
                             ),
                             (
                                 2,
                                 aux_local_env.locations_to_state(((2, 2),)),
                                 aux_local_env.locations_to_state(((2, 1),))
                             )
                         ))

        # Assert no conflict is found for agents 0 and 1
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 1, {}))

        # Assert no conflict is found for agents 0 and 2
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 2, {}))

    def test_couple_detect_conflict_3_agents_multiple_agents_in_group(self):
        """This test may sometime be used to test detecting a conflict for only a couple of agents.

            The test will make sure that agent 0 got no conflicts with 1 and 2 while agents 1 and 2 do get a conflict.
            Now agent 1 will be a part of a group contains both agent 0 and 1 ([0,1]). This way agent 1 index in its
            group will be 1 and not 0. This case is catching a bug I had previously.
        """
        grid = MapfGrid(['...',
                         '...',
                         '...'])

        agents_starts = ((0, 0), (2, 0), (2, 2))
        agents_goals = ((0, 2), (2, 2), (2, 0))
        env = MapfEnv(grid, 3, agents_starts, agents_goals, 0, -1, 1, -0.01, OptimizationCriteria.Makespan)
        single_agent_env = MapfEnv(grid, 1, (agents_starts[0],), (agents_goals[0],), 0, -1, 1, -0.01,
                                   OptimizationCriteria.Makespan)
        env01 = get_local_view(env, [0, 1])

        # >>S
        # SSS
        # SSS
        policy0 = {
            0: ACTIONS.index(RIGHT),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(STAY),
            3: ACTIONS.index(RIGHT),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(STAY),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(STAY),
        }

        # SSS
        # SSS
        # >>S
        policy1 = {
            0: ACTIONS.index(STAY),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(RIGHT),
            3: ACTIONS.index(STAY),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(RIGHT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(STAY),
        }

        # policy01 is a cross between agent 0 and agent 1
        policy01 = {}
        for s0 in range(9):
            for s1 in range(9):
                joint_state = env01.locations_to_state(
                    (single_agent_env.state_to_locations(s0)[0], single_agent_env.state_to_locations(s1)[0])
                )
                policy01[joint_state] = vector_action_to_integer(
                    (integer_action_to_vector(policy0[s0], 1)[0], integer_action_to_vector(policy1[s1], 1)[0])
                )

        # SSS
        # SSS
        # S<<
        policy2 = {
            0: ACTIONS.index(STAY),
            1: ACTIONS.index(STAY),
            2: ACTIONS.index(STAY),
            3: ACTIONS.index(STAY),
            4: ACTIONS.index(STAY),
            5: ACTIONS.index(LEFT),
            6: ACTIONS.index(STAY),
            7: ACTIONS.index(STAY),
            8: ACTIONS.index(LEFT),
        }

        joint_policy = CrossedPolicy(env, [DictPolicy(env01, 1.0, policy01),
                                           DictPolicy(get_local_view(env, [2]), 1.0, policy2)],
                                     [[0, 1], [2]])

        aux_local_env = get_local_view(env, [0])

        # Assert a conflict is found for agents 1 and 2
        self.assertEqual(couple_detect_conflict(env, joint_policy, 2, 1, {}),
                         (
                             (
                                 2,
                                 aux_local_env.locations_to_state(((2, 2),)),
                                 aux_local_env.locations_to_state(((2, 1),))
                             ),
                             (
                                 1,
                                 aux_local_env.locations_to_state(((2, 0),)),
                                 aux_local_env.locations_to_state(((2, 1),))
                             )
                         ))

        # Assert no conflict is found for agents 0 and 1
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 1, {}))

        # Assert no conflict is found for agents 0 and 2
        self.assertIsNone(couple_detect_conflict(env, joint_policy, 0, 2, {}))

    def test_shared_states(self):
        env = create_mapf_env('room-32-32-4', 9, 2, 0, -1000, 0, -1, OptimizationCriteria.Makespan)

        low_level_plan_func = partial(fixed_iterations_count_rtdp,
                                      partial(local_views_prioritized_value_iteration_min_heuristic, 1.0), 1.0,
                                      100)

        policy = solve_independently_and_cross(env,
                                               [[0], [1]],
                                               low_level_plan_func,
                                               {})
        shared_states, n0, n1 = get_shared_states(env, policy, 0, 1, {})

        # The agents are switching between this two states and therefore these states are shared.
        s1 = env.loc_to_int[(21, 20)]
        s2 = env.loc_to_int[(21, 19)]

        # Assert the conflict parameters are right
        self.assertIn(s1, shared_states)
        self.assertIn(s2, shared_states)


if __name__ == '__main__':
    unittest.main(verbosity=2)
