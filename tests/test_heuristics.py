import unittest

from gym_mapf.envs.utils import MapfGrid, create_mapf_env, get_local_view
from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria
from solvers.vi import value_iteration, prioritized_value_iteration
from solvers.rtdp import (dijkstra_min_heuristic,
                          dijkstra_sum_heuristic,
                          local_views_prioritized_value_iteration_sum_heuristic,
                          local_views_prioritized_value_iteration_min_heuristic)


class HeuristicsTest(unittest.TestCase):
    def test_dijkstra_simple_env(self):
        """Test dijkstra algorithm on an environment which I can eye-ball testing"""
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
        ])

        agents_starts = ((0, 0),)
        agents_goals = ((0, 4),)

        env = MapfEnv(grid, 1, agents_starts, agents_goals, 0, -1000, 0, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for s in range(env.nS):
            self.assertEqual(dijkstra_func(s), vi_policy.v[s])

    def test_dijkstra_room_env(self):
        """Test dijkstra algorithm on a large, complex environment."""
        env = create_mapf_env('room-32-32-4', 1, 1, 0, -1000, -1, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for i in range(env.nS):
            self.assertEqual(dijkstra_func(i), vi_policy.v[i])

    def test_dijkstra_large_goal_reward(self):
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
        ])

        agents_starts = ((0, 0),)
        agents_goals = ((0, 4),)

        env = MapfEnv(grid, 1, agents_starts, agents_goals, 0, -1000, 100, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for i in range(env.nS):
            self.assertEqual(dijkstra_func(i), vi_policy.v[i])

    def test_dijkstra_sum_sanity_room_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0, -1000, 100, -1, OptimizationCriteria.Makespan)
        env0 = get_local_view(env, [0])
        env1 = get_local_view(env, [1])

        dijkstra_func = dijkstra_sum_heuristic(env)
        vi_policy0 = prioritized_value_iteration(1.0, env0, {})
        vi_policy1 = prioritized_value_iteration(1.0, env1, {})

        for s in range(env.nS):
            expected_reward = 0
            someone_not_in_goal = False
            s_locations = env.state_to_locations(s)

            s0 = env0.locations_to_state((s_locations[0],))
            s1 = env1.locations_to_state((s_locations[1],))

            if s_locations[0] != env.agents_goals[0]:
                someone_not_in_goal = True
                expected_reward += vi_policy0.v[s0] - env.reward_of_goal

            if s_locations[1] != env.agents_goals[1]:
                someone_not_in_goal = True
                expected_reward += vi_policy1.v[s1] - env.reward_of_goal

            if someone_not_in_goal:
                expected_reward += env.reward_of_goal

            self.assertEqual(dijkstra_func(s), expected_reward)

    def test_pvi_sum_sanity_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0.2, -1000, 100, -1, OptimizationCriteria.Makespan)
        env0 = get_local_view(env, [0])
        env1 = get_local_view(env, [1])

        heuristic_func = local_views_prioritized_value_iteration_sum_heuristic(1.0, env)
        vi_policy0 = prioritized_value_iteration(1.0, env0, {})
        vi_policy1 = prioritized_value_iteration(1.0, env1, {})

        for s in range(env.nS):
            expected_reward = 0
            someone_not_in_goal = False
            s_locations = env.state_to_locations(s)

            s0 = env0.locations_to_state((s_locations[0],))
            s1 = env1.locations_to_state((s_locations[1],))

            if s_locations[0] != env.agents_goals[0]:
                someone_not_in_goal = True
                expected_reward += vi_policy0.v[s0] - env.reward_of_goal

            if s_locations[1] != env.agents_goals[1]:
                someone_not_in_goal = True
                expected_reward += vi_policy1.v[s1] - env.reward_of_goal

            if someone_not_in_goal:
                expected_reward += env.reward_of_goal

            self.assertEqual(round(heuristic_func(s), 8), round(heuristic_func(s), 8))

    def test_pvi_min_sanity_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0.2, -1000, 100, -1, OptimizationCriteria.Makespan)
        envs = [get_local_view(env, [i]) for i in range(env.n_agents)]

        heuristic_func = local_views_prioritized_value_iteration_min_heuristic(1.0, env)
        vi_policy = [prioritized_value_iteration(1.0, envs[i], {}) for i in range(env.n_agents)]

        for s in range(env.nS):
            locations = env.state_to_locations(s)
            local_states = [envs[i].locations_to_state((locations[i],)) for i in range(env.n_agents)]
            relevant_values = [vi_policy[i].v[local_states[i]]
                               for i in range(env.n_agents) if locations[i] != env.agents_goals[i]]
            if not relevant_values:
                expected_value = 0
            else:
                expected_value = min(relevant_values)
            self.assertEqual(heuristic_func(s), expected_value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
