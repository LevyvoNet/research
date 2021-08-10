import unittest

from gym_mapf.envs.utils import MapfGrid, create_mapf_env, get_local_view
from gym_mapf.envs.mapf_env import MapfEnv, MultiAgentState, OptimizationCriteria
from gym_mapf.envs.grid import SingleAgentState
from solvers.vi import value_iteration, prioritized_value_iteration
from solvers.rtdp import (dijkstra_min_heuristic,
                          dijkstra_sum_heuristic,
                          local_views_prioritized_value_iteration_min_heuristic,
                          local_views_prioritized_value_iteration_sum_heuristic)


class HeuristicsTest(unittest.TestCase):
    def test_dijkstra_simple_env(self):
        """Test dijkstra algorithm on an environment which I can eye-ball testing"""
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
        ])

        start_state = MultiAgentState({0: SingleAgentState(0, 0)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 4)}, grid)

        env = MapfEnv(grid, 1, start_state, goal_state, 0, -1000, 0, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for s in env.observation_space:
            self.assertEqual(dijkstra_func(s), vi_policy.v[s])

    def test_dijkstra_room_env(self):
        """Test dijkstra algorithm on a large, complex environment."""
        env = create_mapf_env('room-32-32-4', 1, 1, 0, -1000, -1, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)

        vi_policy = value_iteration(1.0, env, {})

        for s in env.observation_space:
            self.assertEqual(dijkstra_func(s), vi_policy.v[s])

    def test_dijkstra_large_goal_reward(self):
        grid = MapfGrid([
            '..@..',
            '..@..',
            '.....',
        ])

        start_state = MultiAgentState({0: SingleAgentState(0, 0)}, grid)
        goal_state = MultiAgentState({0: SingleAgentState(0, 4)}, grid)

        env = MapfEnv(grid, 1, start_state, goal_state, 0, -1000, 100, -1, OptimizationCriteria.Makespan)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for s in env.observation_space:
            self.assertEqual(dijkstra_func(s), vi_policy.v[s])

    def test_dijkstra_sum_sanity_room_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0, -1000, 100, -1, OptimizationCriteria.Makespan)
        env0 = get_local_view(env, [0])
        env1 = get_local_view(env, [1])

        dijkstra_func = dijkstra_sum_heuristic(env)
        vi_policy0 = prioritized_value_iteration(1.0, env0, {})
        vi_policy1 = prioritized_value_iteration(1.0, env1, {})

        for s in env.observation_space:
            self.assertEqual(dijkstra_func(s),
                             vi_policy0.v[MultiAgentState({0: s[0]}, env0.grid)] + vi_policy1.v[MultiAgentState({1: s[1]}), env1.grid])

    def test_pvi_sum_sanity_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0.2, -1000, 100, -1, OptimizationCriteria.Makespan)
        env0 = get_local_view(env, [0])
        env1 = get_local_view(env, [1])

        heuristic_func = local_views_prioritized_value_iteration_sum_heuristic(1.0, env)
        vi_policy0 = prioritized_value_iteration(1.0, env0, {})
        vi_policy1 = prioritized_value_iteration(1.0, env1, {})

        for s in env.observation_space:
            self.assertEqual(heuristic_func(s),
                             vi_policy0.v[MultiAgentState({0: s[0]}, env0.grid)] + vi_policy1.v[MultiAgentState({1: s[1]}, env1.grid)])

    def test_pvi_min_sanity_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0.2, -1000, 100, -1, OptimizationCriteria.Makespan)
        envs = [get_local_view(env, [i]) for i in env.agents]

        heuristic_func = local_views_prioritized_value_iteration_min_heuristic(1.0, env)
        vi_policy = [prioritized_value_iteration(1.0, envs[i], {}) for i in env.agents]

        for s in env.observation_space:
            relevant_values = [vi_policy[i].v[MultiAgentState({i: s[i]}, envs[i].grid)]
                               for i in env.agents if s[i] != env.goal_state[i]]
            if not relevant_values:
                expected_value = 0
            else:
                expected_value = min(relevant_values)
            self.assertEqual(heuristic_func(s), expected_value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
