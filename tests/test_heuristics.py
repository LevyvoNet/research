import unittest

from gym_mapf.envs.utils import MapfGrid, create_mapf_env, get_local_view
from gym_mapf.envs.mapf_env import MapfEnv
from solvers.vi import value_iteration,prioritized_value_iteration
from solvers.rtdp import dijkstra_min_heuristic, dijkstra_sum_heuristic


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

        env = MapfEnv(grid, 1, agents_starts, agents_goals, 0, 0, -1000, -1, -1)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for i in range(env.nS):
            self.assertEqual(dijkstra_func(i), vi_policy.v[i])

    def test_dijkstra_room_env(self):
        """Test dijkstra algorithm on a large, complex environment."""
        env = create_mapf_env('room-32-32-4', 1, 1, 0, 0, -1000, -1, -1)

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

        env = MapfEnv(grid, 1, agents_starts, agents_goals, 0, 0, -1000, 100, -1)

        dijkstra_func = dijkstra_min_heuristic(env)
        vi_policy = value_iteration(1.0, env, {})

        for i in range(env.nS):
            self.assertEqual(dijkstra_func(i), vi_policy.v[i])

    def test_dijkstra_sum_sanity_room_env_large_goal_reward(self):
        env = create_mapf_env('sanity-2-8', None, 2, 0, 0, -1000, 100, -1)
        env0 = get_local_view(env, [0])
        env1 = get_local_view(env, [1])

        dijkstra_func = dijkstra_sum_heuristic(env)
        vi_policy0 = prioritized_value_iteration(1.0, env0, {})
        vi_policy1 = prioritized_value_iteration(1.0, env1, {})

        for s in range(env.nS):
            s0 = env0.locations_to_state((env.state_to_locations(s)[0],))
            s1 = env0.locations_to_state((env.state_to_locations(s)[1],))
            #
            # if dijkstra_func(s)!=vi_policy0.v[s0] + vi_policy1.v[s1]:
            #     import ipdb
            #     ipdb.set_trace()

            self.assertEqual(dijkstra_func(s), vi_policy0.v[s0] + vi_policy1.v[s1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
