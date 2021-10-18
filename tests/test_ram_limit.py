import unittest
from functools import partial
from typing import Dict, Callable
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.envs.utils import create_mapf_env
from solvers.utils import Policy
from solvers import (ValueIterationPolicy, PrioritizedValueIterationPolicy, PolicyIterationPolicy)


class RamLimitTest(unittest.TestCase):
    def solve(self, env) -> Callable[[MapfEnv, Dict], Policy]:
        """Return the concrete planner"""
        raise unittest.SkipTest("This is an abstract test case")

    def test_multiple_agents_env(self):
        """Assert that when trying to solver a large environment we are not exceeding the RAM limit."""
        # Note the large number of agents
        env = create_mapf_env('room-32-32-4', 12, 6, 0.1, 0.1, -1000, -1, -1)

        policy = self.solve(env)

        self.assertIs(policy.v, None)
        self.assertEqual(policy.info['end_reason'], 'out_of_memory')


class ValueIterationRamLimitTest(RamLimitTest):
    def solve(self, env) -> Callable[[MapfEnv, Dict], Policy]:
        return ValueIterationPolicy().attach_env(env, 1.0).train()


class PrioritizedValueIterationRamLimitTest(RamLimitTest):
    def solve(self, env) -> Callable[[MapfEnv, Dict], Policy]:
        return PrioritizedValueIterationPolicy().attach_env(env, 1.0).train()


class PolicyIterationRamLimitTest(RamLimitTest):
    def solve(self, env) -> Callable[[MapfEnv, Dict], Policy]:
        return PolicyIterationPolicy().attach_env(env, 1.0).train()


if __name__ == '__main__':
    unittest.main(verbosity=2)
