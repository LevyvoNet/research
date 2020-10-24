import unittest

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP, DOWN, STAY)
from gym_mapf.envs.utils import MapfGrid, create_mapf_env

from research.tests.solvers_tests.performance import (AbstractPerformanceTest,
                                                      WEAK_SOLVERS,
                                                      STRONG_SOLVERS,
                                                      EXPERIMENT_SOLVERS)


class EasyTestNo1(AbstractPerformanceTest):
    grid = MapfGrid(['...',
                     '@.@'])
    agents_starts = ((0, 0), (0, 2))
    agents_goals = ((0, 2), (0, 0))

    env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)
    env_str = 'MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)'
    solver_describers = WEAK_SOLVERS + STRONG_SOLVERS


class DifficultTestNo1(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:13;n_agents:2;fail_prob:0"
    solver_describers = STRONG_SOLVERS


class DifficultTestNo2(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:12;n_agents:2;fail_prob:0"
    solver_describers = STRONG_SOLVERS


class DifficultTestNo3(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1)
    env_str = "map:room-32-32-4;scen:1;n_agents:2;fail_prob:0"
    solver_describers = STRONG_SOLVERS


class DifficultTestNo4(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 12, 2, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:12;n_agents:2;fail_prob:0.1"
    solver_describers = STRONG_SOLVERS


class DifficultTestNo5(AbstractPerformanceTest):
    grid = MapfGrid([
        '...',
        '@.@',
        '@.@',
        '...'])

    agent_starts = ((0, 0), (0, 2))
    agents_goals = ((3, 0), (3, 2))

    env = MapfEnv(grid, 2, agent_starts, agents_goals,
                  0.0, 0.0, -1000, -1, -1)
    env_str = "MapfEnv(grid, 2, agent_starts, agents_goals,0.0, 0.0, -1000, -1, -1)"
    solver_describers = STRONG_SOLVERS


class DifficultTestNo6(AbstractPerformanceTest):
    env = create_mapf_env('empty-8-8', 1, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:empty-8-8;scen:1;n_agents:3;fail_prob:0.1; 2 conflicts"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo7(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 2, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:2;n_agents:3;fail_prob:0.1;no conflicts"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo8(AbstractPerformanceTest):
    env = create_mapf_env('empty-8-8', 6, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:empty-8-8;scen:6;n_agents:3;fail_prob:0.1"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo9(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 10, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:10;n_agents:3;fail_prob:0.1;no conflicts"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo10(AbstractPerformanceTest):
    env = create_mapf_env('room-32-32-4', 12, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:room-32-32-4;scen:12;n_agents:3;fail_prob:0.1;no conflicts"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo11(AbstractPerformanceTest):
    env = create_mapf_env('empty-16-16', 7, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:empty-16-16;scen:7;n_agents:3;fail_prob:0.1;2 conflicts"
    solver_describers = EXPERIMENT_SOLVERS


class DifficultTestNo12(AbstractPerformanceTest):
    env = create_mapf_env('empty-16-16', 1, 3, 0.1, 0.1, -1000, -1, -1)
    env_str = "map:empty-16-16;scen:1;n_agents:3;fail_prob:0.1;2 conflicts"
    solver_describers = EXPERIMENT_SOLVERS


if __name__ == '__main__':
    unittest.main(verbosity=2)
