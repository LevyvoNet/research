import pytest
import itertools

from available_solvers import *
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    OptimizationCriteria,
                                    vector_action_to_integer,
                                    STAY,
                                    UP,
                                    DOWN)
from solvers.utils import Policy
from tests.benchmark_solvers_on_envs import (lvl_to_solvers,
                                             benchmark_solver_on_env,
                                             sanity_independent,
                                             RESULT_OK,
                                             symmetrical_bottleneck)


def generate_all_solvers():
    all_makespan = [
        (policy, OptimizationCriteria.Makespan)
        for policy in itertools.chain(*lvl_to_solvers.values())
    ]

    all_soc = [
        (policy, OptimizationCriteria.SoC)
        for policy in itertools.chain(*lvl_to_solvers.values())
    ]

    return all_makespan + all_soc


all_tested_solvers = generate_all_solvers()


@pytest.mark.parametrize('policy, optimization_criteria', all_tested_solvers)
def test_corridor_switch_no_clash_possible(policy: Policy,
                                           optimization_criteria: OptimizationCriteria):
    eval_max_steps = 200
    eval_n_episodes = 100

    grid = MapfGrid(['...',
                     '@.@'])
    start_locations = ((0, 0), (0, 2))
    goal_locations = ((0, 2), (0, 0))

    # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
    env = MapfEnv(grid, 2, start_locations, goal_locations, 0.2, -0.001, 0, -1, optimization_criteria)

    policy.attach_env(env, 1.0).train()

    # Assert no conflict is possible
    interesting_locations = ((1, 1), (0, 1))
    interesting_state = env.locations_to_state(interesting_locations)
    stay_up = vector_action_to_integer((STAY, UP))
    down_up = vector_action_to_integer((DOWN, UP))
    expected_possible_actions = [stay_up, down_up]

    assert policy.act(interesting_state) in expected_possible_actions

    eval_info = policy.evaluate(eval_n_episodes, eval_max_steps)

    assert not eval_info['clashed']

    assert eval_info['success_rate'] > 0
