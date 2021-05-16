import pytest
import stopit
import itertools
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    vector_action_to_integer,
                                    UP,
                                    DOWN,
                                    STAY)
from gym_mapf.envs.utils import create_mapf_env
from solvers.utils import evaluate_policy
from available_solvers import *
from tests.performance_utils import *

TEST_SINGLE_SCENARIO_TIMEOUT = 300

weak_tested_solvers = [
    value_iteration_describer,
    policy_iteration_describer,
    fixed_iter_rtdp_min_describer,
    rtdp_stop_no_improvement_min_heuristic_describer,
    id_vi_describer,
    long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
    ma_rtdp_sum_describer,
    ma_rtdp_dijkstra_min_describer,
    ma_rtdp_dijkstra_sum_describer,
]

mid_tested_solvers = [
    id_ma_rtdp_describer,
    long_ma_rtdp_min_pvi_describer,
    long_ma_rtdp_min_dijkstra_describer,
    id_rtdp_describer,
]

strong_tested_solvers = [
    long_rtdp_stop_no_improvement_sum_heuristic_describer,
    long_ma_rtdp_sum_pvi_describer,
    long_id_rtdp_sum_pvi_describer,
    long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
    long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer,
    long_ma_rtdp_sum_dijkstra_describer
]

all_tested_solvers = weak_tested_solvers + strong_tested_solvers

easy_envs = [
    (
        MapfEnv(MapfGrid(['.' * 8] * 8), 1, ((7, 0),), ((0, 7),), 0.1, 0.1, -1000, -1, -1),
        'empty_grid_single_agent'
    ),

    (
        MapfEnv(MapfGrid(['..@...',
                          '..@...',
                          '......',
                          '..@...'
                          '..@...']), 2, ((2, 0), (2, 5)), ((2, 5), (2, 0)), 0, 0, -0.001, -1, -1),
        'symmetrical bottle-neck deterministic'
    ),

    (
        MapfEnv(MapfGrid(['..@...',
                          '..@...',
                          '......',
                          '..@...'
                          '..@...']), 2, ((2, 0), (2, 5)), ((2, 5), (2, 0)), 0, 0, -0.001, 100, -1),
        'symmetrical bottle-neck deterministic large goal reward'
    ),

    (
        MapfEnv(MapfGrid(['..@...',
                          '..@...',
                          '......',
                          '..@...'
                          '..@...']), 2, ((2, 0), (2, 5)), ((2, 5), (2, 0)), 0.1, 0.1, -0.001, -1, -1),
        'symmetrical bottle-neck stochastic'
    ),

    (
        MapfEnv(MapfGrid(['..@..',
                          '..@..',
                          '.....',
                          '..@..'
                          '..@..']), 2, ((2, 0), (2, 4)), ((2, 4), (2, 0)), 0, 0, -0.001, -1, -1),
        'Asymmetrical bottle-neck deterministic'
    ),

    (
        MapfEnv(MapfGrid(['..@..',
                          '..@..',
                          '.....',
                          '..@..'
                          '..@..']), 2, ((2, 0), (2, 4)), ((2, 4), (2, 0)), 0, 0, -0.001, 100, -1),
        'Asymmetrical bottle-neck deterministic large goal reward'
    ),

    (
        MapfEnv(MapfGrid(['..@..',
                          '..@..',
                          '.....',
                          '..@..'
                          '..@..']), 2, ((2, 0), (2, 4)), ((2, 4), (2, 0)), 0, 0, -0.001, 100, -1),
        'Asymmetrical bottle-neck stochastic'
    ),
]

mid_envs = [
    (
        create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, -1, -1),
        'room-32-32-4 scen 12 - 2 agents deterministic'
    ),
    (
        create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1),
        'room-32-32-4 scen 1 - 2 agents deterministic'
    ),
    (
        MapfEnv(MapfGrid([
            '...',
            '@.@',
            '@.@',
            '...']), 2, ((0, 0), (0, 2)), ((3, 0), (3, 2)), 0.0, 0.0, -1000, -1, -1),
        'hand crafted env'
    ),
    (
        create_mapf_env('room-32-32-4', 12, 2, 0.1, 0.1, -1000, -1, -1),
        'room-32-32-4 scen 12 - stochastic'
    ),
    (
        create_mapf_env('sanity-3-8', None, 3, 0.1, 0.1, -1000, -1, -1),
        'sanity 3 agents stochastic'
    ),
]

difficult_envs = [
    (
        create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, -1, -1),
        'room-32-32-4 scen 13 - 2 agents 1 conflict'
    ),
    (
        create_mapf_env('sanity-2-32', 1, 3, 0.1, 0.1, -1000, -1, -1),
        'conflict between pair and single large map'
    )
]


def generate_solver_env_combinations():
    # Initialize with all solvers on easy envs
    combs = [(env, env_name, solver_describer)
             for (env, env_name), solver_describer in itertools.product(easy_envs, all_tested_solvers)]

    # Add mid and strong solvers for mid envs
    combs += [(env, env_name, solver_describer)
              for (env, env_name), solver_describer in itertools.product(mid_envs, mid_tested_solvers)]

    # Add the strong solvers on the difficult envs
    combs += [(env, env_name, solver_describer)
              for (env, env_name), solver_describer in itertools.product(difficult_envs, strong_tested_solvers)]

    return combs


TEST_DATA = generate_solver_env_combinations()


@pytest.mark.parametrize('env, env_name, solver_describer', TEST_DATA)
def test_solver_on_env(env: MapfEnv, env_name: str, solver_describer: SolverDescriber):
    info = {}
    start = time.time()

    # Try to solve with a time limit
    with stopit.SignalTimeout(TEST_SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
        try:
            policy = solver_describer.func(env, info)
        except stopit.utils.TimeoutException:
            print(f'solver {solver_describer.description} got timeout on {env_name}', end=' ')
            import ipdb
            ipdb.set_trace()
            assert False

    solve_time = round(time.time() - start, 2)

    reward, clashed, _ = evaluate_policy(policy, 100, 1000)

    print(f'env:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.description}', end=' ')

    assert not clashed

    # Assert the reward was not -1000
    assert reward >= 999 * env.reward_of_living + env.reward_of_goal


@pytest.mark.parametrize('solver_describer', all_tested_solvers)
def test_corridor_switch_no_clash_possible(solver_describer: SolverDescriber):
    grid = MapfGrid(['...',
                     '@.@'])
    agents_starts = ((0, 0), (0, 2))
    agents_goals = ((0, 2), (0, 0))

    # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
    env = MapfEnv(grid, 2, agents_starts, agents_goals, 0.1, 0.1, -0.001, -1, -1)

    info = {}
    policy = solver_describer.func(env, info)

    # Assert no conflict is possible
    interesting_state = env.locations_to_state(((1, 1), (0, 1)))
    expected_possible_actions = [vector_action_to_integer((STAY, UP)),
                                 vector_action_to_integer((DOWN, UP))]

    assert policy.act(interesting_state) in expected_possible_actions

    # Check the policy performance
    reward, clashed, _ = evaluate_policy(policy, 100, 100)

    # Make sure no clash happened
    assert not clashed

    # Assert the reward is reasonable
    assert reward >= 100.0 * env.reward_of_living
