import math
import time
import gc
import datetime

import pytest
import stopit
import itertools
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria
from gym_mapf.envs.utils import create_mapf_env
from solvers.utils import evaluate_policy
from available_solvers import *
from tests.performance_utils import *
from typing import Callable

TEST_SINGLE_SCENARIO_TIMEOUT = 300


# Envs
def empty_grid_single_agent(optimization_criteria):
    grid = MapfGrid(['.' * 8] * 8)
    return MapfEnv(grid,
                   1,
                   ((7, 0),),
                   ((0, 7),),
                   0.2,
                   -1000, 0, -1,
                   optimization_criteria)


def symmetrical_bottleneck(fail_prob, goal_reward, optimization_criteria):
    grid = MapfGrid(['..@...',
                     '..@...',
                     '......',
                     '..@...'
                     '..@...'])
    return MapfEnv(grid,
                   2,
                   MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 5)}, grid),
                   MultiAgentState({0: SingleAgentState(2, 5), 1: SingleAgentState(2, 0)}, grid),
                   fail_prob,
                   -0.001, goal_reward, -1,
                   optimization_criteria)


def asymmetrical_bottleneck(fail_prob, goal_reward, optimization_criteria):
    grid = MapfGrid(['..@..',
                     '..@..',
                     '.....',
                     '..@..'
                     '..@..'])
    return MapfEnv(grid,
                   2,
                   MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 4)}, grid),
                   MultiAgentState({0: SingleAgentState(2, 4), 1: SingleAgentState(2, 0)}, grid),
                   fail_prob,
                   -0.001, goal_reward, -1,
                   optimization_criteria)


def room_32_32_4_2_agents(scen_id, fail_prob, optimization_criteria):
    return create_mapf_env('room-32-32-4', scen_id, 2, fail_prob, -1000, 0, -1, optimization_criteria)


def long_bottleneck(optimization_criteria):
    grid = MapfGrid(['...',
                     '@.@',
                     '@.@',
                     '...'])
    start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
    goal_state = MultiAgentState({0: SingleAgentState(3, 0), 1: SingleAgentState(3, 2)}, grid)

    return MapfEnv(grid, 2, start_state, goal_state, 0, -1000, 0, -1, optimization_criteria)


def sanity_3_8(optimization_criteria):
    return create_mapf_env('sanity-3-8', None, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_2_32(optimization_criteria):
    return create_mapf_env('sanity-2-32', 1, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_general(n_rooms, n_agents, room_size, fail_prob, optimization_criteria):
    return create_mapf_env(f'sanity-{n_rooms}-{room_size}', 1, n_agents, fail_prob, -1000, 0, -1, optimization_criteria)


lvl_to_solvers = {
    0: [
        # value_iteration_describer,
        # policy_iteration_describer,
        # prioritized_value_iteration_describer,
        # id_vi_describer,
        # fixed_iter_rtdp_min_describer,
        # rtdp_stop_no_improvement_min_heuristic_describer,
        # long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
        ma_rtdp_pvi_sum_describer,
        # ma_rtdp_dijkstra_min_describer,
        # ma_rtdp_dijkstra_sum_describer,
    ],
    1: [
        # id_ma_rtdp_min_pvi_describer,
        # id_rtdp_describer,
    ],
    2: [
        # long_rtdp_stop_no_improvement_sum_heuristic_describer,
        # long_id_rtdp_sum_pvi_describer,
        # long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
        # long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer,
        # long_ma_rtdp_min_pvi_describer,
        # long_ma_rtdp_min_dijkstra_describer,
        # long_ma_rtdp_min_rtdp_dijkstra_describer
    ],
    3: [
        # long_ma_rtdp_sum_pvi_describer,
        # long_ma_rtdp_sum_dijkstra_describer,
        # long_ma_rtdp_sum_rtdp_dijkstra_describer,
        # long_id_rtdp_sum_pvi_describer
    ]
}

lvl_to_env = {
    0: [
        (empty_grid_single_agent, 'empty_grid_single_agent'),
        (partial(symmetrical_bottleneck, 0, 0), 'symmetrical_bottle_neck_deterministic'),
        (partial(symmetrical_bottleneck, 0, 100), 'symmetrical_bottle_neck_deterministic_large_goal_reward'),
        (partial(symmetrical_bottleneck, 0.2, 0), 'symmetrical_bottle_neck_stochastic'),
        (partial(symmetrical_bottleneck, 0.2, 100), 'symmetrical_bottle_neck_stochastic_large_goal_reward'),
        (partial(asymmetrical_bottleneck, 0, 0), 'Asymmetrical_bottle_neck_deterministic'),
        (partial(asymmetrical_bottleneck, 0, 100), 'Asymmetrical_bottle_neck_deterministic_large_goal_reward'),
        (partial(asymmetrical_bottleneck, 0.2, 0), 'Asymmetrical_bottle-neck_stochastic'),
        (partial(asymmetrical_bottleneck, 0.2, 100), 'Asymmetrical_bottle_neck_stochastic_large_goal_reward')
    ],
    1: [
        (partial(room_32_32_4_2_agents, 12, 0), 'room-32-32-4_scen_12_2_agents_deterministic'),
        (partial(room_32_32_4_2_agents, 1, 0), 'room-32-32-4_scen_1_2_agents_deterministic'),
        (long_bottleneck, 'long_bottleneck_deterministic'),
        (partial(room_32_32_4_2_agents, 12, 0.2), 'room-32-32-4_scen_12_2_agents_stochastic'),
        (partial(room_32_32_4_2_agents, 1, 0.2), 'room-32-32-4_scen_1_2_agents_stochastic'),
        (sanity_3_8, 'sanity_3_agents_stochastic'),

    ],
    2: [
        (partial(room_32_32_4_2_agents, 13, 0), 'room-32-32-4_scen_13_2_agents_1_conflict_deterministic'),
        (partial(room_32_32_4_2_agents, 13, 0.2), 'room-32-32-4_scen_13_2_agents_1_conflict_stochastic'),

    ],
    3: [
        (sanity_2_32, 'conflict_between_pair_and_single_large_map'),
    ]
}


def generate_solver_env_combinations(max_env_lvl):
    all_soc = []
    all_makespan = []
    for env_lvl in range(max_env_lvl + 1):
        for (env_func, env_name) in lvl_to_env[env_lvl]:
            for solver_lvl in lvl_to_solvers.keys():
                if solver_lvl >= env_lvl:
                    for solver_describer in lvl_to_solvers[solver_lvl]:
                        all_soc.append((env_func,
                                        f'{env_name}_soc',
                                        solver_describer,
                                        solver_describer.short_description,
                                        OptimizationCriteria.SoC))

                        all_makespan.append((env_func,
                                             f'{env_name}_makespan',
                                             solver_describer,
                                             solver_describer.short_description,
                                             OptimizationCriteria.Makespan))

    return all_makespan + all_soc


def generate_all_solvers():
    all_makespan = [
        (solver_describer, solver_describer.short_description, OptimizationCriteria.Makespan)
        for solver_describer in itertools.chain(*lvl_to_solvers.values())
    ]

    all_soc = [
        (solver_describer, solver_describer.short_description, OptimizationCriteria.SoC)
        for solver_describer in itertools.chain(*lvl_to_solvers.values())
    ]

    return all_makespan + all_soc


ALL_SOLVERS = generate_all_solvers()
TEST_DATA = generate_solver_env_combinations(max(lvl_to_env.keys()))


def cleanup_on_fail(cond, env, policy, test_done=False):
    if not cond:
        del policy.env
        del policy
        del env
        gc.collect()

    if not test_done:
        assert cond


def cleanup(env, policy):
    cleanup_on_fail(False, env, policy, test_done=True)


def print_status(env_name, reward, solve_time, solver_description):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f'\n{now_str} env:{env_name}, reward:{reward}, time:{solve_time}, solver:{solver_description}', end=' ')


@pytest.mark.parametrize('env_func, env_name, solver_describer, solver_name, optimization_criteria', TEST_DATA)
def test_solver_on_env(env_func: Callable[[OptimizationCriteria], MapfEnv],
                       env_name: str,
                       solver_describer: SolverDescriber,
                       solver_name: str,
                       optimization_criteria: OptimizationCriteria):
    # Parameters and Constants
    env = env_func(optimization_criteria)
    policy = None
    info = {}
    eval_max_steps = 1000
    eval_n_episodes = 100

    # Start the test
    start = time.time()

    # Try to solve with a time limit
    with stopit.SignalTimeout(TEST_SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
        try:
            policy = solver_describer.func(env, info)
        except stopit.utils.TimeoutException:
            print_status(env_name, -math.inf, 'timeout', solver_describer.short_description)
            cleanup_on_fail(False, env, policy)
            # assert False

    solve_time = round(time.time() - start, 2)

    reward, clashed, _ = evaluate_policy(policy, eval_n_episodes, eval_max_steps)

    print_status(env_name, reward, solve_time, solver_describer.short_description)

    cleanup_on_fail(not clashed, env, policy)
    # assert not clashed

    # Assert some kind of convergence
    if optimization_criteria == OptimizationCriteria.Makespan:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living + env.reward_of_goal  # Makespan
    else:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living * env.n_agents + env.reward_of_goal  # SoC

    cleanup_on_fail(reward >= min_converged_reward, env, policy)

    cleanup(env, policy)


@pytest.mark.parametrize('solver_describer, solver_name, optimization_criteria', ALL_SOLVERS)
def test_corridor_switch_no_clash_possible(solver_describer: SolverDescriber,
                                           solver_name: str,
                                           optimization_criteria: OptimizationCriteria):
    optimization_criteria_to_str = {
        OptimizationCriteria.Makespan: 'makespan',
        OptimizationCriteria.SoC: 'soc'
    }
    env_name = f'corridor_switch_{optimization_criteria_to_str[optimization_criteria]}'
    eval_max_steps = 200
    n_episodes = 100

    grid = MapfGrid(['...',
                     '@.@'])
    start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
    goal_state = MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 0)}, grid)

    # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
    env = MapfEnv(grid, 2, start_state, goal_state, 0.2, -0.001, 0, -1, optimization_criteria)

    info = {}
    start = time.time()
    policy = solver_describer.func(env, info)
    solve_time = round(time.time() - start, 2)

    # Assert no conflict is possible
    interesting_state = MultiAgentState({0: SingleAgentState(1, 1), 1: SingleAgentState(0, 1)}, grid)
    expected_possible_actions = [
        MultiAgentAction({0: SingleAgentAction.STAY, 1: SingleAgentAction.UP}),
        MultiAgentAction({0: SingleAgentAction.DOWN, 1: SingleAgentAction.UP})
    ]

    assert policy.act(interesting_state) in expected_possible_actions

    # Check the policy performance
    reward, clashed, _ = evaluate_policy(policy, n_episodes, eval_max_steps)

    print_status(env_name, reward, solve_time, solver_describer.short_description)

    # Make sure no clash happened
    assert not clashed
    # Assert the reward is reasonable
    # Assert some kind of convergence
    if optimization_criteria == OptimizationCriteria.Makespan:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living + env.reward_of_goal  # Makespan
    else:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living * env.n_agents + env.reward_of_goal  # SoC

    assert reward >= min_converged_reward


def main_profile():
    start = time.time()
    grid = MapfGrid(['..@..',
                     '..@..',
                     '.....',
                     '..@..'
                     '..@..'])
    env = MapfEnv(grid,
                  2,
                  MultiAgentState({0: SingleAgentState(2, 0), 1: SingleAgentState(2, 4)}, grid),
                  MultiAgentState({0: SingleAgentState(2, 4), 1: SingleAgentState(2, 0)}, grid),
                  0,
                  -0.001, 100, -1,
                  OptimizationCriteria.Makespan)
    env_name = 'symmetrical bottle-neck deterministic large goal reward'

    solver_describer = ma_rtdp_sum_pvi_describer

    # Try to solve with a time limit
    with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT * 1.5, swallow_exc=False) as timeout_ctx:
        try:
            info = {}
            policy = solver_describer.func(env, info)
        except stopit.utils.TimeoutException:
            print('timout')
            assert False

    solve_time = time.time() - start

    reward, clashed, episode_rewards = evaluate_policy(policy, 100, 1000)

    print(f'env:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.short_description}', end=' ')


if __name__ == '__main__':
    main_profile()
