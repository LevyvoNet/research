import time

import pytest
import stopit
import itertools
from gym_mapf.envs.grid import MapfGrid, SingleAgentAction, SingleAgentState
from gym_mapf.envs.mapf_env import MapfEnv, OptimizationCriteria, MultiAgentAction, MultiAgentState
from gym_mapf.envs.utils import create_mapf_env
from solvers.utils import evaluate_policy
from available_solvers import *
from tests.performance_utils import *
from typing import Callable

TEST_SINGLE_SCENARIO_TIMEOUT = 300

weak_tested_solvers = [
    value_iteration_describer,
    # policy_iteration_describer,
    # prioritized_value_iteration_describer,
    # id_vi_describer,
    # fixed_iter_rtdp_min_describer,
    # rtdp_stop_no_improvement_min_heuristic_describer,
    # long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
    ma_rtdp_sum_describer,
    # ma_rtdp_dijkstra_min_describer,
    # ma_rtdp_dijkstra_sum_describer,
]

mid_tested_solvers = [
    # id_ma_rtdp_min_pvi_describer,
    # long_ma_rtdp_min_pvi_describer,
    # long_ma_rtdp_min_dijkstra_describer,
    # id_rtdp_describer,
]

strong_tested_solvers = [
    # long_rtdp_stop_no_improvement_sum_heuristic_describer,
    # long_ma_rtdp_sum_pvi_describer,
    # long_id_rtdp_sum_pvi_describer,
    # long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
    # long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer,
    # long_ma_rtdp_sum_dijkstra_describer
]

all_tested_solvers = weak_tested_solvers + mid_tested_solvers + strong_tested_solvers


# Envs
def empty_grid_single_agent(optimization_criteria):
    grid = MapfGrid(['.' * 8] * 8)
    return MapfEnv(grid,
                   1,
                   MultiAgentState({0: SingleAgentState(7, 0)}, grid),
                   MultiAgentState({0: SingleAgentState(0, 7)}, grid),
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


easy_envs = [
    (empty_grid_single_agent, 'empty_grid_single_agent'),
    (partial(symmetrical_bottleneck, 0, 0), 'symmetrical_bottle_neck_deterministic'),
    (partial(symmetrical_bottleneck, 0, 100), 'symmetrical_bottle_neck_deterministic_large_goal_reward'),
    (partial(symmetrical_bottleneck, 0.2, 0), 'symmetrical_bottle_neck_stochastic'),
    (partial(symmetrical_bottleneck, 0.2, 100), 'symmetrical_bottle_neck_stochastic_large_goal_reward'),
    (partial(asymmetrical_bottleneck, 0, 0), 'Asymmetrical_bottle_neck_deterministic'),
    (partial(asymmetrical_bottleneck, 0, 100), 'Asymmetrical_bottle_neck_deterministic_large_goal_reward'),
    (partial(asymmetrical_bottleneck, 0.2, 0), 'Asymmetrical_bottle-neck_stochastic'),
    (partial(asymmetrical_bottleneck, 0.2, 100), 'Asymmetrical_bottle_neck_stochastic_large_goal_reward')
]

mid_envs = [
    (partial(room_32_32_4_2_agents, 12, 0), 'room-32-32-4_scen_12_2_agents_deterministic'),
    (partial(room_32_32_4_2_agents, 1, 0), 'room-32-32-4_scen_1_2_agents_deterministic'),
    (long_bottleneck, 'long_bottleneck_deterministic'),
    (partial(room_32_32_4_2_agents, 12, 0.1), 'room-32-32-4_scen_12_2_agents_stochastic'),
    (sanity_3_8, 'sanity_3_agents_stochastic')
]

difficult_envs = [
    (partial(room_32_32_4_2_agents, 13, 0), 'room-32-32-4_scen_13_2_agents_1_conflict_deterministic'),
    (sanity_2_32, 'conflict_between_pair_and_single_large_map')
]


def generate_solver_env_combinations():
    combs = []
    easy = itertools.product(easy_envs, all_tested_solvers)
    mid = itertools.product(mid_envs, mid_tested_solvers)
    difficult = itertools.product(difficult_envs, strong_tested_solvers)

    all_makespan = [(env_func, f'{env_name}_makespan', solver_describer, OptimizationCriteria.Makespan)
                    for (env_func, env_name), solver_describer in itertools.chain(easy, mid, difficult)]
    all_soc = [(env_func, f'{env_name}_soc', solver_describer, OptimizationCriteria.SoC)
               for (env_func, env_name), solver_describer in itertools.chain(easy, mid, difficult)]

    return all_soc + all_makespan


TEST_DATA = generate_solver_env_combinations()


@pytest.mark.parametrize('env_func, env_name, solver_describer, optimization_criteria', TEST_DATA)
def test_solver_on_env(env_func: Callable[[OptimizationCriteria], MapfEnv],
                       env_name: str,
                       solver_describer: SolverDescriber,
                       optimization_criteria: OptimizationCriteria):
    # Parameters and Constants
    env = env_func(optimization_criteria)
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
            print(f'\nsolver {solver_describer.short_description} got timeout on {env_name}', end=' ')
            assert False

    solve_time = round(time.time() - start, 2)

    reward, clashed, _ = evaluate_policy(policy, eval_n_episodes, eval_max_steps)

    print(f'\nenv:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.short_description}',
          end=' ')

    assert not clashed

    # Assert some kind of convergence
    if optimization_criteria == OptimizationCriteria.Makespan:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living + env.reward_of_goal  # Makespan
    else:
        min_converged_reward = (eval_max_steps - 1) * env.reward_of_living * env.n_agents + env.reward_of_goal  # SoC

    assert reward >= min_converged_reward


@pytest.mark.parametrize('solver_describer', all_tested_solvers)
def test_corridor_switch_no_clash_possible(solver_describer: SolverDescriber):
    print(f'env:corridor_switch, solver:{solver_describer.short_description}', end=' ')
    grid = MapfGrid(['...',
                     '@.@'])
    start_state = MultiAgentState({0: SingleAgentState(0, 0), 1: SingleAgentState(0, 2)}, grid)
    goal_state = MultiAgentState({0: SingleAgentState(0, 2), 1: SingleAgentState(0, 0)}, grid)

    # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
    env = MapfEnv(grid, 2, start_state, goal_state, 0.2, -0.001, 0, -1, OptimizationCriteria.Makespan)

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
    reward, clashed, _ = evaluate_policy(policy, 100, 200)

    print(f'\nenv:corridor_switch, reward:{reward}, time: {solve_time}, solver:{solver_describer.short_description}',
          end=' ')

    # Make sure no clash happened
    assert not clashed
    # Assert the reward is reasonable
    assert reward >= 100.0 * env.reward_of_living


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

    solver_describer = ma_rtdp_sum_describer

    # Try to solve with a time limit
    with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT * 1.5, swallow_exc=False) as timeout_ctx:
        try:
            info = {}
            policy = solver_describer.func(env, info)
        except stopit.utils.TimeoutException:
            print('timout')
            assert False

    solve_time = time.time() - start

    # import ipdb
    # ipdb.set_trace()

    reward, clashed, episode_rewards = evaluate_policy(policy, 100, 1000)

    print(f'env:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.short_description}', end=' ')


if __name__ == '__main__':
    main_profile()
