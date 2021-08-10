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
    policy_iteration_describer,
    prioritized_value_iteration_describer,
    # id_vi_describer,
    fixed_iter_rtdp_min_describer,
    rtdp_stop_no_improvement_min_heuristic_describer,
    long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
    ma_rtdp_sum_describer,
    ma_rtdp_dijkstra_min_describer,
    ma_rtdp_dijkstra_sum_describer,
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
def empty_grid_single_agent():
    grid = MapfGrid(['.' * 8] * 8)
    return MapfEnv(grid,
                   1,
                   MultiAgentState({0: SingleAgentState(7, 0)}, grid),
                   MultiAgentState({0: SingleAgentState(0, 7)}, grid),
                   0.2,
                   -1000, 0, -1,
                   OptimizationCriteria.Makespan)


def symmetrical_bottleneck(fail_prob, goal_reward):
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
                   OptimizationCriteria.Makespan)


def asymmetrical_bottleneck(fail_prob, goal_reward):
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
                   OptimizationCriteria.Makespan)


easy_envs = [
    (empty_grid_single_agent, 'empty_grid_single_agent'),
    (partial(symmetrical_bottleneck, 0, 0), 'symmetrical bottle-neck deterministic'),
    (partial(symmetrical_bottleneck, 0, 100), 'symmetrical bottle-neck deterministic large goal reward'),
    (partial(symmetrical_bottleneck, 0.2, 0), 'symmetrical bottle-neck stochastic'),
    (partial(symmetrical_bottleneck, 0.2, 100), 'symmetrical bottle-neck stochastic large goal reward'),
    (partial(asymmetrical_bottleneck, 0, 0), 'Asymmetrical bottle-neck deterministic'),
    (partial(asymmetrical_bottleneck, 0, 100), 'Asymmetrical bottle-neck deterministic large goal reward'),
    (partial(asymmetrical_bottleneck, 0.2, 0), 'Asymmetrical bottle-neck stochastic'),
    (partial(asymmetrical_bottleneck, 0.2, 100), 'Asymmetrical bottle-neck stochastic large goal reward')
]

mid_envs = [
    # (
    #     create_mapf_env('room-32-32-4', 12, 2, 0, 0, -1000, 0, -1),
    #     'room-32-32-4 scen 12 - 2 agents deterministic'
    # ),
    # (
    #     create_mapf_env('room-32-32-4', 1, 2, 0, 0, -1000, 0, -1),
    #     'room-32-32-4 scen 1 - 2 agents deterministic'
    # ),
    # (
    #     MapfEnv(MapfGrid([
    #         '...',
    #         '@.@',
    #         '@.@',
    #         '...']), 2, ((0, 0), (0, 2)), ((3, 0), (3, 2)), 0.0, 0.0, -1000, 0, -1),
    #     'hand crafted env'
    # ),
    # (
    #     create_mapf_env('room-32-32-4', 12, 2, 0.1, 0.1, -1000, 0, -1),
    #     'room-32-32-4 scen 12 - stochastic'
    # ),
    # (
    #     create_mapf_env('sanity-3-8', None, 3, 0.1, 0.1, -1000, 0, -1),
    #     'sanity 3 agents stochastic'
    # ),
]

difficult_envs = [
    # (
    #     create_mapf_env('room-32-32-4', 13, 2, 0, 0, -1000, 0, -1),
    #     'room-32-32-4 scen 13 - 2 agents 1 conflict'
    # ),
    # (
    #     create_mapf_env('sanity-2-32', 1, 3, 0.1, 0.1, -1000, 0, -1),
    #     'conflict between pair and single large map'
    # )
]


def generate_solver_env_combinations():
    # Initialize with all solvers on easy envs
    combs = [(env_func, env_name, solver_describer)
             for (env_func, env_name), solver_describer in itertools.product(easy_envs, all_tested_solvers)]

    # Add mid and strong solvers for mid envs
    combs += [(env_func, env_name, solver_describer)
              for (env_func, env_name), solver_describer in itertools.product(mid_envs, mid_tested_solvers)]

    # Add the strong solvers on the difficult envs
    combs += [(env_func, env_name, solver_describer)
              for (env_func, env_name), solver_describer in itertools.product(difficult_envs, strong_tested_solvers)]

    return combs


TEST_DATA = generate_solver_env_combinations()


@pytest.mark.parametrize('env_func, env_name, solver_describer', TEST_DATA)
def test_solver_on_env(env_func: Callable[[], MapfEnv], env_name: str, solver_describer: SolverDescriber):
    # print(f'starting env:{env_name}, solver:{solver_describer.short_description}', end=' ')

    env = env_func()
    info = {}
    start = time.time()

    # Try to solve with a time limit
    with stopit.SignalTimeout(TEST_SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
        try:
            policy = solver_describer.func(env, info)
        except stopit.utils.TimeoutException:
            print(f'solver {solver_describer.description} got timeout on {env_name}', end=' ')
            assert False

    solve_time = round(time.time() - start, 2)

    reward, clashed, _ = evaluate_policy(policy, 100, 1000)

    print(f'env:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.short_description}', end=' ')

    assert not clashed

    # Assert the reward was not -1000
    assert reward >= 999 * env.reward_of_living + env.reward_of_goal

    # # TODO: delete this afterwards
    # for s in env.observation_space:
    #     if not env._is_terminal_state(s) and policy.act(s) != policy._act_in_unfamiliar_state(s):
    #         bug_sum = sum([p * (r + policy.v[s_next])
    #                        for p, s_next, r, _ in env.P[s][policy._act_in_unfamiliar_state(s)]])
    #         real_sum = sum([p * (r + policy.v[s_next])
    #                         for p, s_next, r, _ in env.P[s][policy.act(s)]])
    #         if bug_sum != real_sum or abs(policy.v[s] - bug_sum) > 1e-2:
    #             import ipdb
    #             ipdb.set_trace()
    #
    #         real_action = policy.act(s)
    #         bug_action = policy._act_in_unfamiliar_state(s)
    #         print(f'real_action={real_action}')
    #         print(f'bug_action={bug_action}')


@pytest.mark.parametrize('solver_describer', all_tested_solvers)
def test_corridor_switch_no_clash_possible(solver_describer: SolverDescriber):
    print(f'env:corridor_switch, solver:{solver_describer.description}', end=' ')
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

    print(f'env:corridor_switch, reward:{reward}, time: {solve_time}, solver:{solver_describer.description}', end=' ')

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
    with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT*1.5, swallow_exc=False) as timeout_ctx:
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

    print(f'env:{env_name}, reward:{reward}, time: {solve_time}, solver:{solver_describer.description}', end=' ')


if __name__ == '__main__':
    main_profile()
