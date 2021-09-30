import math
import time
import datetime

import stopit
import itertools
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    OptimizationCriteria,
                                    vector_action_to_integer,
                                    STAY,
                                    UP,
                                    DOWN)
from gym_mapf.envs.utils import create_mapf_env

from solvers.utils import evaluate_policy
from available_solvers import *
# from tests.performance_utils import *
from typing import Callable

TEST_SINGLE_SCENARIO_TIMEOUT = 300

RESULT_CLASHED = 'CLASHED'
RESULT_TIMEOUT = 'TIMEOUT'
RESULT_NOT_CONVERGED = 'NOT_CONVERGED'
RESULT_DANGEROUS_ACTION = 'DANGEROUS_ACTION'
RESULT_EXCEPTION = 'EXCEPTION'
RESULT_OK = 'OK'


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
                   ((2, 0), (2, 5)),
                   ((2, 5), (2, 0)),
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
                   ((2, 0), (2, 4)),
                   ((2, 4), (2, 0)),
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
    start_locations = ((0, 0), (0, 2))
    goal_locations = ((3, 0), (3, 2))

    return MapfEnv(grid, 2, start_locations, goal_locations, 0, -1000, 0, -1, optimization_criteria)


def sanity_3_8(optimization_criteria):
    return create_mapf_env('sanity-3-8', None, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_2_32(optimization_criteria):
    return create_mapf_env('sanity-2-32', 1, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_general(n_rooms, n_agents, room_size, fail_prob, optimization_criteria):
    return create_mapf_env(f'sanity-{n_rooms}-{room_size}', 1, n_agents, fail_prob, -1000, 0, -1, optimization_criteria)


lvl_to_solvers = {
    0: [
        value_iteration_describer,
        policy_iteration_describer,
        prioritized_value_iteration_describer,
        id_vi_describer,
        fixed_iter_rtdp_min_describer,
        rtdp_stop_no_improvement_min_heuristic_describer,
        ma_rtdp_pvi_sum_describer,
        ma_rtdp_dijkstra_min_describer,
        ma_rtdp_dijkstra_sum_describer,
        ma_rtdp_pvi_min_describer,
    ],
    1: [
        id_ma_rtdp_pvi_min_describer,
        id_ma_rtdp_pvi_sum_describer,
        id_rtdp_pvi_min_describer,
        id_rtdp_pvi_sum_describer,
    ],
    2: [
        long_rtdp_stop_no_improvement_sum_heuristic_describer,
        long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
        long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer,
        long_ma_rtdp_pvi_min_describer,
        long_ma_rtdp_min_dijkstra_describer,
        long_id_ma_rtdp_min_dijkstra_describer,
        long_id_ma_rtdp_min_pvi_describer,
        long_id_ma_rtdp_min_rtdp_dijkstra_describer,
        long_ma_rtdp_min_rtdp_dijkstra_describer
    ],
    3: [
        long_ma_rtdp_pvi_sum_describer,
        long_ma_rtdp_sum_dijkstra_describer,
        long_ma_rtdp_sum_rtdp_dijkstra_describer,
        long_id_rtdp_sum_pvi_describer,
        long_id_ma_rtdp_sum_pvi_describer,
        long_id_ma_rtdp_sum_dijkstra_describer,
        long_id_ma_rtdp_sum_rtdp_dijkstra_describer
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
                                        OptimizationCriteria.SoC))

                        all_makespan.append((env_func,
                                             f'{env_name}_makespan',
                                             solver_describer,
                                             OptimizationCriteria.Makespan))

    return all_makespan + all_soc


def generate_all_solvers():
    all_makespan = [
        (solver_describer, OptimizationCriteria.Makespan)
        for solver_describer in itertools.chain(*lvl_to_solvers.values())
    ]

    all_soc = [
        (solver_describer, OptimizationCriteria.SoC)
        for solver_describer in itertools.chain(*lvl_to_solvers.values())
    ]

    return all_makespan + all_soc


ALL_SOLVERS = generate_all_solvers()
TEST_DATA = generate_solver_env_combinations(max(lvl_to_env.keys()))


def print_status(env_name, reward, solve_time, solver_description, success_rate, extra_info: SolverExtraInfo = None):
    status_str = ', '.join([
        f'reward:{reward}',
        f'rate:{success_rate}',
        f'time:{solve_time}',
        f'solver:{solver_description}, ',
    ])

    if extra_info is None:
        extra_info_str = ''
    else:
        extra_info_str = ', '.join([
            f'init_time:{extra_info.solver_init_time}',
            f'eval_time:{extra_info.total_evaluation_time}',
            f'n_conflicts:{extra_info.n_conflicts}',
            f'conflicts_time:{extra_info.conflict_detection_time}',
            f'iters:{extra_info.n_iterations}',
            f'visited:{extra_info.n_visited_states}',
            f'last:{extra_info.last_MDR}',
        ])

    print(status_str + extra_info_str)


def benchmark_solver_on_env(env_func: Callable[[OptimizationCriteria], MapfEnv],
                            env_name: str,
                            solver_describer: SolverDescriber,
                            optimization_criteria: OptimizationCriteria):
    try:
        # Parameters and Constants
        env = env_func(optimization_criteria)
        train_info = {}
        eval_max_steps = 1000
        eval_n_episodes = 100

        # Start the test
        start = time.time()

        # Try to solve with a time limit
        with stopit.SignalTimeout(TEST_SINGLE_SCENARIO_TIMEOUT, swallow_exc=False):
            try:
                policy = solver_describer.func(env, train_info)
            except stopit.utils.TimeoutException:
                extra_info = solver_describer.extra_info(train_info)
                print_status(env_name, -math.inf, 'timeout', solver_describer.short_description, 0, extra_info)
                return RESULT_TIMEOUT

        solve_time = round(time.time() - start, 2)

        eval_info = evaluate_policy(policy, eval_n_episodes, eval_max_steps)
        extra_info = solver_describer.extra_info(train_info)

        print_status(env_name,
                     eval_info['MDR'],
                     solve_time,
                     solver_describer.short_description,
                     eval_info['success_rate'],
                     extra_info)

        if eval_info['clashed']:
            return RESULT_CLASHED

        if eval_info['success_rate'] == 0:
            return RESULT_NOT_CONVERGED

        return RESULT_OK

    except Exception as e:
        print(repr(e))
        return RESULT_EXCEPTION


def test_corridor_switch_no_clash_possible(solver_describer: SolverDescriber,
                                           optimization_criteria: OptimizationCriteria):
    optimization_criteria_to_str = {
        OptimizationCriteria.Makespan: 'makespan',
        OptimizationCriteria.SoC: 'soc'
    }
    env_name = f'corridor_switch_{optimization_criteria_to_str[optimization_criteria]}'
    eval_max_steps = 200
    eval_n_episodes = 100

    grid = MapfGrid(['...',
                     '@.@'])
    start_locations = ((0, 0), (0, 2))
    goal_locations = ((0, 2), (0, 0))

    # These parameters are for making sure that the solver avoids collision regardless of reward efficiency
    env = MapfEnv(grid, 2, start_locations, goal_locations, 0.2, -0.001, 0, -1, optimization_criteria)

    train_info = {}
    start = time.time()
    policy = solver_describer.func(env, train_info)
    solve_time = round(time.time() - start, 2)

    # Assert no conflict is possible
    interesting_locations = ((1, 1), (0, 1))
    interesting_state = env.locations_to_state(interesting_locations)
    stay_up = vector_action_to_integer((STAY, UP))
    down_up = vector_action_to_integer((DOWN, UP))
    expected_possible_actions = [stay_up, down_up]

    if policy.act(interesting_state) not in expected_possible_actions:
        return RESULT_DANGEROUS_ACTION

        # Check the policy performance

    eval_info = evaluate_policy(policy, eval_n_episodes, eval_max_steps)
    extra_info = solver_describer.extra_info(train_info)

    print_status(env_name, eval_info['MDR'], solve_time, solver_describer.short_description, eval_info['success_rate'],
                 extra_info)

    if eval_info['clashed']:
        return RESULT_CLASHED

    if eval_info['success_rate'] == 0:
        return RESULT_NOT_CONVERGED

    return RESULT_OK


def main():
    max_env_lvl = max(lvl_to_env.keys())

    n_items = len(list(generate_solver_env_combinations(max_env_lvl))) + len(list(generate_all_solvers()))
    print(f'running {n_items} items')
    bad_results = []

    # Corridor switch
    prev_env_name = None
    for solver_describer, optimization_criteria in generate_all_solvers():
        # Calculate the env_name
        if optimization_criteria == OptimizationCriteria.Makespan:
            env_name = f'corridor_switch_makespan'
        if optimization_criteria == OptimizationCriteria.SoC:
            env_name = f'corridor_switch_soc'
        # Just nicer to view
        if prev_env_name != env_name:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f'\n{now_str} env:{env_name}')
        prev_env_name = env_name

        result = test_corridor_switch_no_clash_possible(solver_describer, optimization_criteria)
        if result != RESULT_OK:
            bad_results.append((solver_describer.short_description, env_name, result))
    print('')

    # All other envs
    prev_env_name = None
    for env_func, env_name, solver_describer, optimization_criteria in generate_solver_env_combinations(max_env_lvl):
        # Just nicer to view
        if prev_env_name != env_name:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f'\n{now_str} env:{env_name}')
        prev_env_name = env_name

        result = benchmark_solver_on_env(env_func, env_name, solver_describer, optimization_criteria)
        if result != RESULT_OK:
            bad_results.append((solver_describer.short_description, env_name, result))
    print('')

    if len(bad_results) != 0:
        print('The errors are')
        for i, (solver_name, env_name, bad_result) in enumerate(bad_results):
            print(f'{i}. {solver_name}, {env_name}, {bad_result}')

        exit(1)


if __name__ == '__main__':
    main()
