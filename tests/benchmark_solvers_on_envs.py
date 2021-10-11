import datetime
import math
import os
import sys
import time
from typing import Callable

import stopit

from available_solvers import *
from gym_mapf.envs.grid import MapfGrid
from gym_mapf.envs.mapf_env import (MapfEnv,
                                    OptimizationCriteria)
from gym_mapf.envs.utils import create_mapf_env
from solvers.utils import evaluate_policy

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


def sanity_independent(room_size, n_agents, optimization_criteria):
    n_rooms = n_agents
    return create_mapf_env(f'sanity-{n_rooms}-{room_size}', None, n_agents, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_3_agents_room_size_8_independent(optimization_criteria):
    return create_mapf_env('sanity-3-8', None, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_2_32_3_agents(optimization_criteria):
    return create_mapf_env('sanity-2-32', None, 3, 0.2, -1000, 0, -1, optimization_criteria)


def sanity_general(n_rooms, room_size, n_agents, optimization_criteria):
    return create_mapf_env(f'sanity-{n_rooms}-{room_size}', None, n_agents, 0.2, -1000, 0, -1, optimization_criteria)


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
        # id_ma_rtdp_pvi_min_describer,
        # id_ma_rtdp_pvi_sum_describer,
        # id_rtdp_pvi_min_describer,
        # id_rtdp_pvi_sum_describer,
    ],
    2: [
        # long_rtdp_stop_no_improvement_sum_heuristic_describer,
        # long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer,
        # long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer,
        # long_ma_rtdp_pvi_min_describer,
        # long_ma_rtdp_min_dijkstra_describer,
        # long_id_ma_rtdp_min_dijkstra_describer,
        # long_id_ma_rtdp_min_pvi_describer,
        # long_id_ma_rtdp_min_rtdp_dijkstra_describer,
        # long_ma_rtdp_min_rtdp_dijkstra_describer,
        # long_ma_rtdp_pvi_sum_describer,
        # long_ma_rtdp_sum_dijkstra_describer,
        # long_ma_rtdp_sum_rtdp_dijkstra_describer,
        # long_id_rtdp_sum_pvi_describer,
        # long_id_ma_rtdp_sum_pvi_describer,
    ],
    3: [
        # long_id_ma_rtdp_sum_dijkstra_describer,
        # long_id_ma_rtdp_sum_rtdp_dijkstra_describer
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
        (sanity_3_agents_room_size_8_independent, 'sanity_3_agents_independent_stochastic'),

    ],
    2: [
        (partial(room_32_32_4_2_agents, 13, 0), 'room-32-32-4_scen_13_2_agents_1_conflict_deterministic'),
        (partial(room_32_32_4_2_agents, 13, 0.2), 'room-32-32-4_scen_13_2_agents_1_conflict_stochastic'),
        (partial(sanity_independent, 8, 8), 'sanity-independent-8X8-8-agents'),
        (partial(sanity_independent, 8, 16), 'sanity-independent-16X16-8-agents'),
        (partial(sanity_independent, 8, 32), 'sanity-independent-32X32-8-agents'),
    ],
    3: [
        (partial(room_32_32_4_2_agents, 13, 0.2), 'room-32-32-4_scen_13_4_agents_stochastic'),
        (sanity_2_32_3_agents, 'conflict_between_pair_and_single_large_map'),
        (partial(sanity_independent, 16, 8), 'sanity-independent-8X8-16-agents'),
        (partial(sanity_independent, 32, 8), 'sanity-independent-8X8-32-agents'),
        (partial(sanity_independent, 16, 16), 'sanity-independent-16X16-16-agents'),
        (partial(sanity_independent, 32, 16), 'sanity-independent-16X16-32-agents'),
        (partial(sanity_independent, 16, 32), 'sanity-independent-32X32-16-agents'),
        (partial(sanity_independent, 32, 32), 'sanity-independent-32X32-32-agents'),
        (partial(sanity_general, 8, 8, 16), 'sanity-8-rooms-8X8-16-agents'),
        (partial(sanity_general, 8, 16, 16), 'sanity-8-rooms-16X16-16-agents'),
        (partial(sanity_general, 8, 32, 16), 'sanity-8-rooms-32X32-16-agents'),

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

    return all_makespan
    return all_makespan + all_soc


TEST_DATA = generate_solver_env_combinations(max(lvl_to_env.keys()))


def print_status(reward, solve_time, solver_description, success_rate, extra_info: SolverExtraInfo = None):
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
                print_status(-math.inf, 'timeout', solver_describer.short_description, 0, extra_info)
                return RESULT_TIMEOUT

        solve_time = round(time.time() - start, 2)

        eval_info = evaluate_policy(policy, eval_n_episodes, eval_max_steps)
        extra_info = solver_describer.extra_info(train_info)

        print_status(eval_info['MDR'],
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


def main():
    max_env_lvl = max(lvl_to_env.keys())

    n_items = len(list(generate_solver_env_combinations(max_env_lvl)))
    print(f'running {n_items} items')
    bad_results = []

    prev_env_name = None
    for env_func, env_name, solver_describer, optimization_criteria in generate_solver_env_combinations(max_env_lvl):
        # Just nicer to view
        if prev_env_name != env_name:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(f'\n{now_str} env:{env_name}')
        prev_env_name = env_name

        # This is a hack for not dealing with some memory leak somewhere inside benchmark_solver_on_env function.
        read_fd, write_fd = os.pipe()
        sys.stdout.flush()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            result = benchmark_solver_on_env(env_func, env_name, solver_describer, optimization_criteria)
            write_file = os.fdopen(write_fd, 'w')
            write_file.write(result)
            write_file.close()
            exit(0)
        else:
            os.close(write_fd)
            os.waitpid(pid, 0)
            read_file = os.fdopen(read_fd, 'r')
            result = read_file.read()
            read_file.close()

        # result = benchmark_solver_on_env(env_func, env_name, solver_describer, optimization_criteria)
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
