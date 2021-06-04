import time
import stopit
import pandas as pd
import unittest

from available_solvers import *
from solvers.utils import evaluate_policy

pd.options.display.max_columns = 20

EXCLUDED_SOLVERS = [
    rtdp_stop_no_improvement_determinsitic_heuristic_describer,
]

WEAK_SOLVERS = [
    value_iteration_describer,
    policy_iteration_describer,
]

STRONG_SOLVERS = [
    rtdp_stop_no_improvement_min_heuristic_describer,
    id_rtdp_describer,
    ma_rtdp_min_describer,
    fixed_iter_rtdp_min_describer,
    id_ma_rtdp_min_pvi_describer,

]

EXPERIMENT_SOLVERS = [
    long_ma_rtdp_min_pvi_describer,
    long_id_ma_rtdp_min_pvi_describer,
    long_id_rtdp_min_pvi_describer,

    long_ma_rtdp_sum_pvi_describer,
    long_id_ma_rtdp_sum_pvi_describer,
    long_id_rtdp_sum_pvi_describer,

    long_ma_rtdp_min_dijkstra_describer,
    long_id_ma_rtdp_min_dijkstra_describer,
    long_id_rtdp_min_dijkstra_describer,

    long_ma_rtdp_sum_dijkstra_describer,
    long_id_ma_rtdp_sum_dijkstra_describer,
    long_id_rtdp_sum_dijkstra_describer
]

SINGLE_SCENARIO_TIMEOUT = 300  # seconds

# Set option for pandas to print only full lines
pd.set_option('display.expand_frame_repr', False)


def timeout_handler(signum, frame):
    raise TimeoutError()


def benchmark_planners_on_env(env, env_str, solver_describers):
    results_df = pd.DataFrame(columns=[
        'env',
        'solver',
        'time',
        'avg_reward',
        'clashed'
    ])

    for solver_describer in solver_describers:
        solver_str = solver_describer.short_description
        solve_func = solver_describer.func

        # Fill default values in case of a timeout (we will not be able to evaluate the policy)
        solved = True
        reward, clashed = -1000, False

        # Prepare for running
        print(f'Running {solver_str} on {env_str}')

        # Run with time limit
        with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
            try:
                start = time.time()
                info = {}
                policy = solve_func(env, info)
            except stopit.utils.TimeoutException:
                print(f'{solver_str} on {env_str} got timeout')
                solved = False

        # Evaluate policy is solved
        if solved:
            print(f'evaluating policy calculated by {solver_str} on {env_str}')
            reward, clashed, _ = evaluate_policy(policy, 1000, 1000)

        # Measure time
        total_time = time.time() - start

        # Collect results
        row = {
            'env': env_str,
            'solver': solver_str,
            'time': total_time,
            'avg_reward': reward,
            'clashed': clashed
        }
        row.update(dict(solver_describer.extra_info(info)._asdict()))

        # Insert new row to results data frame
        results_df = results_df.append(row, ignore_index=True)

    # print the result
    print(f'-----{env_str}------')
    print(results_df)

    return results_df


def run_all(kwargs_list):
    df = benchmark_planners_on_env(**kwargs_list[0])

    # more than 1 always
    for kwargs in kwargs_list[1:]:
        df = df.append(benchmark_planners_on_env(**kwargs))

    print(df)

    return df
