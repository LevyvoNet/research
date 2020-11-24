import time

import pandas as pd
import stopit

from available_solvers import *
from solvers.utils import evaluate_policy

EXCLUDED_SOLVERS = [
    rtdp_stop_no_improvement_determinsitic_heuristic_describer,
]

WEAK_SOLVERS = [
    value_iteration_describer,
    policy_iteration_describer
]
STRONG_SOLVERS = [
    rtdp_stop_no_improvement_min_heuristic_describer,
    rtdp_stop_no_improvement_sum_heuristic_describer,
    id_rtdp_describer,
    ma_rtdp_sum_describer,
    ma_rtdp_min_describer,
    fixed_iter_rtdp_min_describer

]

EXPERIMENT_SOLVERS = [
    long_id_rtdp_describer,
    long_rtdp_stop_no_improvement_min_heuristic_describer,
    long_ma_rtdp_min_describer,
    long_ma_rtdp_sum_describer,
]

SINGLE_SCENARIO_TIMEOUT = 300  # seconds


def benchmark_planners_on_env(env, env_str, solver_describers):
    results_df = pd.DataFrame(columns=[
        'env',
        'solver',
        'time',
        'avg_reward',
        'clashed'
    ])

    for solver_describer in solver_describers:
        solver_str = solver_describer.description
        solve_func = solver_describer.func

        # Assume solved
        solved = True
        reward, clashed = -1000, False

        # Run with time limit
        with stopit.ThreadingTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
            print(f'Running {solver_str} on {env_str}')
            try:
                start = time.time()
                info = {}
                policy = solve_func(env, info)
            except stopit.utils.TimeoutException:
                solved = False

        # Evaluate policy is solved
        if solved:
            reward, clashed = evaluate_policy(policy, 100, 1000)

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
