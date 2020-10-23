import time
import unittest
import pandas as pd
import stopit

from research.solvers.utils import evaluate_policy
from research.available_solvers import *

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


class AbstractPerformanceTest(unittest.TestCase):
    solver_describers = []
    env_str = None
    env = None

    def test_planners_performance(self):
        if self.env is None:
            raise unittest.SkipTest("This is an abstract test case")

        results_df = pd.DataFrame(columns=[
            'solver',
            'time',
            'avg_reward',
            'clashed'
        ])

        for solver_describer in self.solver_describers:
            solver_str = solver_describer.description
            solve_func = solver_describer.func

            # Assume solved
            solved = True
            reward, clashed = -1000, False

            print(f'running {solver_str} on {self.env_str}')
            # Run with time limit
            with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
                try:
                    start = time.time()
                    info = {}
                    policy = solve_func(self.env, info)
                except stopit.utils.TimeoutException:
                    solved = False

            # Evaulate policy is solved
            if solved:
                reward, clashed = evaluate_policy(policy, 100, 1000)

            # Measure time
            total_time = time.time() - start

            # Collect results
            results_df = results_df.append({
                'solver': solver_str,
                'time': total_time,
                'avg_reward': reward,
                'clashed': clashed
            }
                , ignore_index=True)
        # print the result
        print(f'-----{self.env_str}------')
        print(results_df)
