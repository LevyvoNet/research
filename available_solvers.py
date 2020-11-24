from collections import namedtuple

from solvers import (value_iteration,
                     stop_when_no_improvement_between_batches_rtdp,
                     fixed_iterations_count_rtdp,
                     ma_rtdp,
                     policy_iteration,
                     id)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          local_views_prioritized_value_iteration_sum_heuristic,
                          deterministic_relaxation_prioritized_value_iteration_heuristic)
from functools import partial

SolverDescriber = namedtuple('SolverDescriber', [
    'description',
    'func',
    'info_df'
])

InfoDF = namedtuple('InfoDf', [
    'n_conflicts',
    'conflict_detection_time',
    'solver_init_time'
])


def default_info_df(info):
    info_df = InfoDF('-', '-', '-')
    info_df_dict = dict(info_df._asdict())

    return info_df_dict


def id_info_df(info):
    info_df = default_info_df(info)
    info_df_dict = dict(info_df._asdict())

    n_agents = len(info['iterations'][0]['joint_policy']) - 1

    # Number of found conflicts
    info_df_dict['n_conflicts'] = len(info['iterations']) - 1

    # Total time for conflicts detection
    info_df_dict['conflict_detection_time'] = sum([info['iterations'][i]['detect_conflict_time']
                                                   for i in range(len(info['iterations']))])

    # This time mostly matters for heuristics calculation (on RTDP for example)
    info_df_dict['solver_init_time'] = sum([sum([info['iterations'][j]['joint_policy'][f"[{i}]"]['initialization_time']
                                                 for i in range(n_agents)])
                                            for j in range(len(info['iterations']))])

    return info_df_dict


local_min_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_min_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
    info_df=default_info_df
)

local_sum_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_sum_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
    info_df=default_info_df
)

deterministic_relaxation_pvi_heuristic_describer = SolverDescriber(
    description="determistic_pvi_heuristic(gamma=1.0)",
    func=deterministic_relaxation_prioritized_value_iteration_heuristic,
    info_df=default_info_df
)

value_iteration_describer = SolverDescriber(
    description='value_iteration(gamma=1.0)',
    func=partial(value_iteration, 1.0),
    info_df=default_info_df
)

policy_iteration_describer = SolverDescriber(
    description='policy_iteration(gamma=1.0)',
    func=partial(policy_iteration, 1.0),
    info_df=default_info_df)

rtdp_stop_no_improvement_min_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    info_df=default_info_df
)

long_rtdp_stop_no_improvement_min_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    info_df=default_info_df
)

rtdp_stop_no_improvement_sum_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_sum_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    info_df=default_info_df
)

long_rtdp_stop_no_improvement_sum_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_sum_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    info_df=default_info_df
)

rtdp_stop_no_improvement_determinsitic_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{deterministic_relaxation_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 deterministic_relaxation_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    info_df=default_info_df
)

long_rtdp_stop_no_improvement_determinsitic_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{deterministic_relaxation_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 deterministic_relaxation_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 1000),
    info_df=default_info_df
)
id_rtdp_describer = SolverDescriber(
    description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, rtdp_stop_no_improvement_min_heuristic_describer.func),
    info_df=id_info_df
)

long_id_rtdp_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, long_rtdp_stop_no_improvement_min_heuristic_describer.func),
    info_df=id_info_df
)

ma_rtdp_sum_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_sum_pvi_heuristic_describer.description,}'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    info_df=default_info_df
)

long_ma_rtdp_sum_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_sum_pvi_heuristic_describer.description,}'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    info_df=default_info_df
)

ma_rtdp_min_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500,',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    info_df=default_info_df
)

long_ma_rtdp_min_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    info_df=default_info_df
)

fixed_iter_rtdp_min_describer = SolverDescriber(
    description=f'fixed_iters_rtdp('
                f'{local_min_pvi_heuristic_describer.description}'
                f'gamma=1.0'
                f'iters=400',
    func=partial(fixed_iterations_count_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 400),
    info_df=default_info_df
)
