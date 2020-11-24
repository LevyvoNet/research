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
    'extra_info'
])

SolverExtraInfo = namedtuple('SolverExtraInfo', [
    'n_conflicts',
    'conflict_detection_time',
    'solver_init_time'
])


def default_extra_info(info):
    return SolverExtraInfo('-', '-', '-')


def id_extra_info(info):
    extra_info = default_extra_info(info)
    extra_info_dict = dict(extra_info._asdict())

    n_agents = len(info['iterations'][0]['joint_policy']) - 1

    # Number of found conflicts
    extra_info_dict['n_conflicts'] = len(info['iterations']) - 1

    # Total time for conflicts detection,
    # In case of all agents merged, the last iteration might not have a detect_conflict_time and therefore the 'get'.
    extra_info_dict['conflict_detection_time'] = sum([info['iterations'][i].get('detect_conflict_time', 0)
                                                      for i in range(len(info['iterations']))])

    # This time mostly matters for heuristics calculation (on RTDP for example)
    extra_info_dict['solver_init_time'] = sum(
        [sum([info['iterations'][j]['joint_policy'][f"[{i}]"]['initialization_time']
              for i in range(n_agents)])
         for j in range(len(info['iterations']))])

    return SolverExtraInfo(**extra_info_dict)


local_min_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_min_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
    extra_info=default_extra_info
)

local_sum_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_sum_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
    extra_info=default_extra_info
)

deterministic_relaxation_pvi_heuristic_describer = SolverDescriber(
    description="determistic_pvi_heuristic(gamma=1.0)",
    func=deterministic_relaxation_prioritized_value_iteration_heuristic,
    extra_info=default_extra_info
)

value_iteration_describer = SolverDescriber(
    description='value_iteration(gamma=1.0)',
    func=partial(value_iteration, 1.0),
    extra_info=default_extra_info
)

policy_iteration_describer = SolverDescriber(
    description='policy_iteration(gamma=1.0)',
    func=partial(policy_iteration, 1.0),
    extra_info=default_extra_info)

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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
)
id_rtdp_describer = SolverDescriber(
    description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, rtdp_stop_no_improvement_min_heuristic_describer.func),
    extra_info=id_extra_info
)

long_id_rtdp_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, long_rtdp_stop_no_improvement_min_heuristic_describer.func),
    extra_info=id_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
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
    extra_info=default_extra_info
)
