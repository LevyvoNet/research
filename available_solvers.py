from collections import namedtuple
from functools import partial

from solvers import (value_iteration,
                     stop_when_no_improvement_between_batches_rtdp,
                     fixed_iterations_count_rtdp,
                     ma_rtdp,
                     policy_iteration,
                     id)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          local_views_prioritized_value_iteration_sum_heuristic,
                          deterministic_relaxation_prioritized_value_iteration_heuristic)

SolverDescriber = namedtuple('SolverDescriber', [
    'description',
    'func',
    'extra_info',
    'short_description',
])

SolverExtraInfo = namedtuple('SolverExtraInfo', [
    'n_conflicts',
    'conflict_detection_time',
    'solver_init_time',
    'total_evaluation_time'
])


def default_extra_info(info):
    return SolverExtraInfo('-', '-', '-', '-')


def ma_rtdp_extra_info(info):
    extra_info = default_extra_info(info)
    extra_info_dict = dict(extra_info._asdict())

    # Set initialization time
    extra_info_dict['solver_init_time'] = info['initialization_time']

    # Set evaluation time
    extra_info_dict['total_evaluation_time'] = info['total_evaluation_time']

    return SolverExtraInfo(**extra_info_dict)


def id_extra_info(info):
    extra_info = default_extra_info(info)
    extra_info_dict = dict(extra_info._asdict())

    # Number of found conflicts
    extra_info_dict['n_conflicts'] = len(info['iterations']) - 1

    # Total time for conflicts detection,
    # In case of all agents merged, the last iteration might not have a detect_conflict_time and therefore the 'get'.
    extra_info_dict['conflict_detection_time'] = sum([info['iterations'][i].get('detect_conflict_time', 0)
                                                      for i in range(len(info['iterations']))])

    # This time mostly matters for heuristics calculation (on RTDP for example)
    extra_info_dict['solver_init_time'] = sum(
        [sum([info['iterations'][j]['joint_policy'][key]['initialization_time']
              for key in info['iterations'][j]['joint_policy'].keys() if all([key.startswith('['), key.endswith(']')])])
         for j in range(len(info['iterations']))])

    return SolverExtraInfo(**extra_info_dict)


local_min_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_min_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
    extra_info=default_extra_info,
    short_description='pvi'
)

local_sum_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_sum_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
    extra_info=default_extra_info,
    short_description='sum_pvi'
)

deterministic_relaxation_pvi_heuristic_describer = SolverDescriber(
    description="deterministic_pvi_heuristic(gamma=1.0)",
    func=deterministic_relaxation_prioritized_value_iteration_heuristic,
    extra_info=default_extra_info,
    short_description='deter_pvi'
)

value_iteration_describer = SolverDescriber(
    description='value_iteration(gamma=1.0)',
    func=partial(value_iteration, 1.0),
    extra_info=default_extra_info,
    short_description='vi'
)

policy_iteration_describer = SolverDescriber(
    description='policy_iteration(gamma=1.0)',
    func=partial(policy_iteration, 1.0),
    extra_info=default_extra_info,
    short_description='pi'
)

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
    extra_info=default_extra_info,
    short_description='rtdp'
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
    extra_info=default_extra_info,
    short_description='long_rtdp'
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
    extra_info=default_extra_info,
    short_description='sum_rtdp'
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
    extra_info=default_extra_info,
    short_description='long_sum_rtdp'
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
    extra_info=default_extra_info,
    short_description='deter_rtdp'
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
    extra_info=default_extra_info,
    short_description='long_deter_rtdp'
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
    extra_info=ma_rtdp_extra_info,
    short_description='sum_ma_rtdp'
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
    extra_info=ma_rtdp_extra_info,
    short_description='long_sum_ma_rtdp'
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
    extra_info=ma_rtdp_extra_info,
    short_description='ma_rtdp'
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
    extra_info=ma_rtdp_extra_info,
    short_description='long_ma_rtdp'
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
    extra_info=default_extra_info,
    short_description='fixed_rtdp'
)

id_rtdp_describer = SolverDescriber(
    description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, rtdp_stop_no_improvement_min_heuristic_describer.func),
    extra_info=id_extra_info,
    short_description='id_rtdp'
)

long_id_rtdp_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, long_rtdp_stop_no_improvement_min_heuristic_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_rtdp'
)

id_ma_rtdp_describer = SolverDescriber(
    description=f'ID({ma_rtdp_min_describer.description})',
    func=partial(id, ma_rtdp_min_describer.func),
    extra_info=id_extra_info,
    short_description='id_ma_rtdp'
)

long_id_ma_rtdp_describer = SolverDescriber(
    description=f'ID({long_ma_rtdp_min_describer.description})',
    func=partial(id, long_ma_rtdp_min_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_ma_rtdp'
)
