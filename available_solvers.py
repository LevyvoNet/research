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
                          deterministic_relaxation_prioritized_value_iteration_heuristic,
                          dijkstra_min_heuristic,
                          dijkstra_sum_heuristic,
                          solution_heuristic_min,
                          solution_heuristic_sum,
                          fixed_iterations_rtdp_merge,
                          stop_when_no_improvement_between_batches_rtdp_merge)

from solvers.ma_rtdp import ma_rtdp_merge

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
    extra_info_dict['solver_init_time'] = info.get('initialization_time', '*')

    # Set evaluation time
    extra_info_dict['total_evaluation_time'] = info.get('total_evaluation_time', '*')

    return SolverExtraInfo(**extra_info_dict)


def id_extra_info(info):
    extra_info = default_extra_info(info)
    extra_info_dict = dict(extra_info._asdict())

    # Number of found conflicts
    try:
        extra_info_dict['n_conflicts'] = len(info['iterations']) - 1

        # Total time for conflicts detection,
        # In case of all agents merged, the last iteration might not have a detect_conflict_time and therefore the 'get'.
        extra_info_dict['conflict_detection_time'] = sum([info['iterations'][i].get('detect_conflict_time', 0)
                                                          for i in range(len(info['iterations']))])

        # This time mostly matters for heuristics calculation (on RTDP for example)
        extra_info_dict['solver_init_time'] = sum(
            [sum([info['iterations'][j]['joint_policy'][key]['initialization_time']
                  for key in info['iterations'][j]['joint_policy'].keys() if all([key.startswith('['),
                                                                                  key.endswith(']')])
                  ])
             for j in range(len(info['iterations']))])
    except:
        print('something terrible happened')

    return SolverExtraInfo(**extra_info_dict)


# Heuristics ##########################################################################################################

local_min_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_min_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_min_heuristic, 1.0),
    extra_info=default_extra_info,
    short_description='pvi'
)

# The sum heuristic is not admissible, do not use it for now
local_sum_pvi_heuristic_describer = SolverDescriber(
    description='local_view_pvi_sum_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0),
    extra_info=default_extra_info,
    short_description='sum_pvi'
)

local_min_dijkstra_heuristic_describer = SolverDescriber(
    description='min_dijkstra_heuristic',
    func=dijkstra_min_heuristic,
    extra_info=default_extra_info,
    short_description='dijkstra'
)

local_sum_dijkstra_heuristic_describer = SolverDescriber(
    description='sum_dijkstra_heuristic',
    func=dijkstra_sum_heuristic,
    extra_info=default_extra_info,
    short_description='dijkstra'
)

deterministic_relaxation_pvi_heuristic_describer = SolverDescriber(
    description="deterministic_pvi_heuristic(gamma=1.0)",
    func=deterministic_relaxation_prioritized_value_iteration_heuristic,
    extra_info=default_extra_info,
    short_description='deter_pvi'
)

solution_heuristic_min_describer = SolverDescriber(
    description='sol_min_heuristic',
    func=solution_heuristic_min,
    extra_info=default_extra_info,
    short_description='sol_min_heuristic'
)

solution_heuristic_sum_describer = SolverDescriber(
    description='sol_sum_heuristic',
    func=solution_heuristic_sum,
    extra_info=default_extra_info,
    short_description='sol_sum_heuristic'
)

# ID Low Level Mergers ################################################################################################

DEFAULT_LOW_LEVEL_MERGER = None

rtdp_stop_no_improvement_min_merger_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp_merge('
                f'{solution_heuristic_min_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
                 solution_heuristic_min_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=default_extra_info,
    short_description='rtdp_min_merger'
)

long_rtdp_stop_no_improvement_min_merger_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp_merge('
                f'{solution_heuristic_min_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
                 solution_heuristic_min_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_rtdp_min_merger'
)

rtdp_stop_no_improvement_sum_merger_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp_merge('
                f'{solution_heuristic_sum_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
                 solution_heuristic_sum_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=default_extra_info,
    short_description='rtdp_sum_merger'
)

long_rtdp_stop_no_improvement_sum_merger_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp_merge('
                f'{solution_heuristic_sum_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
                 solution_heuristic_sum_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_rtdp_sum_merger'
)

ma_rtdp_min_merger_describer = SolverDescriber(
    description=f'ma_rtdp_merge('
                f'{solution_heuristic_min_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(ma_rtdp_merge,
                 solution_heuristic_min_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=default_extra_info,
    short_description='ma_rtdp_min_merger'
)

long_ma_rtdp_min_merger_describer = SolverDescriber(
    description=f'ma_rtdp_merge('
                f'{solution_heuristic_min_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(ma_rtdp_merge,
                 solution_heuristic_min_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_ma_rtdp_min_merger'
)

ma_rtdp_sum_merger_describer = SolverDescriber(
    description=f'ma_rtdp_merge('
                f'{solution_heuristic_sum_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(ma_rtdp_merge,
                 solution_heuristic_sum_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=default_extra_info,
    short_description='ma_rtdp_sum_merger'
)

long_ma_rtdp_sum_merger_describer = SolverDescriber(
    description=f'ma_rtdp_merge('
                f'{solution_heuristic_sum_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(ma_rtdp_merge,
                 solution_heuristic_sum_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_ma_rtdp_sum_merger'
)

# Solvers #############################################################################################################

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
    short_description='rtdp_pvi_min'
)

long_rtdp_stop_no_improvement_min_pvi_heuristic_describer = SolverDescriber(
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
    short_description='long_rtdp_pvi_min'
)

long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer = SolverDescriber(
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
    short_description='long_rtdp_pvi_sum'
)

long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_min_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_min_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_rtdp_dijkstra_min'
)

long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer = SolverDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_sum_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_sum_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=default_extra_info,
    short_description='long_rtdp_dijkstra_sum'
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
    short_description='long_sum_pvi_rtdp'
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
                 10000),
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
    short_description='ma_rtdp_pvi_sum'
)

ma_rtdp_min_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500,',
    func=partial(ma_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=ma_rtdp_extra_info,
    short_description='ma_rtdp'
)

ma_rtdp_dijkstra_min_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500,',
    func=partial(ma_rtdp,
                 local_min_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=ma_rtdp_extra_info,
    short_description='ma_rtdp_dijkstra_min'
)

ma_rtdp_dijkstra_sum_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_sum_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500,',
    func=partial(ma_rtdp,
                 local_sum_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 500),
    extra_info=ma_rtdp_extra_info,
    short_description='ma_rtdp_dijkstra_sum'
)

long_ma_rtdp_min_pvi_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=ma_rtdp_extra_info,
    short_description='long_ma_rtdp_pvi_min'
)

long_ma_rtdp_sum_pvi_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_sum_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=ma_rtdp_extra_info,
    short_description='long_ma_rtdp_pvi_sum'
)

long_ma_rtdp_min_dijkstra_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_min_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_min_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=ma_rtdp_extra_info,
    short_description='long_ma_rtdp_dijkstra_min'
)

long_ma_rtdp_sum_dijkstra_describer = SolverDescriber(
    description=f'ma_rtdp('
                f'{local_sum_dijkstra_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_sum_dijkstra_heuristic_describer.func,
                 1.0,
                 100,
                 10000),
    extra_info=ma_rtdp_extra_info,
    short_description='long_ma_rtdp_dijkstra_sum'
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

id_vi_describer = SolverDescriber(
    description=f'ID({value_iteration_describer.description})',
    func=partial(id,
                 value_iteration_describer.func,
                 DEFAULT_LOW_LEVEL_MERGER),
    extra_info=id_extra_info,
    short_description='id_vi'
)

id_rtdp_describer = SolverDescriber(
    description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id,
                 rtdp_stop_no_improvement_min_heuristic_describer.func,
                 rtdp_stop_no_improvement_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='id_rtdp_pvi_min'
)

long_id_rtdp_min_pvi_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_min_pvi_heuristic_describer.description})',
    func=partial(id,
                 long_rtdp_stop_no_improvement_min_pvi_heuristic_describer.func,
                 long_rtdp_stop_no_improvement_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_rtdp_pvi_min'
)

long_id_rtdp_sum_pvi_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer.description})',
    func=partial(id,
                 long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer.func,
                 long_rtdp_stop_no_improvement_sum_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_rtdp_pvi_sum'
)

long_id_rtdp_min_dijkstra_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer.description})',
    func=partial(id,
                 long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer.func,
                 long_rtdp_stop_no_improvement_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_rtdp_dijkstra_min'
)

long_id_rtdp_sum_dijkstra_describer = SolverDescriber(
    description=f'ID({long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer.description})',
    func=partial(id,
                 long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer.func,
                 long_rtdp_stop_no_improvement_sum_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_rtdp_dijkstra_sum'
)

id_ma_rtdp_min_pvi_describer = SolverDescriber(
    description=f'ID({ma_rtdp_min_describer.description})',
    func=partial(id,
                 ma_rtdp_min_describer.func,
                 ma_rtdp_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='id_ma_rtdp'
)

long_id_ma_rtdp_min_pvi_describer = SolverDescriber(
    description=f'ID({long_ma_rtdp_min_pvi_describer.description})',
    func=partial(id,
                 long_ma_rtdp_min_pvi_describer.func,
                 long_ma_rtdp_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_ma_rtdp'
)

long_id_ma_rtdp_sum_pvi_describer = SolverDescriber(
    description=f'ID({long_ma_rtdp_sum_pvi_describer.description})',
    func=partial(id,
                 long_ma_rtdp_sum_pvi_describer.func,
                 long_ma_rtdp_sum_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_ma_rtdp_pvi_sum'
)

long_id_ma_rtdp_min_dijkstra_describer = SolverDescriber(
    description=f'ID({long_ma_rtdp_min_dijkstra_describer.description})',
    func=partial(id,
                 long_ma_rtdp_min_dijkstra_describer.func,
                 long_ma_rtdp_min_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_ma_rtdp_dijkstra'
)

long_id_ma_rtdp_sum_dijkstra_describer = SolverDescriber(
    description=f'ID({long_ma_rtdp_sum_dijkstra_describer.description})',
    func=partial(id,
                 long_ma_rtdp_sum_dijkstra_describer.func,
                 long_ma_rtdp_sum_merger_describer.func),
    extra_info=id_extra_info,
    short_description='long_id_ma_rtdp_dijkstra'
)
