from collections import namedtuple

from research.solvers import (value_iteration,
                              stop_when_no_improvement_between_batches_rtdp,
                              fixed_iterations_count_rtdp,
                              ma_rtdp,
                              policy_iteration,
                              id)
from research.solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                                   local_views_prioritized_value_iteration_sum_heuristic,
                                   deterministic_relaxation_prioritized_value_iteration_heuristic)
from functools import partial

FunctionDescriber = namedtuple('Solver', [
    'description',
    'func'
])

local_min_pvi_heuristic_describer = FunctionDescriber(
    description='local_view_pvi_min_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_min_heuristic, 1.0))

local_sum_pvi_heuristic_describer = FunctionDescriber(
    description='local_view_pvi_sum_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0))

deterministic_relaxation_pvi_heuristic_describer = FunctionDescriber(
    description="determistic_pvi_heuristic(gamma=1.0)",
    func=deterministic_relaxation_prioritized_value_iteration_heuristic
)

value_iteration_describer = FunctionDescriber(
    description='value_iteration(gamma=1.0)',
    func=partial(value_iteration, 1.0))

policy_iteration_describer = FunctionDescriber(
    description='policy_iteration(gamma=1.0)',
    func=partial(policy_iteration, 1.0))

rtdp_stop_no_improvement_min_heuristic_describer = FunctionDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500)
)

long_rtdp_stop_no_improvement_min_heuristic_describer = FunctionDescriber(
    description=f'long stop_no_improvement_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000)
)

rtdp_stop_no_improvement_sum_heuristic_describer = FunctionDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_sum_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500)
)

long_rtdp_stop_no_improvement_sum_heuristic_describer = FunctionDescriber(
    description=f'long stop_no_improvement_rtdp('
                f'{local_sum_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000)
)

rtdp_stop_no_improvement_determinsitic_heuristic_describer = FunctionDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{deterministic_relaxation_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 deterministic_relaxation_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500)
)

long_rtdp_stop_no_improvement_determinsitic_heuristic_describer = FunctionDescriber(
    description=f'long stop_no_improvement_rtdp('
                f'{deterministic_relaxation_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 deterministic_relaxation_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 1000)
)
id_rtdp_describer = FunctionDescriber(
    description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, rtdp_stop_no_improvement_min_heuristic_describer.func)
)

long_id_rtdp_describer = FunctionDescriber(
    description=f'long ID({long_rtdp_stop_no_improvement_min_heuristic_describer.description})',
    func=partial(id, long_rtdp_stop_no_improvement_min_heuristic_describer.func)
)

ma_rtdp_sum_describer = FunctionDescriber(
    description=f'ma_rtdp('
                f'{local_sum_pvi_heuristic_describer.description,}'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500)
)

long_ma_rtdp_sum_describer = FunctionDescriber(
    description=f'long ma_rtdp('
                f'{local_sum_pvi_heuristic_describer.description,}'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000)
)

ma_rtdp_min_describer = FunctionDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=500,',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 500)
)

long_ma_rtdp_min_describer = FunctionDescriber(
    description=f'ma_rtdp('
                f'{local_min_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000,',
    func=partial(ma_rtdp,
                 local_sum_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000)
)

fixed_iter_rtdp_min_describer = FunctionDescriber(
    description=f'fixed_iters_rtdp('
                f'{local_min_pvi_heuristic_describer.description}'
                f'gamma=1.0'
                f'iters=400',
    func=partial(fixed_iterations_count_rtdp,
                 local_min_pvi_heuristic_describer.func,
                 1.0,
                 400)
)
