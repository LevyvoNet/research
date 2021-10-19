from collections import namedtuple
from functools import partial

from solvers import (ValueIterationPolicy,
                     PrioritizedValueIterationPolicy,
                     PolicyIterationPolicy,
                     RtdpPolicy,
                     MultiagentRtdpPolicy,
                     IdPolicy)
from solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                          local_views_prioritized_value_iteration_sum_heuristic,
                          deterministic_relaxation_prioritized_value_iteration_heuristic,
                          dijkstra_min_heuristic,
                          dijkstra_sum_heuristic,
                          solution_heuristic_min,
                          solution_heuristic_sum,
                          rtdp_dijkstra_sum_heuristic,
                          rtdp_dijkstra_min_heuristic,
                          fixed_iterations_rtdp_merge,
                          stop_when_no_improvement_between_batches_rtdp_merge, )

from solvers.ma_rtdp import ma_rtdp_merge

# Heuristics ##########################################################################################################
pvi_min_h = partial(local_views_prioritized_value_iteration_min_heuristic, 1.0)
pvi_sum_h = partial(local_views_prioritized_value_iteration_sum_heuristic, 1.0)
dijkstra_min_h = dijkstra_min_heuristic
dijkstra_sum_h = dijkstra_sum_heuristic
rtdp_dijkstra_min_h = partial(rtdp_dijkstra_min_heuristic, 1.0, 500)
rtdp_dijkstra_sum_h = partial(rtdp_dijkstra_sum_heuristic, 1.0, 500)
solution_heuristic_min_h = solution_heuristic_min
solution_heuristic_sum_h = solution_heuristic_sum

# ID Low Level Mergers ################################################################################################

DEFAULT_LOW_LEVEL_MERGER = None
rtdp_sum_merger = partial(stop_when_no_improvement_between_batches_rtdp_merge,
                          solution_heuristic_sum_h,
                          100,
                          10000)

ma_rtdp_sum_merger = partial(ma_rtdp_merge,
                             solution_heuristic_sum_h,
                             100,
                             10000)


# rtdp_stop_no_improvement_min_merger_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp_merge('
#                 f'{solution_heuristic_min_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
#                  solution_heuristic_min_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=default_extra_info,
#     short_description='rtdp_min_merger'
# )
#
# long_rtdp_stop_no_improvement_min_merger_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp_merge('
#                 f'{solution_heuristic_min_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
#                  solution_heuristic_min_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=default_extra_info,
#     short_description='long_rtdp_min_merger'
# )
#
# rtdp_stop_no_improvement_sum_merger_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp_merge('
#                 f'{solution_heuristic_sum_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
#                  solution_heuristic_sum_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=default_extra_info,
#     short_description='rtdp_sum_merger'
# )
#
# long_rtdp_stop_no_improvement_sum_merger_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp_merge('
#                 f'{solution_heuristic_sum_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp_merge,
#                  solution_heuristic_sum_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=default_extra_info,
#     short_description='long_rtdp_sum_merger'
# )
#
# ma_rtdp_min_merger_describer = SolverDescriber(
#     description=f'ma_rtdp_merge('
#                 f'{solution_heuristic_min_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(ma_rtdp_merge,
#                  solution_heuristic_min_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=default_extra_info,
#     short_description='ma_rtdp_min_merger'
# )
#
# long_ma_rtdp_min_merger_describer = SolverDescriber(
#     description=f'ma_rtdp_merge('
#                 f'{solution_heuristic_min_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(ma_rtdp_merge,
#                  solution_heuristic_min_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=default_extra_info,
#     short_description='long_ma_rtdp_min_merger'
# )
#
# ma_rtdp_sum_merger_describer = SolverDescriber(
#     description=f'ma_rtdp_merge('
#                 f'{solution_heuristic_sum_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(ma_rtdp_merge,
#                  solution_heuristic_sum_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=default_extra_info,
#     short_description='ma_rtdp_sum_merger'
# )
#
# long_ma_rtdp_sum_merger_describer = SolverDescriber(
#     description=f'ma_rtdp_merge('
#                 f'{solution_heuristic_sum_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(ma_rtdp_merge,
#                  solution_heuristic_sum_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=default_extra_info,
#     short_description='long_ma_rtdp_sum_merger'
# )

# Solvers #############################################################################################################
def vi_creator(env, gamma):
    return ValueIterationPolicy(env, gamma, 'vi')


def pvi_creator(env, gamma):
    PrioritizedValueIterationPolicy(env, gamma, 'pvi')


def pi_creator(env, gamma):
    PolicyIterationPolicy(env, gamma, 'pi')


def rtdp_pvi_sum_creator(env, gamma):
    return RtdpPolicy(env, gamma, pvi_sum_h, 100, 10000, 'rtdp_pvi_sum')


def rtdp_dijkstra_sum_creator(env, gamma):
    return RtdpPolicy(env, gamma, dijkstra_sum_h, 100, 10000, 'rtdp_dijkstra_sum')


def rtdp_rtdp_dijkstra_sum_creator(env, gamma):
    return RtdpPolicy(env, gamma, rtdp_dijkstra_sum_h, 100, 10000, 'rtdp_rtdp_dijkstra_sum')


def ma_rtdp_pvi_sum_creator(env, gamma):
    return MultiagentRtdpPolicy(env, gamma, pvi_sum_h, 100, 10000, 'ma_rtdp_pvi_sum')


def ma_rtdp_dijkstra_sum_creator(env, gamma):
    return MultiagentRtdpPolicy(env, gamma, dijkstra_sum_h, 100, 10000, 'ma_rtdp_dijkstra_sum')


def ma_rtdp_rtdp_dijkstra_sum_creator(env, gamma):
    return MultiagentRtdpPolicy(rtdp_dijkstra_sum_h, 100, 10000, 'ma_rtdp_rtdp_dijkstra_sum')


def id_vi_creator(env, gamma):
    return IdPolicy(env, gamma, vi_creator, None, 'id_vi')


def id_rtdp_dijkstra_sum_creator(env, gamma):
    return IdPolicy(env, gamma, rtdp_dijkstra_sum_creator, rtdp_sum_merger, 'id_rtdp_dijsktra_sum')


def id_ma_rtdp_dijkstra_sum_creator(env, gamma):
    return IdPolicy(env, gamma, ma_rtdp_dijkstra_sum_creator, ma_rtdp_sum_merger, 'id_ma_rtdp_dijsktra_sum')


def id_rtdp_pvi_sum_creator(env, gamma):
    return IdPolicy(env, gamma, rtdp_pvi_sum_creator, rtdp_sum_merger, 'id_rtdp_pvi_sum')


def id_ma_rtdp_pvi_sum_creator(env, gamma):
    return IdPolicy(env, gamma, ma_rtdp_pvi_sum_creator, ma_rtdp_sum_merger, 'id_ma_rtdp_pvi_sum')

# prioritized_value_iteration_describer = SolverDescriber(
#     description='prioritized_value_iteration(gamma=1.0)',
#     func=partial(prioritized_value_iteration, 1.0),
#     extra_info=default_extra_info,
#     short_description='pvi'
# )
#
# policy_iteration_describer = SolverDescriber(
#     description='policy_iteration(gamma=1.0)',
#     func=partial(policy_iteration, 1.0),
#     extra_info=default_extra_info,
#     short_description='pi'
# )
#
# rtdp_stop_no_improvement_min_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_min_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_min_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=rtdp_extra_info,
#     short_description='rtdp_pvi_min'
# )
#
# long_rtdp_stop_no_improvement_min_pvi_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_min_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_min_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_pvi_min'
# )
#
# long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_sum_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_sum_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_pvi_sum'
# )
#
# long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_min_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_min_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_dijkstra_min'
# )
#
# long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_sum_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_sum_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_dijkstra_sum'
# )
#
# rtdp_stop_no_improvement_sum_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_sum_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_sum_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=rtdp_extra_info,
#     short_description='rtdp_pvi_sum'
# )
#
# long_rtdp_stop_no_improvement_sum_heuristic_describer = SolverDescriber(
#     description=f'stop_no_improvement_rtdp('
#                 f'{local_sum_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_sum_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_pvi_sum'
# )
#
# long_rtdp_stop_no_improvement_sum_rtdp_dijkstra_heuristic_describer = SolverDescriber(
#     description=f'rtdp('
#                 f'{local_sum_rtdp_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_sum_rtdp_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_rtdp_dijkstra_sum'
# )
#
# long_rtdp_stop_no_improvement_min_rtdp_dijkstra_heuristic_describer = SolverDescriber(
#     description=f'rtdp('
#                 f'{local_min_rtdp_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000)',
#     func=partial(stop_when_no_improvement_between_batches_rtdp,
#                  local_min_rtdp_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=rtdp_extra_info,
#     short_description='long_rtdp_rtdp_dijkstra_min'
# )
#
# ma_rtdp_pvi_sum_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_sum_pvi_heuristic_describer.description,}'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000',
#     func=partial(ma_rtdp,
#                  local_sum_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='ma_rtdp_pvi_sum'
# )
#
# ma_rtdp_pvi_min_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_min_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000,',
#     func=partial(ma_rtdp,
#                  local_min_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='ma_rtdp_pvi_min'
# )
#
# ma_rtdp_dijkstra_min_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_min_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000,',
#     func=partial(ma_rtdp,
#                  local_min_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='ma_rtdp_dijkstra_min'
# )
#
# ma_rtdp_dijkstra_sum_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_sum_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=1000,',
#     func=partial(ma_rtdp,
#                  local_sum_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  1000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='ma_rtdp_dijkstra_sum'
# )
#
# long_ma_rtdp_pvi_min_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_min_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_min_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_pvi_min'
# )
#
# long_ma_rtdp_pvi_sum_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_sum_pvi_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_sum_pvi_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_pvi_sum'
# )
#
# long_ma_rtdp_min_dijkstra_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_min_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_min_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_dijkstra_min'
# )
#
# long_ma_rtdp_sum_dijkstra_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_sum_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_sum_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_dijkstra_sum'
# )
#
# long_ma_rtdp_sum_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_sum_rtdp_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_sum_rtdp_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_rtdp_dijkstra_sum'
# )
#
# long_ma_rtdp_min_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ma_rtdp('
#                 f'{local_min_rtdp_dijkstra_heuristic_describer.description},'
#                 f'gamma=1.0,'
#                 f'batch_size=100,'
#                 f'max_iters=10000,',
#     func=partial(ma_rtdp,
#                  local_min_rtdp_dijkstra_heuristic_describer.func,
#                  1.0,
#                  100,
#                  10000),
#     extra_info=ma_rtdp_extra_info,
#     short_description='long_ma_rtdp_rtdp_dijkstra_min'
# )
#
# fixed_iter_rtdp_min_describer = SolverDescriber(
#     description=f'fixed_iters_rtdp('
#                 f'{local_min_pvi_heuristic_describer.description}'
#                 f'gamma=1.0'
#                 f'iters=400',
#     func=partial(fixed_iterations_count_rtdp,
#                  local_min_pvi_heuristic_describer.func,
#                  1.0,
#                  400),
#     extra_info=default_extra_info,
#     short_description='fixed_rtdp'
# )
#
# id_vi_describer = SolverDescriber(
#     description=f'ID({value_iteration_describer.description})',
#     func=partial(id,
#                  value_iteration_describer.func,
#                  DEFAULT_LOW_LEVEL_MERGER),
#     extra_info=id_extra_info,
#     short_description='id_vi'
# )
#
# id_rtdp_pvi_min_describer = SolverDescriber(
#     description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
#     func=partial(id,
#                  rtdp_stop_no_improvement_min_heuristic_describer.func,
#                  rtdp_stop_no_improvement_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='id_rtdp_pvi_min'
# )
#
# id_rtdp_pvi_sum_describer = SolverDescriber(
#     description=f'ID({rtdp_stop_no_improvement_min_heuristic_describer.description})',
#     func=partial(id,
#                  rtdp_stop_no_improvement_sum_heuristic_describer.func,
#                  rtdp_stop_no_improvement_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='id_rtdp_pvi_sum'
# )
#
# long_id_rtdp_min_pvi_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_min_pvi_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_min_pvi_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_pvi_min'
# )
#
# long_id_rtdp_sum_pvi_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_sum_pvi_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_pvi_sum'
# )
#
# long_id_rtdp_min_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_min_dijkstra_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_dijkstra_min'
# )
#
# long_id_rtdp_sum_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_dijkstra_sum'
# )
#
# long_id_rtdp_sum_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_sum_rtdp_dijkstra_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_sum_rtdp_dijkstra_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_rtdp_dijkstra_sum'
# )
#
# long_id_rtdp_min_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_rtdp_stop_no_improvement_min_rtdp_dijkstra_heuristic_describer.description})',
#     func=partial(id,
#                  long_rtdp_stop_no_improvement_min_rtdp_dijkstra_heuristic_describer.func,
#                  long_rtdp_stop_no_improvement_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_rtdp_rtdp_dijkstra_min'
# )
#
# id_ma_rtdp_pvi_min_describer = SolverDescriber(
#     description=f'ID({ma_rtdp_pvi_min_describer.description})',
#     func=partial(id,
#                  ma_rtdp_pvi_min_describer.func,
#                  ma_rtdp_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='id_ma_rtdp_pvi_min'
# )
#
# id_ma_rtdp_pvi_sum_describer = SolverDescriber(
#     description=f'ID({ma_rtdp_pvi_sum_describer.description})',
#     func=partial(id,
#                  ma_rtdp_pvi_sum_describer.func,
#                  ma_rtdp_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='id_ma_rtdp_pvi_sum'
# )
#
# long_id_ma_rtdp_min_pvi_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_pvi_min_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_pvi_min_describer.func,
#                  long_ma_rtdp_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_pvi_min'
# )
#
# long_id_ma_rtdp_sum_pvi_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_pvi_sum_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_pvi_sum_describer.func,
#                  long_ma_rtdp_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_pvi_sum'
# )
#
# long_id_ma_rtdp_min_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_min_dijkstra_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_min_dijkstra_describer.func,
#                  long_ma_rtdp_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_dijkstra_min'
# )
#
# long_id_ma_rtdp_sum_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_sum_dijkstra_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_sum_dijkstra_describer.func,
#                  long_ma_rtdp_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_dijkstra_sum'
# )
#
# long_id_ma_rtdp_sum_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_sum_rtdp_dijkstra_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_sum_rtdp_dijkstra_describer.func,
#                  long_ma_rtdp_sum_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_rtdp_dijkstra_sum'
# )
#
# long_id_ma_rtdp_min_rtdp_dijkstra_describer = SolverDescriber(
#     description=f'ID({long_ma_rtdp_min_rtdp_dijkstra_describer.description})',
#     func=partial(id,
#                  long_ma_rtdp_min_rtdp_dijkstra_describer.func,
#                  long_ma_rtdp_min_merger_describer.func),
#     extra_info=id_extra_info,
#     short_description='long_id_ma_rtdp_rtdp_dijkstra_min'
# )
