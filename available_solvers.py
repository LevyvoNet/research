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
# TODO: the fact that I'm passing gamma here is a hack, perhaps the heuristic functions should receive gamma and env.
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
