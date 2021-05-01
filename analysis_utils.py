import matplotlib.pyplot as plt
import pymongo
from typing import Callable, Dict

from available_solvers import id_extra_info


def _annotate_row(axs, row_idx, annotation):
    pad = 5
    axs[row_idx, 0].annotate(annotation, xy=(0, 0.5), xytext=(-axs[row_idx, 0].yaxis.labelpad - pad, 0),
                             xycoords=axs[row_idx, 0].yaxis.label, textcoords='offset points',
                             size='large', ha='right', va='center')


def solved_percentage(instances: pymongo.cursor.Cursor):
    solved_count = len([instance for instance in instances
                        if instance['end_reason'] == 'done' and not instance['clashed']])

    instances.rewind()

    timeout_count = len([instance for instance in instances
                         if instance['end_reason'] == 'timeout'])

    if solved_count + timeout_count == 0:
        from IPython.core.debugger import set_trace
        set_trace()

    return solved_count / (solved_count + timeout_count)


def mean_reward(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances), start=1):
        sum += instance['average_reward']

    if count == 0:
        return -1000

    return sum / count


def mean_time(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances), start=1):
        sum += instance['total_time']

    if count == 0:
        return 600

    return sum / count


def mean_makespan_bound(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances), start=1):
        makespan = min(instance['self_agent_reward'])
        sum += makespan

    if count == 0:
        return 0

    return sum / count


def mean_conflict_count(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] != 'invalid', instances), start=1):
        conflict_count = id_extra_info(instance['solver_data']).n_conflicts
        # conflict_count = len([iteration
        #                       for iteration in instance['solver_data']['iterations']
        #                       if 'conflict' in iteration])

        sum += conflict_count

    if count == 0:
        return 0

    return sum / count
