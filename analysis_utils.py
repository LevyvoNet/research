import matplotlib.pyplot as plt
import pymongo
from typing import Callable, Dict


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

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances)):
        sum += instance['average_reward']

    if count == 0:
        return -1000

    return sum / count


def mean_time(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances)):
        sum += instance['total_time']

    if count == 0:
        return 600

    return sum / count


def mean_makespan_bound(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] == 'done', instances)):
        makespan = min(instance['self_agent_reward'])
        sum += makespan

    if count == 0:
        return 0

    return sum / count


def mean_conflict_count(instances: pymongo.cursor.Cursor):
    sum = 0
    count = 0

    for count, instance in enumerate(filter(lambda ins: ins['end_reason'] != 'invalid', instances)):
        conflict_count = len([iteration
                              for iteration in instance['solver_data']['iterations']
                              if 'conflict' in iteration])

        sum += conflict_count

    if count == 0:
        return 0

    return sum / count


def row_col_analysis(row_parameter: str,
                     col_parameter: str,
                     x_axis: str,
                     y: Callable[[pymongo.cursor.Cursor], float],
                     curve_parameter: str,
                     x_label: str,
                     y_label: str,
                     curve_parameter_legend_name: Dict,
                     collection: pymongo.collection.Collection):
    row_values = collection.distinct(row_parameter)
    col_values = collection.distinct(col_parameter)
    x_values = collection.distinct(x_axis)
    curves_values = collection.distinct(curve_parameter)

    fig, axs = plt.subplots(nrows=len(row_values),
                            ncols=len(col_values),
                            figsize=(48, 48))

    for row_idx, row_value in enumerate(row_values):
        _annotate_row(axs, row_idx, row_value)
        for col_idx, col_value in enumerate(col_values):
            # Set general properties for the current plot
            axs[row_idx, col_idx].set_title(f'{col_parameter}={col_value}')
            axs[row_idx, col_idx].set(xlabel=x_label, ylabel=y_label)

            # Now plot each curve
            for curve_value in curves_values:
                # Apply y function over x's
                xs = [
                    collection.find({
                        row_parameter: row_value,
                        col_parameter: col_value,
                        x_axis: x_value,
                        curve_parameter: curve_value,
                    })
                    for x_value in x_values]
                # try:
                y_values = [y(x) for x in xs]
                # except Exception as e:
                #     print(f'Exception was thrown for parameters:\n'
                #           f'{row_parameter}: {row_value}\n'
                #           f'{col_parameter}: {col_value}\n'
                #           f'{curve_parameter}: {curve_value}\n')
                #     print(e)

                # Add a new curve
                axs[row_idx, col_idx].plot(x_values,
                                           y_values,
                                           label=curve_parameter_legend_name[curve_value])

                axs[row_idx, col_idx].legend()

    return fig, axs
