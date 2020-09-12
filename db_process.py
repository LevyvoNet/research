import multiprocessing
import functools
import time
from typing import Callable

from logger_process import INFO


def insert_to_db_loop(init_json_collection: Callable[[Callable], object],
                      q: multiprocessing.Manager().Queue,
                      log_func,
                      n_total_instances: int):
    """Insert results from queue to remote mongo database"""
    # First, get a DB connection
    collection = init_json_collection(log_func)

    # Set counter of solved problems and start measure time
    n_solved = 0
    t0 = time.time()

    # Send to remove DB instances data
    while True:
        instance_data = q.get()
        collection.insert_one(instance_data)
        n_solved += 1

        # Log about the progress every 10 instances, don't let it fail the process.
        # This is a best effort.
        try:
            if n_solved % 10 == 0:
                done_ratio = round(n_solved / n_total_instances, 1)
                minutes_from_beginning = round((time.time() - t0) / 60, 2)

                log_func(INFO,
                         f'solved {n_solved}/{n_total_instances}='
                         f'{done_ratio * 100}%'
                         f' after {minutes_from_beginning} minutes.'
                         f' Expected time left is {round(minutes_from_beginning * ((1 - done_ratio) / done_ratio), 2)} minutes')
        except Exception:
            pass


def insert_to_db_func(q: multiprocessing.Manager().Queue, instance_data):
    q.put(instance_data)


def start_db_process(init_json_collection: Callable[[Callable], object],
                     q: multiprocessing.Manager().Queue,
                     log_func,
                     n_total_instances):
    p = multiprocessing.Process(target=insert_to_db_loop,
                                args=(init_json_collection,
                                      q,
                                      log_func,
                                      n_total_instances))

    p.start()

    return p, functools.partial(insert_to_db_func, q)
