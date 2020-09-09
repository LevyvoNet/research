import multiprocessing
import logging
import functools
from typing import Callable

ERROR = 'error'
INFO = 'info'
DEBUG = 'debug'


def init_logger(date_str: str) -> logging.Logger:
    # create logger with 'spam_application'
    logger = logging.getLogger(date_str)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{date_str}.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level, for debug purpose
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def log_func(q: multiprocessing.Queue, severity: str, message: str):
    """Export a function for users in order to send messages to log"""
    q.put((severity, message))


def write_to_log_loop(date_str: str, q: multiprocessing.Queue):
    logger = init_logger(date_str)

    # fetch messages from queue and write to the log
    while True:
        severity, message = q.get()

        # Log with the proper severity
        getattr(logger, severity)(message)


def start_logger_process(date_str: str, q: multiprocessing.Queue) -> [multiprocessing.Process,
                                                                      Callable[[str, str], None]]:
    """Start the log process

    Returns:
        log_func - a function that other processes can call for sending log messages
        pid?
    """
    p = multiprocessing.Process(target=write_to_log_loop, args=(date_str, q))

    p.start()

    return p, functools.partial(log_func, q)
