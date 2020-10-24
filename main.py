import argparse
import os
import json
import datetime
import multiprocessing
import itertools
import time
import stopit
from typing import Iterable
from collections import namedtuple
from functools import partial, reduce
from pathos.multiprocessing import ProcessPool

from research.logger_process import start_logger_process, ERROR, INFO, DEBUG
from research.db_process import start_db_process

from gym_mapf.envs.utils import create_mapf_env, get_local_view
from research.solvers.utils import evaluate_policy
from research.solvers.rtdp import (local_views_prioritized_value_iteration_min_heuristic,
                                   local_views_prioritized_value_iteration_sum_heuristic)
from research.solvers import (id,
                              value_iteration,
                              prioritized_value_iteration,
                              policy_iteration,
                              rtdp,
                              stop_when_no_improvement_between_batches_rtdp,
                              fixed_iterations_count_rtdp,
                              lrtdp
                              )
from research.available_solvers import *

# *************** Dependency Injection *************************************************************************
import research.db_providers.tinymongo_db_provider as db_provider

# import db_providers.pymongo_db_provider as db_provider

# *************** DB parameters ********************************************************************************
DB_NAME = 'uncertain_mapf_benchmarks'

# *************** Running parameters ***************************************************************************
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 5 * SECONDS_IN_MINUTE
CHUNK_SIZE = 25  # How many instances to solve in a single process

# *************** 'Structs' definitions ************************************************************************
InstanceMetaData = namedtuple('InstanceMetaData', [
    'map',
    'scen_id',
    'fail_prob',
    'n_agents',
    'solver',
    'plan_func',
])


def id_query(instance):
    if type(instance) == InstanceMetaData:
        return {
            'map': instance.map,
            'scen_id': instance.scen_id,
            'fail_prob': instance.fail_prob,
            'n_agents': instance.n_agents,
            'solver': instance.solver
        }
    else:
        return {
            'map': instance['map'],
            'scen_id': instance['scen_id'],
            'fail_prob': instance['fail_prob'],
            'n_agents': instance['n_agents'],
            'solver': instance['solver']
        }


# ************* Experiment parameters **************************************************************************

POSSIBLE_MAPS = [
    'room-32-32-4',
    'room-64-64-8',
    # 'room-64-64-16',
    'empty-8-8',
    'empty-16-16',
    'empty-32-32',
    'empty-48-48',
]
POSSIBLE_N_AGENTS = list(range(1, 5))

# fail prob here is the total probability to fail (half for right, half for left)
POSSIBLE_FAIL_PROB = [
    0,
    0.1,
    0.2,
    0.3,
]

SCENES_PER_MAP_COUNT = 25
POSSIBLE_SCEN_IDS = list(range(1, SCENES_PER_MAP_COUNT + 1))

POSSIBLE_SOLVERS = [
    long_ma_rtdp_min_describer,
    long_ma_rtdp_sum_describer,
]

TOTAL_INSTANCES_COUNT = reduce(lambda x, y: x * len(y),
                               [
                                   POSSIBLE_MAPS,
                                   POSSIBLE_N_AGENTS,
                                   POSSIBLE_FAIL_PROB,
                                   POSSIBLE_SCEN_IDS,
                                   POSSIBLE_SOLVERS,
                               ],
                               1)


def instances_chunks_generator(instances: Iterable, chunk_size: int):
    local_instances = iter(instances)

    while True:
        chunk = list(itertools.islice(local_instances, chunk_size))
        if len(chunk) < chunk_size:
            break

        yield chunk

    yield chunk


def full_instances_chunks_generator(chunk_size: int):
    products = itertools.product(POSSIBLE_MAPS,
                                 POSSIBLE_N_AGENTS,
                                 POSSIBLE_FAIL_PROB,
                                 POSSIBLE_SCEN_IDS,
                                 POSSIBLE_SOLVERS)
    all_instances = map(
        lambda comb: InstanceMetaData(comb[0],
                                      comb[3],
                                      comb[2],
                                      comb[1],
                                      comb[4].description,
                                      comb[4].func)
        , products
    )

    return instances_chunks_generator(all_instances, chunk_size)


def solve_single_instance(log_func, insert_to_db_func, instance: InstanceMetaData):
    instance_data = {
        'type': 'instance_data',
        'map': instance.map,
        'scen_id': instance.scen_id,
        'fail_prob': instance.fail_prob,
        'n_agents': instance.n_agents,
        'solver': instance.solver
    }
    configuration_string = '_'.join([f'{key}:{value}'
                                     for key, value in instance_data.items()])
    log_func(DEBUG, f'starting {configuration_string}')

    # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
    try:
        env = create_mapf_env(instance.map,
                              instance.scen_id,
                              instance.n_agents,
                              instance.fail_prob / 2,
                              instance.fail_prob / 2,
                              -1000,
                              -1,
                              -1)
    except KeyError:
        log_func(ERROR, f'{instance.map}:{instance.scen_id} with {instance.n_agents} agents is invalid')
        instance_data.update({'solver_data': {},
                              'end_reason': 'invalid_env'})
        insert_to_db_func(instance_data)
        return


    # Run the solver
    instance_data.update({'solver_data': {}})
    with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
        try:
            start = time.time()
            policy = instance.plan_func(env, instance_data['solver_data'])
            if policy is not None:
                # policy might be None if the problem is too big for the solver
                reward, clashed = evaluate_policy(policy, 100, 1000)
                instance_data['average_reward'] = reward
                instance_data['clashed'] = clashed
        except stopit.utils.TimeoutException:
            log_func(DEBUG, f'scen {instance.scen_id} on map {instance.map} with solver {instance.solver} got timeout')
            instance_data['end_reason'] = 'timeout'

        end = time.time()
        instance_data['total_time'] = round(end - start, 2)

    if 'end_reason' not in instance_data:
        instance_data['end_reason'] = 'done'

    instance_data['self_agent_reward'] = []
    log_func(DEBUG, f'starting solving independent agents for {configuration_string}')
    for i in range(env.n_agents):
        pvi_plan_func = partial(prioritized_value_iteration, 1.0)
        local_env = get_local_view(env, [i])
        policy = pvi_plan_func(local_env, {})
        local_env.reset()
        self_agent_reward = float(policy.v[local_env.s])
        instance_data['self_agent_reward'].append(self_agent_reward)

    log_func(DEBUG, f'inserting {configuration_string} to DB')
    # Insert stats about this instance to the DB
    insert_to_db_func(instance_data)


def dump_leftovers(collection_name):
    print(f'dumping leftovers of collection {collection_name}')

    with db_provider.get_client(db_provider.CONNECT_STR) as client:
        collection = client[DB_NAME][collection_name]
    # Save queries (might be to free remote DB)
    already_solved_instances = list(collection.find())
    all_instances = [instance
                     for chunk in full_instances_chunks_generator(CHUNK_SIZE)
                     for instance in chunk]

    def was_not_solved(instance):
        for solved_instance in already_solved_instances:
            if id_query(instance) == id_query(solved_instance):
                return False

        return True

    remain_instances = list(filter(was_not_solved, all_instances))

    # Convert to JSON and write to file
    file_name = f'{collection_name}_leftovers.json'
    json_instances = [id_query(instance) for instance in remain_instances]
    with open(file_name, 'w') as f:
        f.write(json.dumps({collection_name: json_instances}))

    print(f'done dumping leftovers of collection {collection_name}')
    return file_name


def get_collection_name_and_instances_from_file(file_name):
    print(f'loading instances from {file_name}')
    with open(file_name, 'r') as f:
        json_obj = json.loads(f.read())

    collection_name = next(iter(json_obj.keys()))
    leftover_instances = []
    for instance_dict in json_obj[collection_name]:
        solver = list(filter(lambda x: x.description == instance_dict['solver'], POSSIBLE_SOLVERS))[0]
        leftover_instances.append(InstanceMetaData(map=instance_dict['map'],
                                                   scen_id=instance_dict['scen_id'],
                                                   fail_prob=instance_dict['fail_prob'],
                                                   n_agents=instance_dict['n_agents'],
                                                   solver=solver.description,
                                                   plan_func=solver.func))

    # Set the generator for the required instances
    instances_chunks = instances_chunks_generator(leftover_instances, CHUNK_SIZE)

    return collection_name, instances_chunks, len(leftover_instances)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--resume', help='Resume existing experiment', required=False)
    parser.add_argument('--from-file', help='Resume existing experiment', required=False)
    parser.add_argument('--dump-leftovers', help='Resume existing experiment', required=False)
    args = vars(parser.parse_args())

    if args['resume'] is not None:
        file_name = dump_leftovers(args['resume'])
        os.system(f'python main.py --from-file {file_name}')

    if args['dump_leftovers'] is not None:
        dump_leftovers(args['dump_leftovers'])
        return

    if args['from_file'] is not None:
        # Get collection name and instances
        collection_name, instances_chunks, instances_count = get_collection_name_and_instances_from_file(
            args['from_file'])
    else:
        collection_name = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3), 'GMT')).strftime(
            "%Y-%m-%d_%H:%M")
        instances_chunks = full_instances_chunks_generator(CHUNK_SIZE)
        instances_count = TOTAL_INSTANCES_COUNT

    # start logger process
    logger_q = multiprocessing.Manager().Queue()
    logger_process, log_func = start_logger_process(collection_name, logger_q)

    # Log about the experiment starting
    log_func(INFO, f'Running {instances_count} instances, expecting eventual {TOTAL_INSTANCES_COUNT}.')

    # start db process
    db_q = multiprocessing.Manager().Queue()
    # init_collection_func = partial(init_mongodb_collection, CLOUD_MONGODB_URL, DB_NAME, collection_name)
    init_collection_func = partial(db_provider.init_collection, db_provider.CONNECT_STR, DB_NAME,
                                   collection_name)
    db_process, insert_to_db_func = start_db_process(init_collection_func,
                                                     db_q,
                                                     log_func,
                                                     instances_count)

    # define the solving function
    def solve_instances(instances):
        for instance in instances:
            solve_single_instance(log_func, insert_to_db_func, instance)

        return True

    # Solve batches of instances processes from the pool
    # TODO: find another way, the poo.map function acts weird sometimes
    with ProcessPool() as pool:
        log_func(INFO, f'Number of CPUs is {pool.ncpus}')
        pool.map(solve_instances, instances_chunks)

    # Wait for the db and logger queues to be empty
    while any([
        not logger_q.empty(),
        not db_q.empty()]
    ):
        time.sleep(5)

    # This is a patch bug fix - wait until the last instance data is inserted to DB
    # TODO: find a better way
    time.sleep(2)

    # Now terminate infinite processes
    logger_process.terminate()
    db_process.terminate()


if __name__ == '__main__':
    main()
