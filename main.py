import argparse
import datetime
import multiprocessing
import itertools
import time
import stopit
from typing import Iterable
from collections import namedtuple
from functools import partial, reduce
from pathos.multiprocessing import ProcessPool

from logger_process import start_logger_process, ERROR, INFO, DEBUG
from db_process import start_db_process

from gym_mapf.envs.utils import create_mapf_env, get_local_view
from gym_mapf.solvers.utils import evaluate_policy
from gym_mapf.solvers.rtdp import local_views_prioritized_value_iteration_heuristic
from gym_mapf.solvers import (id,
                              value_iteration,
                              prioritized_value_iteration,
                              policy_iteration,
                              rtdp,
                              stop_when_no_improvement_between_batches_rtdp,
                              fixed_iterations_count_rtdp,
                              lrtdp
                              )

# *************** Dependency Injection *************************************************************************
import db_providers.tinymongo_db_provider as db_provider

# import db_providers.pymongo_db_provider as db_provider

# *************** DB parameters ********************************************************************************
DB_NAME = 'uncertain_mapf_benchmarks'

# *************** Running parameters ***************************************************************************
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 5 * SECONDS_IN_MINUTE
CHUNK_SIZE = 10  # How many instances to solve in a single process

# *************** 'Structs' definitions ************************************************************************
FunctionDescriber = namedtuple('Solver', [
    'description',
    'func'
])

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
    # 'room-64-64-8',
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

local_pvi_heuristic_describer = FunctionDescriber(
    description='local_view_pvi_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_heuristic, 1.0))

vi_describer = FunctionDescriber(
    description='value_iteration(gamma=1.0)',
    func=partial(value_iteration, 1.0))

rtdp_stop_no_improvement_describer = FunctionDescriber(
    description=f'stop_no_improvement_rtdp('
                f'{local_pvi_heuristic_describer.description},'
                f'gamma=1.0,'
                f'batch_size=100,'
                f'max_iters=10000)',
    func=partial(stop_when_no_improvement_between_batches_rtdp,
                 local_pvi_heuristic_describer.func,
                 1.0,
                 100,
                 10000)
)

id_rtdp_describer = FunctionDescriber(
    description=f'ID({rtdp_stop_no_improvement_describer.description})',
    func=partial(id, rtdp_stop_no_improvement_describer.func)
)

POSSIBLE_SOLVERS = [
    id_rtdp_describer,
    rtdp_stop_no_improvement_describer,
    # vi_describer,
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
    for i in range(env.n_agents):
        pvi_plan_func = partial(prioritized_value_iteration, 1.0)
        local_env = get_local_view(env, [i])
        policy = pvi_plan_func(local_env, {})
        local_env.reset()
        self_agent_reward = float(policy.v[local_env.s])
        instance_data['self_agent_reward'].append(self_agent_reward)

    # Insert stats about this instance to the DB
    insert_to_db_func(instance_data)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-r', '--resume', help='Resume existing experiment', required=False)
    args = vars(parser.parse_args())

    # Initialize the experiment data - what is the collection and what are the instances to solve. This depends
    # On the --resume parameter for the script.
    if args['resume'] is None:
        collection_name = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3), 'GMT')).strftime(
            "%Y-%m-%d_%H:%M")
        instances_chunks = full_instances_chunks_generator(CHUNK_SIZE)
    else:
        # We need to resume to an existing experiment
        collection_name = args['resume']
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

            # Create instances generator with remaining instances only
            instances_chunks = instances_chunks_generator(remain_instances, CHUNK_SIZE)

    # start logger process
    logger_q = multiprocessing.Manager().Queue()
    logger_process, log_func = start_logger_process(collection_name, logger_q)

    # Log about the experiment starting
    if args['resume'] is None:
        expected_instances_count = TOTAL_INSTANCES_COUNT
        log_func(INFO, f'Expecting about {expected_instances_count} documents in the collection in the end. '
                       f'This might be a little bit lower because of invalid environments')
    else:
        expected_instances_count = len(remain_instances)
        log_func(INFO, f'Resuming {collection_name}. {expected_instances_count} instances remaining.'
                       f'Expecting about {TOTAL_INSTANCES_COUNT} in the end.'
                       f'his might be a little bit lower because of invalid environments.')

    # start db process
    db_q = multiprocessing.Manager().Queue()
    # init_collection_func = partial(init_mongodb_collection, CLOUD_MONGODB_URL, DB_NAME, collection_name)
    init_collection_func = partial(db_provider.init_collection, db_provider.CONNECT_STR, DB_NAME, collection_name)
    db_process, insert_to_db_func = start_db_process(init_collection_func,
                                                     db_q,
                                                     log_func,
                                                     expected_instances_count)

    # define the solving function
    def solve_instances(instances):
        for instance in instances:
            solve_single_instance(log_func, insert_to_db_func, instance)

        return True

    # Solve batches of instances processes from the pool
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
