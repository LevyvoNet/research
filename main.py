import datetime
import multiprocessing
import itertools
import time
import stopit
from collections import namedtuple
from functools import partial, reduce

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

# *************** DB parameters ****************************************************
MONGODB_URL = "mongodb://localhost:27017/"
CLOUD_MONGODB_URL = "mongodb+srv://mapf_benchmark:mapf_benchmark@mapf-g2l6q.gcp.mongodb.net/test"
DB_NAME = 'uncertain_mapf_benchmarks'

# *************** Running parameters ***********************************************
N_PROCESSES = 1
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 5 * SECONDS_IN_MINUTE
CHUNK_SIZE = 20

# *************** 'Structs' definitions ********************************************
FunctionDescriber = namedtuple('Solver', [
    'description',
    'func'
])

InstanceData = namedtuple('InstanceData', [
    'map',
    'scen_id',
    'fail_prob',
    'n_agents',
    'solver',
    'plan_func',
])

# ************* Experiment parameters **********************************************

POSSIBLE_MAPS = [
    # 'room-32-32-4',
    # 'room-64-64-8',
    # 'room-64-64-16',
    'empty-8-8',
    'empty-16-16',
    # 'empty-32-32',
    # 'empty-48-48',
]
POSSIBLE_N_AGENTS = list(range(1, 3))

# fail prob here is the total probability to fail (half for right, half for left)
POSSIBLE_FAIL_PROB = [
    0,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.4
]

SCENES_PER_MAP_COUNT = 5
POSSIBLE_SCEN_IDS = list(range(1, SCENES_PER_MAP_COUNT + 1))

local_pvi_heuristic_describer = FunctionDescriber(
    description='local_view_pvi_heuristic(gamma=1.0)',
    func=partial(local_views_prioritized_value_iteration_heuristic, 1.0))

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
]

EXPECTED_N_INSTANCES = reduce(lambda x, y: x * len(y),
                              [
                                  POSSIBLE_MAPS,
                                  POSSIBLE_N_AGENTS,
                                  POSSIBLE_FAIL_PROB,
                                  POSSIBLE_SCEN_IDS,
                                  POSSIBLE_SOLVERS,
                              ],
                              1)


def instances_chunks_generator(chunk_size: int):
    products = itertools.product(POSSIBLE_MAPS,
                                 POSSIBLE_N_AGENTS,
                                 POSSIBLE_FAIL_PROB,
                                 POSSIBLE_SCEN_IDS,
                                 POSSIBLE_SOLVERS)
    instances = map(
        lambda comb: InstanceData(comb[0],
                                  comb[3],
                                  comb[2],
                                  comb[1],
                                  comb[4].description,
                                  comb[4].func)
        , products
    )

    while True:
        chunk = list(itertools.islice(instances, chunk_size))
        if len(chunk) < chunk_size:
            break

        yield chunk

    yield chunk


def solve_single_instance(log_func, insert_to_db_func, instance: InstanceData):
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
    log_func(INFO, f'starting {configuration_string}')

    # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
    try:
        env = create_mapf_env(map,
                              instance.scen_id,
                              instance.n_agents,
                              instance.fail_prob / 2,
                              instance.fail_prob / 2,
                              -1000,
                              -1,
                              -1)
    except KeyError:
        log_func(ERROR, '{} is invalid'.format(instance.scen_id))
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
            log_func(INFO, f'scen {instance.scen_id} on map {map} got timeout')
            instance_data['end_reason'] = 'timeout'

        end = time.time()
        instance_data['total_time'] = end - start

    if 'end_reason' not in instance_data:
        instance_data['end_reason'] = 'done'

    instance_data['self_agent_reward'] = []
    for i in range(env.n_agents):
        pvi_plan_func = partial(prioritized_value_iteration, 1.0)
        local_env = get_local_view(env, [i])
        policy = pvi_plan_func(local_env, {})
        local_env.reset()
        self_agent_reward = policy.v[local_env.s]
        instance_data['self_agent_reward'].append(self_agent_reward)

    # Insert stats about this instance to the DB
    insert_to_db_func(instance_data)


def main():
    # initialize experiment data
    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3), 'GMT')).strftime("%Y-%m-%d_%H:%M")

    # start logger process
    logger_q = multiprocessing.Queue()
    logger_process, log_func = start_logger_process(date_str, logger_q)

    # start db process
    db_q = multiprocessing.Queue()
    db_process, insert_to_db_func = start_db_process(CLOUD_MONGODB_URL,
                                                     DB_NAME,
                                                     date_str,
                                                     db_q)

    # define the solving function
    def solve_instances(instances):
        for instance in instances:
            solve_single_instance(log_func, insert_to_db_func, instance)

        return True

    # Solve batches of instances processes from the pool
    with multiprocessing.Pool(N_PROCESSES) as pool:
        pool.map(solve_instances, instances_chunks_generator(CHUNK_SIZE))

    # Wait for the db and logger queues to be empty
    while any([logger_q.not_empty,
               db_q.not_empty]):
        time.sleep(1)

    # Now terminate infinite processes
    logger_process.terminate()
    db_process.terminate()
