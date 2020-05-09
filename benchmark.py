import pymongo
import datetime
import stopit
import time

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers import (ID,
                              value_iteration_planning,
                              prioritized_value_iteration_planning,
                              policy_iteration)

MONGODB_URL = "mongodb://localhost:27017/"
# MONGODB_URL = 'mongodb+srv://LevyvoNet:<password>@mapf-benchmarks-yczd5.gcp.mongodb.net/test?retryWrites=true&w=majority'
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 1 * SECONDS_IN_MINUTE

# TODO: someday the solvers will have parameters and will need to be classes with implemented __repr__,__str__
SOLVER_TO_STRING = {ID: 'ID',
                    value_iteration_planning: 'VI',
                    prioritized_value_iteration_planning: 'prioritized_VI'}


def benchmark_main():
    # Experiment configuration
    possible_maps = [
        'room-32-32-4',
        # 'room-64-64-8',
        # 'room-64-64-16'
    ]
    possible_n_agents = [
        # 1,
        2,
        # 3,
    ]
    possible_fail_prob = [
        # 0,
        0.1,
        # 0.2,
    ]
    possible_solvers = [
        ID,
        # VI,
    ]

    # Set the DB stuff for the current experiment
    client = pymongo.MongoClient(MONGODB_URL)
    date_str = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M")
    db = client[f'id-room-benchmark_{date_str}']

    # insert a collection with the experiement parameters
    parameters_data = \
        {
            'possible_maps': possible_maps,
            'possible_n_agents': possible_n_agents,
            'possible_fail_prob': possible_fail_prob,
            'possible_solvers': [SOLVER_TO_STRING[solver] for solver in possible_solvers],
        }
    db['parameters'].insert_one(parameters_data)

    # TODO: do this with itertools
    for map in possible_maps:
        for fail_prob in possible_fail_prob:
            for n_agents in possible_n_agents:
                for scen_id in range(1, 26):
                    for solver in possible_solvers:
                        instance_data = {
                            'map': map,
                            'scen_id': scen_id,
                            'fail_prob': fail_prob,
                            'n_agents': n_agents,
                            'solver': solver.__name__
                        }
                        configuration_string = '_'.join([f'{key}:{value}' for key, value in instance_data.items()])
                        print(f'starting {configuration_string}')

                        # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
                        try:
                            env = create_mapf_env(map, scen_id, n_agents, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
                        except KeyError:
                            print('{} is invalid'.format(scen_id))
                            continue

                        # Run the solver
                        instance_data.update({'solver_data': {}})
                        with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
                            try:
                                start = time.time()
                                solver(env, **{'info': instance_data['solver_data']})
                            except stopit.utils.TimeoutException:
                                print(f'scen {scen_id} on map {map} got timeout')
                                instance_data['end_reason'] = 'timeout'

                            end = time.time()
                            instance_data['total_time'] = end - start

                        if 'end_reason' not in instance_data:
                            instance_data['end_reason'] = 'done'

                        # Insert stats about this instance to the DB
                        db['results'].insert_one(instance_data)


def solve_and_measure_time(env, solver):
    start = time.time()
    ret = solver(env, **{'info': {}})
    print('took {} seconds'.format(time.time() - start))

    return ret


def env_transitions_calc_benchmark():
    fail_prob = 0.1
    # env = create_mapf_env('empty-8-8', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    # env = create_mapf_env('room-32-32-4', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    # test_time_for_env(env, value_iteration_planning)

    # env = create_mapf_env('empty-8-8', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    # env = create_mapf_env('room-32-32-4', 2, 1, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    # prioritized_v = test_time_for_env(env, prioritized_value_iteration_planning)

    # env = create_mapf_env('room-32-32-4', 2, 1, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    env = create_mapf_env('empty-8-8', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    start = time.time()
    for s in range(env.nS):
        for a in range(env.nA):
            x = env.P[s][a]
    print(f'env for loop took {time.time() - start} seconds')

    start = time.time()
    x = 0
    for _ in range(env.nS):
        for _ in range(env.nA):
            x += 1

    print(f'naive for loop took {time.time() - start} seconds')
    print(f'env.nS={env.nS}, env.nA={env.nA}, x={x}')


def compare_solvers():
    fail_prob = 0.1

    env = create_mapf_env('empty-8-8', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    print("value iteration")
    vi_V, vi_policy = solve_and_measure_time(env, value_iteration_planning)

    env = create_mapf_env('empty-8-8', 2, 2, fail_prob / 2, fail_prob / 2, -1, 1, -0.0001)
    print("policy iteration")
    pi_V, pi_policy = solve_and_measure_time(env, policy_iteration)

    # make sure pi has reason
    from gym_mapf.solvers.utils import render_states
    from gym_mapf.envs.mapf_env import integer_action_to_vector
    controversial_states = [s for s in range(env.nS) if vi_policy(s) != pi_policy[s]]
    print(f"There are {len(controversial_states)} controversial states")
    for s in controversial_states:
        print(f"controversial state {s}")
        render_states(env, [s])
        print(f"VI says {integer_action_to_vector(vi_policy(s), env.n_agents)}")
        print(f"PI says {integer_action_to_vector(pi_policy[s], env.n_agents)}")


if __name__ == '__main__':
    compare_solvers()
