import pymongo
import datetime
import stopit
import time

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers import (ID,
                              VI)

MONGODB_URL = "mongodb://localhost:27017/"
MINUTE = 60  # number of seconds in minutes
SINGLE_SCENARIO_TIMEOUT = 5 * MINUTE

# TODO: someday the solvers will have parameters and will need to be classes with implemented __repr__,__str__
SOLVER_TO_STRING = {ID: 'ID', 'VI': 'VI'}


def bechmark_main():
    # Experiment configuration
    possible_maps = [
        'room-32-32-4',
        # 'room-64-64-8',
        # 'room-64-64-16'
    ]
    possible_n_agents = [
        1,
        # 2,
        # 3,
    ]
    possible_fail_prob = [
        0,
        0.1,
        0.2,
    ]
    possible_solvers = [
        ID,
        VI,
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
            'possible_solvers': possible_solvers,
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
                            instance_data['time'] = end - start

                        if 'end_reason' not in instance_data:
                            instance_data['end_reason'] = 'done'

                        # Insert stats about this instance to the DB
                        db['results'].insert_one(instance_data)


if __name__ == '__main__':
    bechmark_main()
