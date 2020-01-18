import pymongo
import datetime
import stopit
import time

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers.id import ID

MONGODB_URL = "mongodb://localhost:27017/"
SINGLE_SCENARIO_TIMEOUT = 60 * 5


def bechmark_main():
    # Experiment configuration
    maps = ['room-32-32-4',
            'room-64-64-8',
            'room-64-64-16'
            ]
    n_agents = 2

    # Set the DB stuff for the current experiment
    client = pymongo.MongoClient(MONGODB_URL)
    date_str = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M")
    db = client[f'id-room-benchmarks_{date_str}']

    for map in maps:
        for scen_id in range(1, 26):
            print(f'starting scen {scen_id} on map {map}')
            scen_data = {
                'scen_id': scen_id,
                'map': map
            }

            # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
            try:
                env = create_mapf_env(map, scen_id, n_agents, 0.05, 0.05, -1, 1, -0.0001)
            except KeyError:
                print('{} is invalid'.format(scen_id))
                continue

            # Run the solver
            scen_data.update({'solver_data': {}})
            with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
                try:
                    start = time.time()
                    policy = ID(env, scen_data['solver_data'])
                except stopit.utils.TimeoutException:
                    print(f'scen {scen_id} on map {map} got timeout')
                    scen_data['end_reason'] = 'timeout'

                end = time.time()
                scen_data['time'] = end - start

            if 'end_reason' not in scen_data:
                scen_data['end_reason'] = 'done'

            db[f"{map}_{n_agents}_agents"].insert_one(scen_data)


if __name__ == '__main__':
    bechmark_main()
