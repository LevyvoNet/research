import pymongo
import json
import datetime

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.solvers.id import ID


def count_conflicts():
    map_name = 'room-32-32-4'
    infos = {}
    for scen_id in range(1, 26):
        print('starting scen {}'.format(scen_id))
        try:
            env = create_mapf_env(map_name, scen_id, 2, 0.05, 0.05, -1, 1, -0.0001)
        except KeyError:
            print('{} is invalid'.format(scen_id))
            continue
        p, info = ID(env)
        info[scen_id] = info
        if len(info['conflicts']) > 0:
            print('found conflict on scen {}'.format(scen_id))
        else:
            print('solved scen {} without conflicts'.format(scen_id))

    with open('results.json', 'w') as f:
        f.write(json.dumps(infos))


def try_mongo():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    date_str = str(datetime.datetime.now())
    db = client[f"mapf_id_room_benchmark_{n_agents}_agents_{date_str}"]
    db.scenarios.insert_one(scen_data)


if __name__ == '__main__':
    count_conflicts()
