import pymongo
import datetime
import stopit
import time

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers import (IdPlanner,
                              ValueIterationPlanner,
                              PrioritizedValueIterationPlanner,
                              PolicyIterationPlanner,
                              RtdpPlanner)
from gym_mapf.solvers.utils import render_states, Policy
from gym_mapf.envs.utils import get_local_view
from gym_mapf.solvers.rtdp import manhattan_heuristic, prioritized_value_iteration_heuristic

MONGODB_URL = "mongodb://localhost:27017/"
ONLINE_MONGODB_URL = "mongodb+srv://mapf_benchmark:mapf_benchmark@mapf-g2l6q.gcp.mongodb.net/test"
# MONGODB_URL = 'mongodb+srv://LevyvoNet:<password>@mapf-benchmarks-yczd5.gcp.mongodb.net/test?retryWrites=true&w=majority'
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 5 * SECONDS_IN_MINUTE
SCENES_PER_MAP_COUNT = 2


def benchmark_main():
    # Experiment configuration
    possible_maps = [
        'room-32-32-4',
        # 'room-64-64-8',
        # 'room-64-64-16'
    ]
    possible_n_agents = [
        1,
        2,
        # 3,
    ]
    possible_fail_prob = [
        0,
        0.1,
        0.2,
    ]

    id_planner = IdPlanner(RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0))

    possible_solvers = [
        id_planner,
        # VI,
    ]

    # TODO: someday the solvers will have parameters and will need to be classes with implemented __repr__,__str__
    SOLVER_TO_STRING = {id_planner: 'ID(RTDP(pvi_heuristic, 100, 1.0))'}

    # Set the DB stuff for the current experiment
    client = pymongo.MongoClient(ONLINE_MONGODB_URL)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    db = client[f'uncertain_mapf_benchmarks']

    # insert a collection with the experiement parameters
    parameters_data = \
        {
            'type': 'parameters',
            'possible_maps': possible_maps,
            'possible_n_agents': possible_n_agents,
            'possible_fail_prob': possible_fail_prob,
            'possible_solvers': [SOLVER_TO_STRING[solver] for solver in possible_solvers],
        }
    db[date_str].insert_one(parameters_data)

    # TODO: do this with itertools
    for map in possible_maps:
        for fail_prob in possible_fail_prob:
            for n_agents in possible_n_agents:
                for scen_id in range(1, SCENES_PER_MAP_COUNT + 1):
                    for planner in possible_solvers:
                        instance_data = {
                            'type': 'instance_data',
                            'map': map,
                            'scen_id': scen_id,
                            'fail_prob': fail_prob,
                            'n_agents': n_agents,
                            'solver': SOLVER_TO_STRING[planner]
                        }
                        configuration_string = '_'.join([f'{key}:{value}' for key, value in instance_data.items()])
                        print(f'starting {configuration_string}')

                        # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
                        try:
                            env = create_mapf_env(map, scen_id, n_agents, fail_prob / 2, fail_prob / 2, -1000, 0, -1)
                        except KeyError:
                            print('{} is invalid'.format(scen_id))
                            continue

                        # Run the solver
                        instance_data.update({'solver_data': {}})
                        with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
                            try:
                                start = time.time()
                                planner.plan(env, instance_data['solver_data'])
                            except stopit.utils.TimeoutException:
                                print(f'scen {scen_id} on map {map} got timeout')
                                instance_data['end_reason'] = 'timeout'

                            end = time.time()
                            instance_data['total_time'] = end - start

                        if 'end_reason' not in instance_data:
                            instance_data['end_reason'] = 'done'

                        # Insert stats about this instance to the DB
                        db[date_str].insert_one(instance_data)


def solve_and_measure_time(env, planner, **kwargs):
    start = time.time()
    kwargs.update({'info': {}})
    ret = planner.plan(env, **kwargs)
    total_time = time.time() - start

    return ret, total_time


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


def evaluate_policy(policy: Policy, n_episodes: int, max_steps: int):
    total_reward = 0
    for i in range(n_episodes):
        policy.env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            new_state, reward, done, info = policy.env.step(policy.act(policy.env.s))
            total_reward += reward
            steps += 1
            if reward == policy.env.reward_of_clash and done:
                print("clash happened, entering debug mode")
                import ipdb
                ipdb.set_trace()

    return total_reward / n_episodes


def play_single_episode(policy: Policy):
    done = False
    policy.env.reset()
    total_reward = 0
    while not done:
        _, r, done, _ = policy.env.step(policy.act(policy.env.s))
        policy.env.render()
        time.sleep(1)
        total_reward += r

    print(f'got reward of {total_reward}')


def compare_planners(planners):
    fail_prob = 0.2

    data = {}
    for planner_name, planner in planners:
        env = create_mapf_env('room-32-32-4', 1, 2, fail_prob / 2, fail_prob / 2, -1000, 0, -1)
        policy, total_time = solve_and_measure_time(env, planner)
        reward = evaluate_policy(policy, 100, 1000)
        data[planner_name] = {'total_time': total_time,
                              'reward': reward}

    # get some sense about the best options available
    local_envs = [get_local_view(env, [i]) for i in range(env.n_agents)]
    local_v = [(PrioritizedValueIterationPlanner(1.0).plan(local_env, {}).v) for local_env in local_envs]
    for i in range(env.n_agents):
        print(f"starting state value for agent {i} is {local_v[i][local_envs[i].s]}")

    for planner_name in data:
        print(f"{planner_name}: {data[planner_name]['total_time']} seconds, {data[planner_name]['reward']} score")


if __name__ == '__main__':
    # compare_planners([
    #     # ("prioritized value iteration", PrioritizedValueIterationPlanner(1.0)),
    #     # ("ID(RTDP(pvi, 100, 1.0))", IdPlanner(RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0))),
    #     ("RTDP(pvi, 100, 1.0)",RtdpPlanner(prioritized_value_iteration_heuristic, 100, 1.0)),
    #     # ("RTDP with manhattan heuristic", RtdpPlanner(manhattan_heuristic, 100, 1.0)),
    # ])

    benchmark_main()
