import pymongo
import datetime
import stopit
import time
from functools import partial

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs.mapf_env import MapfEnv
from gym_mapf.solvers import (id,
                              value_iteration,
                              prioritized_value_iteration,
                              policy_iteration,
                              rtdp,
                              stop_when_no_improvement_between_batches_rtdp,
                              fixed_iterations_count_rtdp,
                              lrtdp
                              )
from gym_mapf.solvers.utils import render_states, Policy, evaluate_policy
from gym_mapf.envs.utils import get_local_view
from gym_mapf.solvers.rtdp import manhattan_heuristic, prioritized_value_iteration_heuristic

MONGODB_URL = "mongodb://localhost:27017/"
ONLINE_MONGODB_URL = "mongodb+srv://mapf_benchmark:mapf_benchmark@mapf-g2l6q.gcp.mongodb.net/test"
# MONGODB_URL = 'mongodb+srv://LevyvoNet:<password>@mapf-benchmarks-yczd5.gcp.mongodb.net/test?retryWrites=true&w=majority'
SECONDS_IN_MINUTE = 60
SINGLE_SCENARIO_TIMEOUT = 5 * SECONDS_IN_MINUTE
SCENES_PER_MAP_COUNT = 25


def benchmark_main():
    # Experiment configuration
    possible_maps = [
        # 'room-32-32-4',
        # 'room-64-64-8',
        # 'room-64-64-16',
        # 'empty-8-8',
        # 'empty-16-16',
        # 'empty-32-32',
        'empty-48-48',
    ]
    possible_n_agents = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    possible_fail_prob = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
    ]

    # Planner creator functions
    def get_id_rtdp_planner():
        low_level_planner = partial(stop_when_no_improvement_between_batches_rtdp,
                                    partial(prioritized_value_iteration_heuristic, 1.0),
                                    1.0,
                                    100,
                                    10000)
        return partial(id, low_level_planner)

    def get_id_vi_planner():
        return partial(id, partial(value_iteration, 1.0))

    def get_vi_planner():
        return partial(value_iteration, 1.0)

    possible_solvers_creators = [
        get_id_rtdp_planner,
        get_id_vi_planner,
        get_vi_planner,
    ]

    # TODO: someday the solvers will have parameters and will need to be classes with implemented __repr__,__str__
    SOLVER_TO_STRING = {
        get_id_rtdp_planner: 'ID(RTDP(heuristic=pvi_heuristic, gamma=1.0, batch_size=100, max_iterations=10000))',
        get_id_vi_planner: 'ID(VI(gamma=1.0))',
        get_vi_planner: VI(gamma=1.0),
    }

    # Set the DB stuff for the current experiment
    client = pymongo.MongoClient(ONLINE_MONGODB_URL)
    date_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3), 'GMT')).strftime("%Y-%m-%d_%H:%M")
    db = client[f'uncertain_mapf_benchmarks']

    # Calculate how much instances we are expecting for sanity check (won't be exactly the same because of bad scenarios)
    expected_documents_number = len(possible_maps) * \
                                len(possible_n_agents) * \
                                len(possible_fail_prob) * \
                                len(possible_solvers_creators) * \
                                SCENES_PER_MAP_COUNT + \
                                1  # +1 for the experiment metadata document

    # insert a collection with the experiment parameters
    parameters_data = \
        {
            'type': 'parameters',
            'possible_maps': possible_maps,
            'possible_n_agents': possible_n_agents,
            'possible_fail_prob': possible_fail_prob,
            'possible_solvers': [SOLVER_TO_STRING[solver] for solver in possible_solvers_creators],
            'expected_documents_number': expected_documents_number
        }
    db[date_str].insert_one(parameters_data)

    # TODO: do this with itertools
    for map in possible_maps:
        for fail_prob in possible_fail_prob:
            for n_agents in possible_n_agents:
                for scen_id in range(1, SCENES_PER_MAP_COUNT + 1):
                    for planner_creator in possible_solvers_creators:
                        instance_data = {
                            'type': 'instance_data',
                            'map': map,
                            'scen_id': scen_id,
                            'fail_prob': fail_prob,
                            'n_agents': n_agents,
                            'solver': SOLVER_TO_STRING[planner_creator]
                        }
                        configuration_string = '_'.join([f'{key}:{value}' for key, value in instance_data.items()])
                        print(f'starting {configuration_string}')

                        # Create mapf env, some of the benchmarks from movingAI might have bugs so be careful
                        try:
                            env = create_mapf_env(map, scen_id, n_agents, fail_prob / 2, fail_prob / 2, -1000, -1, -1)
                        except KeyError:
                            print('{} is invalid'.format(scen_id))
                            continue

                        # Run the solver
                        instance_data.update({'solver_data': {}})
                        with stopit.SignalTimeout(SINGLE_SCENARIO_TIMEOUT, swallow_exc=False) as timeout_ctx:
                            try:
                                start = time.time()
                                planner = planner_creator()
                                policy = planner(env, instance_data['solver_data'])
                                reward, clashed = evaluate_policy(policy, 100, 1000)
                                instance_data['average_reward'] = reward
                                instance_data['clashed'] = clashed
                            except stopit.utils.TimeoutException:
                                print(f'scen {scen_id} on map {map} got timeout')
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
    local_v = [(partial(prioritized_value_iteration, 1.0)(local_env, {}).v) for local_env in local_envs]
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
