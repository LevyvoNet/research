import collections
import time

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs.mapf_env import OptimizationCriteria

from available_solvers import *
from solvers.utils import (solve_independently_and_cross,
                           get_reachable_states)


def main():
    for scen_id in range(1):
        start = time.time()
        # Define the env and low level solver
        env = create_mapf_env('empty-48-48', 1, 4, 0.2, -1000, 0, -1, OptimizationCriteria.Makespan)
        solver = long_rtdp_stop_no_improvement_sum_rtdp_dijkstra_heuristic_describer

        # Split and solve for each agent
        agents_groups = [[i] for i in range(env.n_agents)]
        info = {}
        info['independent_policies'] = {}
        joint_policy = solve_independently_and_cross(env,
                                                     agents_groups,
                                                     solver.func,
                                                     info['independent_policies'])

        print(info)

        # For each reachable state, calculate the agents which can reach it.
        groups = []
        states = collections.defaultdict(set)
        for agent in range(env.n_agents):
            info[f'{agent}_states'] = {}
            reachable_states = get_reachable_states(env, joint_policy, agent, info[f'{agent}_states'])
            for s in reachable_states:
                group_for_s = states[s]
                group_for_s.add(agent)
                added = False
                for group in groups:
                    if group == group_for_s:
                        states[s] = group
                        added = True

                if not added:
                    groups.append(group_for_s)
                    states[s] = group_for_s

        # Find the largest group
        max_group_size = len(max(groups, key=lambda g: len(g)))

        # Calculate conflicts statistics
        print(f'scen {scen_id}, '
              f'max group size: {max_group_size}, '
              f'took {round(time.time() - start)} seconds,  '
              f'There are {len(groups)} groups,  '
              f'groups are:')
        for group in groups:
            print(group)


if __name__ == '__main__':
    main()
