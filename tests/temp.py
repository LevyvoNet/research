import collections
import time
import itertools

from gym_mapf.envs.utils import create_mapf_env
from gym_mapf.envs.mapf_env import OptimizationCriteria

from available_solvers import *
from solvers.utils import (solve_independently_and_cross,
                           get_reachable_states,
                           couple_detect_conflict)


def main():
    for scen_id in range(25, 26):
        start = time.time()
        # Define the env and low level solver
        env = create_mapf_env('empty-48-48', scen_id, 2, 0.2, -1000, 0, -1, OptimizationCriteria.Makespan)
        solver = long_rtdp_stop_no_improvement_sum_dijkstra_heuristic_describer

        # Split and solve for each agent
        agents_groups = [[i] for i in range(env.n_agents)]
        info = {}
        info['independent_policies'] = {}
        joint_policy = solve_independently_and_cross(env,
                                                     agents_groups,
                                                     solver.func,
                                                     info['independent_policies'])

        print(f'solving took {round(time.time() - start)}')

        # Print soft conflicts data
        print('--soft conflicts--------------------------------------------------------------------------')
        start = time.time()
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
        interesting_states = {s: states[s] for s in states if len(states[s]) > 1}

        # Calculate conflicts statistics
        print(f'scen {scen_id}, '
              f'max group size: {max_group_size}, '
              f'took {round(time.time() - start)} seconds,  '
              f'There are {len(groups)} groups,  '
              f'groups are:')
        for group in groups:
            print(group)

        # Print hard conflicts
        print('--hard conflicts--------------------------------------------------------------------------')
        start = time.time()
        couples = []
        for (a1, a2) in itertools.combinations(range(env.n_agents), 2):
            info[f'{a1}_{a2}'] = {}
            conflict = couple_detect_conflict(env, joint_policy, a1, a2, info[f'{a1}_{a2}'])
            if conflict is not None:
                couples.append((a1, a2))

        groups = couples[:]
        found = True
        while found:
            found = False
            for (g1, g2) in itertools.combinations(groups, 2):
                if set(g1).intersection(set(g2)):
                    new_group = tuple(sorted([agent for agent in set(g1).union(set(g2))]))
                    groups.remove(g1)
                    groups.remove(g2)
                    groups.append(new_group)
                    found = True
                    break

        max_group_size = len(max(groups, key=lambda g: len(g)))
        print(f'scen {scen_id}: '
              f'There are {len(couples)} couples, '
              f'There are {len(groups)} groups,'
              f'Max group size: {max_group_size}'
              f'conflicts took {round(time.time() - start)}')

        # print('coules are:')
        # for couple in couples:
        #     print(couple)

        print('groups are:')
        for group in groups:
            print(group)

        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    main()
