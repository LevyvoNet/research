"""
I think this is a complete and optimal algorithm for MAPF under uncertainty and it can be faster than naively merging
like in ID.
"""

import itertools

from solvers.utils import solve_independently_and_cross, get_shared_states, evaluate_policy, CrossedPolicy


def re_plan(policy, dominant_agent, dominant_policy, shared_states):
    """Re-plan for the given policy in a partial squared dimension with other_agent

    Treat only the joint states where the local state of other_agent is in shared_states. For the rest, put something
    like 'don't care' and ignore it.
    """
    raise NotImplementedError()


class ConflictTree:
    def insert(self, reward, policy):
        raise NotImplementedError()

    def pop_best(self):
        raise NotImplementedError()


def my_cbs(low_level_planner, env, info):
    self_groups = [[i] for i in range(env.n_agents)]
    info['independent_policies'] = {}

    # Plan for each agent independently
    self_policy = solve_independently_and_cross(env,
                                                self_groups,
                                                low_level_planner,
                                                info['independent_policies'])
    joint_policy = self_policy
    ct = ConflictTree()
    iter = 0
    while True:
        shared_states = None
        for a1, a2 in itertools.combinations(range(env.n_agents), 2):
            iter += 1
            info[f'{a1}_{a2}_{iter}'] = {}
            shared_states = get_shared_states(env, joint_policy, a1, a2, info[f'{a1}_{a2}_{iter}'])
            if shared_states:
                break

        if not shared_states:
            # No conflict found
            break

        # A conflict was found, re-plan for it
        # Re-plan for a1
        new_a1_policy = re_plan(joint_policy.policies[a1], a2, joint_policy.policies[a2], shared_states)
        new_combined_policy = combine_policies(new_a1_policy, a1, joint_policy)
        mdr = evaluate_policy(new_combined_policy, 100, 100)['MDR']
        ct.insert(mdr, new_combined_policy)

        # Re-plan for a2
        new_a2_policy = re_plan(joint_policy.policies[a2], a1, joint_policy.policies[a1], shared_states)
        new_combined_policy = combine_policies(new_a2_policy, a2, joint_policy)
        mdr = evaluate_policy(new_combined_policy, 100, 100)['MDR']
        ct.insert(mdr, new_combined_policy)

        # Add the new policies to the CT (conflict tree of CBS) and retrieve the next most promising node on the CT
