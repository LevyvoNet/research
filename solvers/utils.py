import itertools
import math
import time
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Dict, Callable
from collections import defaultdict

from gym_mapf.envs.mapf_env import MapfEnv, MultiAgentState, MultiAgentAction
from gym_mapf.envs.utils import get_local_view


class Policy(metaclass=ABCMeta):
    def __init__(self, env: MapfEnv, gamma: float):
        # TODO: deep copy env, don't just copy the reference
        self.env = env
        self.gamma = gamma

    @abstractmethod
    def act(self, s: MultiAgentState) -> MultiAgentAction:
        """Return the policy action for a given state

        Args:
            s (MultiAgentState): a state of self environment

        Returns:
            MultiAgentAction. The best action according to that policy.
        """


class CrossedPolicy(Policy):
    def __init__(self, env, policies, agents_groups):
        super().__init__(env, 1.0)  # This does not matter
        self.policies = policies
        self.envs = [policy.grid for policy in self.policies]
        self.agents_groups = agents_groups

    def act(self, s):
        agent_locations = self.env.state_to_locations(s)
        agent_to_action = {}

        for i in range(len(self.agents_groups)):
            local_env_agent_locations = sum([(agent_locations[agent],)
                                             for agent in self.agents_groups[i]], ())

            local_env_agent_state = self.envs[i].locations_to_state(local_env_agent_locations)

            local_action = self.policies[i].act(local_env_agent_state)

            local_vector_action = integer_action_to_vector(local_action, self.envs[i].n_agents)
            for j, agent in enumerate(self.agents_groups[i]):
                agent_to_action[agent] = local_vector_action[j]
        joint_action_vector = tuple([action for agent, action in sorted(agent_to_action.items())])
        joint_action = vector_action_to_integer(joint_action_vector)

        return joint_action


def print_path_to_state(path: dict, state: int, env: MapfEnv):
    curr_state = state
    print("final state: {}".format(env.state_to_locations(state)))
    while path[curr_state] is not None:
        curr_state, action = path[curr_state]
        print("state: {}, action: {}".format(env.state_to_locations(curr_state),
                                             integer_action_to_vector(action, env.n_agents)))


def couple_detect_conflict(env: MapfEnv,
                           joint_policy: CrossedPolicy,
                           a1: int,
                           a2: int,
                           **kwargs):
    """Detect a conflict for two specific agents"""
    info = kwargs.get('info', {})
    start = time.time()

    a1_group_idx = group_of_agent(joint_policy.agents_groups, a1)
    a2_group_idx = group_of_agent(joint_policy.agents_groups, a2)

    a1_group = joint_policy.agents_groups[a1_group_idx]
    a2_group = joint_policy.agents_groups[a2_group_idx]

    a1_idx_in_group = a1_group.index(a1)
    a2_idx_in_group = a2_group.index(a2)

    a1_group_policy = joint_policy.policies[a1_group_idx]
    a2_group_policy = joint_policy.policies[a2_group_idx]

    env1 = get_local_view(env, a1_group)
    env2 = get_local_view(env, a2_group)

    state_pairs_to_expand = {(env1.s, env2.s)}
    visited_state_pairs = set()

    while len(state_pairs_to_expand) > 0:
        (s1, s2) = state_pairs_to_expand.pop()
        loc1 = env1.state_to_locations(s1)[a1_idx_in_group]
        loc2 = env2.state_to_locations(s2)[a2_idx_in_group]
        visited_state_pairs.add((s1, s2))

        next_states1 = [next_state
                        for _, next_state, _, _ in env1.P[s1][a1_group_policy.act(s1)]]
        next_states2 = [next_state
                        for _, next_state, _, _ in env2.P[s2][a2_group_policy.act(s2)]]

        next_locs1 = set([env1.state_to_locations(n1)[a1_idx_in_group] for n1 in next_states1])
        next_locs2 = set([env2.state_to_locations(n2)[a2_idx_in_group] for n2 in next_states2])

        # Check for a clash conflict
        clash_locs = next_locs1.intersection(next_locs2)
        if clash_locs:
            clash_loc = clash_locs.pop()
            single_agent_local_env = get_local_view(env, [0])
            info['detect_conflict_time'] = round(time.time() - start, 2)

            return (
                (a1,
                 single_agent_local_env.locations_to_state((loc1,)),
                 single_agent_local_env.locations_to_state((clash_loc,))
                 ),
                (a2,
                 single_agent_local_env.locations_to_state((loc2,)),
                 single_agent_local_env.locations_to_state((clash_loc,))
                 )
            )

        # Check for a switch conflict
        if loc1 in next_locs2 and loc2 in next_locs1:
            info['detect_conflict_time'] = round(time.time() - start, 2)
            single_agent_local_env = get_local_view(env, [0])

            return (
                (a1,
                 single_agent_local_env.locations_to_state((loc1,)),
                 single_agent_local_env.locations_to_state((loc2,))
                 ),
                (a2,
                 single_agent_local_env.locations_to_state((loc2,)),
                 single_agent_local_env.locations_to_state((loc1,))
                 )
            )

        # No conflict detected yet, add states to expand and keep searching
        next_pairs = set(itertools.product(next_states1, next_states2))
        state_pairs_to_expand.update(next_pairs.difference(visited_state_pairs))

    # Done expanding all pairs without a clash, return no conflict is possible
    info['detect_conflict_time'] = round(time.time() - start, 2)
    return None


def detect_conflict(env: MapfEnv,
                    joint_policy: CrossedPolicy,
                    **kwargs):
    """Find a conflict between agents.

    A conflict is ((i,,s_i,new_s_i), (j,s_j,new_s_j)) where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * new_s_i = local state which agent i was in after the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * new_s_j - local state which agent j was in after the clash
    """
    info = kwargs.get('info', {})
    start = time.time()

    for (g1, g2) in itertools.combinations(range(len(joint_policy.agents_groups)), 2):
        for (a1, a2) in itertools.product(joint_policy.agents_groups[g1], joint_policy.agents_groups[g2]):
            conflict = couple_detect_conflict(env, joint_policy, a1, a2)
            if conflict is not None:
                info['detect_conflict_time'] = round(time.time() - start, 2)
                return conflict

    info['detect_conflict_time'] = round(time.time() - start, 2)
    return None


def might_conflict(env, state, transitions):
    for prob, new_state, reward, done in transitions:
        if env.is_collision_transition(state, new_state):
            return True

    return False


def safe_actions(env: MapfEnv, s):
    return [a for a in env.action_space
            if not might_conflict(env, s, env.P[s][a])]


def solve_independently_and_cross(env,
                                  agent_groups,
                                  low_level_planner: Callable[[MapfEnv, Dict], Policy],
                                  info: Dict):
    """Solve the MDP MAPF for the local views of the given agent groups

    Args:
        agent_groups (list): a list of lists, each list is a group of agents.
        low_level_planner ((MapfEnv)->Policy): a low level planner to solve the local envs with.
        info (dict): information to update during the solving

        If there is an existing solution for a slightly different division of groups, one can pass it to this fucntion
        for optimization.
    """
    start = time.time()  # TODO: use a decorator for updating info with time measurement
    local_envs = [get_local_view(env, group) for group in agent_groups]

    policies = []
    for group, local_env in zip(agent_groups, local_envs):
        info[f'{group}'] = {}
        policy = low_level_planner(local_env, info[f'{group}'])
        policies.append(policy)

    joint_policy = CrossedPolicy(env, policies, agent_groups)

    end = time.time()
    info['best_joint_policy_time'] = round(end - start, 2)

    return joint_policy


def render_states(env, states):
    s_initial = env.s
    for state in states:
        env.s = state
        print(state)
        env.render()

    env.s = s_initial


def evaluate_policy(policy: Policy, n_episodes: int, max_steps: int, debug=False):
    episodes_rewards = []
    clashed = False
    for i in range(n_episodes):
        policy.env.reset()
        done = False
        steps = 0
        episode_reward = 0
        while not done and steps < max_steps:
            # # debug print
            # if debug:
            #     print(f'steps={steps}')
            #     policy.env.render()
            #     print(f'action will be {policy.act(policy.env.s)}')
            #     time.sleep(0.1)

            prev_state = policy.env.s
            new_state, reward, done, info = policy.env.step(policy.act(policy.env.s))
            episode_reward += reward
            steps += 1
            if policy.env.is_collision_transition(prev_state, new_state):
                clashed = True

        episodes_rewards.append(episode_reward)

    policy.env.reset()

    return sum(episodes_rewards) / n_episodes, clashed, episodes_rewards


class ValueFunctionPolicy(Policy):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)
        self.v = []
        self.policy_cache = {}

    def _act_in_unfamiliar_state(self, s: MultiAgentState):
        max_value = -math.inf
        best_action = None
        for a in self.env.action_space:
            q_sa = 0
            for p, s_, r, done in self.env.P[s][a]:
                if self.env.is_collision_transition(s, s_) and done:
                    q_sa = -math.inf
                    break

                q_sa += (p * (r + self.gamma * self.v[s_]))

            if q_sa > max_value:
                max_value = q_sa
                best_action = a

        # self.policy_cache[s] = best_action
        return best_action

    def act(self, s: MultiAgentState):
        if s in self.policy_cache:
            return self.policy_cache[s]
        else:
            return self._act_in_unfamiliar_state(s)


def group_of_agent(agents_groups, agent_idx):
    groups_of_agent = [i for i in range(len(agents_groups)) if agent_idx in agents_groups[i]]
    # if more than one group contains the given agent something is wrong
    assert len(groups_of_agent) == 1, "agent {} is in more than one group.\n agent groups are:\n {}".format(agent_idx,
                                                                                                            agents_groups)
    return groups_of_agent[0]


def merge_agent_groups(agents_groups, g1, g2):
    return [agents_groups[i] for i in range(len(agents_groups)) if i not in [g1, g2]] + [
        sorted(agents_groups[g1] + agents_groups[g2])]
