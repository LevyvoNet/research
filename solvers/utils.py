import itertools
import time
import collections
from functools import partial
import math
import stopit
from abc import ABCMeta, abstractmethod
from typing import Dict, Callable

import numpy as np

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    integer_action_to_vector,
                                    vector_action_to_integer,
                                    STAY)
from gym_mapf.envs.utils import get_local_view


class Policy(metaclass=ABCMeta):
    def __init__(self, env, gamma, name: str = ''):
        # TODO: deep copy env, don't just copy the reference
        self.env = env
        self.gamma = gamma
        self.policy_cache = {}
        self.info = {}
        self.name = name

    def reset(self):
        self.env.reset()

    @abstractmethod
    def _act_in_unfamiliar_state(self, s: int):
        pass

    def act(self, s):
        """Return the policy action for a given state

        Args:
            s (int): a state of self environment

        Returns:
            int. The best action according to that policy.
        """
        if s in self.policy_cache:
            return self.policy_cache[s]

        return self._act_in_unfamiliar_state(s)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes, max_steps, max_exec_time=None, min_success_rate=0):
        return evaluate_policy(self, n_episodes, max_steps, max_exec_time, min_success_rate)

    def train_info(self):
        return self.info

    def eval_episode_info_update(self, stats: Dict):
        pass

    def eval_episodes_info_process(self, stats: Dict):
        pass


class CrossedPolicy(Policy):
    def __init__(self, env, gamma, policies, agents_groups, name: str = ''):
        super().__init__(env, gamma, name)
        self.policies = policies
        self.envs = [policy.env for policy in self.policies]
        self.agents_groups = agents_groups

    def _act_in_unfamiliar_state(self, s: int):
        a = self.act(s)
        self.policy_cache[s] = a
        return a

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

    def train(self, *args, **kwargs):
        pass


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
                           info):
    """Detect a conflict for two specific agents

    This method is accurate but slow
    """
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


def get_shared_states(env: MapfEnv,
                      joint_policy: CrossedPolicy,
                      agent1: int,
                      agent2: int,
                      info: Dict):
    """This method might detect false positives but not false negatives."""
    info[f'{agent1}'] = {}
    reachable_states1 = get_reachable_states(env, joint_policy, agent1, info[f'{agent1}'])
    info[f'{agent2}'] = {}
    reachable_states2 = get_reachable_states(env, joint_policy, agent2, info[f'{agent2}'])

    intersection = reachable_states1.intersection(reachable_states2)
    if intersection:
        return intersection, len(reachable_states1), len(reachable_states2)

    return None


def detect_conflict_old(env: MapfEnv,
                        joint_policy: CrossedPolicy,
                        info: Dict):
    """Find a conflict between agents.

    A conflict is ((i,,s_i,new_s_i), (j,s_j,new_s_j)) where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * new_s_i = local state which agent i was in after the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * new_s_j - local state which agent j was in after the clash
    """
    start = time.time()

    for (g1, g2) in itertools.combinations(range(len(joint_policy.agents_groups)), 2):
        for (a1, a2) in itertools.product(joint_policy.agents_groups[g1], joint_policy.agents_groups[g2]):
            info[f'{a1}_{a2}'] = {}
            conflict = couple_detect_conflict(env, joint_policy, a1, a2, info[f'{a1}_{a2}'])
            if conflict is not None:
                info['detect_conflict_time'] = round(time.time() - start, 2)
                info['conflict'] = conflict
                return a1, a2, conflict

    info['detect_conflict_time'] = round(time.time() - start, 2)
    return None


def detect_conflict(env: MapfEnv,
                    joint_policy: CrossedPolicy,
                    info: Dict):
    """Find a conflict between agents.

    A conflict is ((i,,s_i,new_s_i), (j,s_j,new_s_j)) where:
    * i - index of first conflicting agent
    * s_i - local state which agent i was in before the clash
    * new_s_i = local state which agent i was in after the clash
    * j - index of second conflicting agent
    * s_j - local state which agent j was in before the clash
    * new_s_j - local state which agent j was in after the clash
    """
    start = time.time()

    for (g1, g2) in itertools.combinations(range(len(joint_policy.agents_groups)), 2):
        for (a1, a2) in itertools.product(joint_policy.agents_groups[g1], joint_policy.agents_groups[g2]):
            info[f'{a1}_{a2}'] = {}
            conflict = get_shared_states(env, joint_policy, a1, a2, info[f'{a1}_{a2}'])
            if conflict is not None:
                info['detect_conflict_time'] = round(time.time() - start, 2)
                info['conflict'] = conflict
                return a1, a2, conflict

    info['detect_conflict_time'] = round(time.time() - start, 2)
    return None


def get_reachable_states(env: MapfEnv,
                         joint_policy: CrossedPolicy,
                         agent: int,
                         info: Dict):
    agent_group_idx = group_of_agent(joint_policy.agents_groups, agent)
    agent_group = joint_policy.agents_groups[agent_group_idx]
    agent_idx_in_group = agent_group.index(agent)
    agent_group_policy = joint_policy.policies[agent_group_idx]
    agent_group_env = get_local_view(env, agent_group)

    reachable_states = set()
    joint_states_to_expand = {agent_group_env.locations_to_state(agent_group_env.agents_starts)}
    expanded_states = set()

    while len(joint_states_to_expand) > 0:
        # Pop the next joint state to expand
        expanded_joint_state = joint_states_to_expand.pop()

        # Add the state of our examined agent to the reachable states set
        locations = agent_group_env.state_to_locations(expanded_joint_state)
        agent_location = locations[agent_idx_in_group]
        agent_state = agent_group_env.loc_to_int[agent_location]
        reachable_states.add(agent_state)

        # Mark the current expanded joint state
        expanded_states.add(expanded_joint_state)

        joint_action = agent_group_policy.act(expanded_joint_state)
        next_joint_states = [next_state
                             for _, next_state, _, _ in agent_group_env.P[expanded_joint_state][joint_action]]

        joint_states_to_expand.update(filter(lambda joint_state: joint_state not in expanded_states,
                                             next_joint_states))

    return reachable_states


def might_conflict(env, s, transitions):
    for (prob, collision), new_state, reward, done in transitions:
        if collision:
            # This is a conflict transition
            return True

    return False


def safe_actions(env: MapfEnv, s):
    return [a for a in range(env.nA)
            if not might_conflict(env, s, env.P[s][a])]


def solve_independently_and_cross(env,
                                  agent_groups,
                                  low_level_policy_creator: Callable[[MapfEnv, float], Policy],
                                  gamma,
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
        policy = low_level_policy_creator(local_env, gamma)
        policy.train()
        info[f'{group}'] = policy.info
        policies.append(policy)

    joint_policy = CrossedPolicy(env, gamma, policies, agent_groups)

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


def evaluate_policy_single_episode(policy, max_steps, stats):
    all_stay_action = vector_action_to_integer((STAY,) * policy.env.n_agents)
    episode_start_time = time.time()
    policy.reset()
    steps = 0
    episode_reward = 0

    while steps < max_steps:
        a = policy.act(policy.env.s)
        if a == all_stay_action:
            break

        _, reward, done, info = policy.env.step(a)
        episode_reward += reward
        steps += 1

        # Check for goal
        if done:
            if not info['collision']:
                stats['episodes_rewards'].append(episode_reward)
                stats['episodes_time'].append(round(time.time() - episode_start_time, 1))
                stats['MDR'] += episode_reward
                stats['mean_exec_time'] += stats['episodes_time'][-1]
            else:
                stats['clashed'] = True
            break


def evaluate_policy(policy: Policy, n_episodes: int, max_steps: int, max_exec_time=None, min_success_rate=0):
    stats = {
        'episodes_rewards': [],
        'episodes_time': [],
        'clashed': False,
        'MDR': 0,
        'mean_exec_time': 0,
    }

    for i in range(n_episodes):
        if max_exec_time is None:
            evaluate_policy_single_episode(policy, max_steps, stats)
        else:
            with stopit.SignalTimeout(max_exec_time, swallow_exc=False):
                try:
                    evaluate_policy_single_episode(policy, max_steps, stats)
                except stopit.utils.TimeoutException:
                    pass

        policy.eval_episode_info_update(stats)

        # If we don't have a chance to be in the minimum success rate, just give up
        if (n_episodes - i - 1) + len(stats['episodes_rewards']) < n_episodes * min_success_rate:
            break

    # Calculate MDR
    if len(stats['episodes_rewards']) == 0:
        stats['MDR'] = -math.inf
        stats['mean_exec_time'] = -math.inf
    else:
        stats['MDR'] = round(stats['MDR'] / len(stats['episodes_rewards']), 1)
        stats['mean_exec_time'] += round(stats['mean_exec_time'] / len(stats['episodes_time']), 1)

    # Calculate success rate
    stats['success_rate'] = round((len(stats['episodes_rewards']) / n_episodes) * 100)

    policy.eval_episodes_info_process(stats)

    policy.reset()

    del stats['episodes_rewards']
    del stats['episodes_time']

    return stats


class ValueFunctionPolicy(Policy):
    def __init__(self, env, gamma, name: str = ''):
        super().__init__(env, gamma, name)
        self.v = []

    def _act_in_unfamiliar_state(self, s: int):
        possible_actions_from_state = safe_actions(self.env, s)
        q_sa = np.zeros(len(possible_actions_from_state))
        for a_idx in range(len(possible_actions_from_state)):
            a = possible_actions_from_state[a_idx]
            for next_sr in self.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                (p, collision), s_, r, _ = next_sr
                q_sa[a_idx] += (p * (r + self.gamma * self.v[s_]))

        best_action = possible_actions_from_state[np.argmax(q_sa)]
        self.policy_cache[s] = best_action
        return best_action


def group_of_agent(agents_groups, agent_idx):
    groups_of_agent = [i for i in range(len(agents_groups)) if agent_idx in agents_groups[i]]
    # if more than one group contains the given agent something is wrong
    assert len(groups_of_agent) == 1, "agent {} is in more than one group.\n agent groups are:\n {}".format(agent_idx,
                                                                                                            agents_groups)
    return groups_of_agent[0]


def merge_agent_groups(agents_groups, g1, g2):
    return [agents_groups[i] for i in range(len(agents_groups)) if i not in [g1, g2]] + [
        sorted(agents_groups[g1] + agents_groups[g2])]


def dijkstra_distance_single_env(env):
    goal_state = env.locations_to_state(env.agents_goals)
    distance = np.full((env.nS,), math.inf)
    visited = np.full((env.nS,), False)

    # Initialize the distance from goal state to 0
    distance[goal_state] = 0

    while not visited.all():
        # Fetch the cheapest unvisited state
        masked_distance = np.ma.masked_array(distance, mask=visited)
        current_state = masked_distance.argmin()
        current_distance = distance[current_state]

        # Update the distance for each of the neighbors
        for n in env.predecessors(current_state):
            distance[n] = min(distance[n], current_distance + 1)

        # Mark the current state as visited
        visited[current_state] = True

    return distance
