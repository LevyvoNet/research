import itertools
import math
import time
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Dict, Callable
from collections import defaultdict

from gym_mapf.envs.mapf_env import MapfEnv, MultiAgentState, MultiAgentAction
from gym_mapf.envs.grid import SingleAgentState
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
        self.envs = [policy.env for policy in self.policies]
        self.agents_groups = agents_groups

    def act(self, s: MultiAgentState):
        joint_action = MultiAgentAction({})
        for i in range(len(self.agents_groups)):
            local_env_agent_state = MultiAgentState({agent: s[agent] for agent in self.agents_groups[i]},
                                                    self.envs[i].grid)

            group_action = self.policies[i].act(local_env_agent_state)
            for agent in self.agents_groups[i]:
                joint_action[agent] = group_action[agent]

        return joint_action


def couple_detect_conflict(env: MapfEnv,
                           joint_policy: CrossedPolicy,
                           agent1: int,
                           agent2: int,
                           **kwargs):
    info = kwargs.get('info', {})
    start = time.time()

    a1_group_idx = group_of_agent(joint_policy.agents_groups, agent1)
    a2_group_idx = group_of_agent(joint_policy.agents_groups, agent2)

    a1_group_policy = joint_policy.policies[a1_group_idx]
    a2_group_policy = joint_policy.policies[a2_group_idx]

    a1_group = joint_policy.agents_groups[a1_group_idx]
    a2_group = joint_policy.agents_groups[a2_group_idx]

    env1 = get_local_view(env, a1_group)
    env2 = get_local_view(env, a2_group)

    state_pairs_to_expand = {(MultiAgentState({agent: env.start_state[agent] for agent in a1_group}, env.grid),
                              MultiAgentState({agent: env.start_state[agent] for agent in a2_group}, env.grid))}
    visited_state_pairs = set()

    while len(state_pairs_to_expand) > 0:
        (joint_s1, joint_s2) = state_pairs_to_expand.pop()
        visited_state_pairs.add((joint_s1.hash_value, joint_s2.hash_value))
        s1 = joint_s1[agent1]
        s2 = joint_s2[agent2]

        next_joint_states1 = [next_state
                              for _, next_state, _, _ in env1.P[joint_s1][a1_group_policy.act(joint_s1)]]
        next_joint_states2 = [next_state
                              for _, next_state, _, _ in env2.P[joint_s2][a2_group_policy.act(joint_s2)]]

        # TODO: improve conflict detection by checking a clash
        #  between two groups instead of two agents

        next_states1 = set([ns[agent1] for ns in next_joint_states1])
        next_states2 = set([ns[agent2] for ns in next_joint_states2])

        # Check for a clash conflict
        clash_states = next_states1.intersection(next_states2)
        if clash_states:
            clash_state = clash_states.pop()
            info['detect_conflict_time'] = round(time.time() - start, 2)
            return (agent1, s1, clash_state), (agent2, s2, clash_state)

        # Check for a switch conflict
        if s1 in next_states2 and s2 in next_states1:
            info['detect_conflict_time'] = round(time.time() - start, 2)
            return (agent1, s1, s2), (agent2, s2, s1)

        state_pairs_to_expand.update(filter(
            lambda joint_state_pair: (joint_state_pair[0].hash_value,
                                      joint_state_pair[1].hash_value) not in visited_state_pairs,
            itertools.product([n for n in next_joint_states1],
                              [n for n in next_joint_states2])))

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
