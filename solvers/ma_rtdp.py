import time
import math
from typing import Dict, Callable
from collections import defaultdict
import numpy as np
from numpy.core.tests.test_nditer import iter_iterindices

from gym_mapf.envs.mapf_env import (MapfEnv,
                                    function_to_get_item_of_object,
                                    STAY,
                                    ACTIONS,
                                    vector_action_to_integer)
from gym_mapf.envs.utils import get_local_view

from solvers.utils import evaluate_policy
from solvers.rtdp import RtdpPolicy


class MultiagentRtdpPolicy(RtdpPolicy):
    def __init__(self, env, gamma, heuristic):
        super(MultiagentRtdpPolicy, self).__init__(env, gamma, heuristic)
        self.q_partial_table = {i: defaultdict(dict) for i in range(env.n_agents)}
        self.local_env_aux = get_local_view(self.env, [0])

    def get_q(self, agent, joint_state, local_action):
        if joint_state in self.q_partial_table[agent]:
            return self.q_partial_table[agent][joint_state][local_action]

        # Calculate Q[s][a] for each possible local action
        for a in range(len(ACTIONS)):
            all_stay = (STAY,) * self.env.n_agents
            joint_action_vector = all_stay[:agent] + (ACTIONS[a],) + all_stay[agent + 1:]
            joint_action = vector_action_to_integer(joint_action_vector)

            # Compute Q[s][a]. In case of a possible clash set the reward to -infinity
            q_value = 0
            for prob, next_state, reward, done in self.env.P[joint_state][joint_action]:
                if reward == self.env.reward_of_clash and done:
                    q_value = -math.inf

                q_value += prob * (reward + (self.gamma * self.v[next_state]))

            self.q_partial_table[agent][joint_state][a] = q_value

        return self.q_partial_table[agent][joint_state][local_action]

    def q_update(self, agent, joint_state, local_action):
        all_stay = (STAY,) * self.env.n_agents

        fake_joint_action = vector_action_to_integer(all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:])

        self.q_partial_table[agent][joint_state][local_action] = sum([prob * (reward + self.gamma * self.v[next_state])
                                                                      for prob, next_state, reward, done in
                                                                      self.env.P[joint_state][fake_joint_action]])

    def v_update(self, joint_state):
        self.v_partial_table[joint_state] = max([max([self.get_q(agent_idx, joint_state, a)
                                                      for a in range(len(ACTIONS))])
                                                 for agent_idx in range(self.env.n_agents)])

    def act(self, joint_state):
        if joint_state in self.policy_cache:
            return self.policy_cache[joint_state]

        joint_action = ()
        all_stay = (STAY,) * self.env.n_agents
        forbidden_states = set()
        for agent in range(self.env.n_agents):
            # TODO: the problem is that the best response is according to joint state even though we are in state s.
            # TODO: we shouldn't actually step in this part...
            local_action = best_response(self, joint_state, agent, forbidden_states, False)
            fake_joint_action_vector = all_stay[:agent] + (ACTIONS[local_action],) + all_stay[agent + 1:]
            fake_joint_action = vector_action_to_integer(fake_joint_action_vector)
            s, r, done, _ = self.env.step(fake_joint_action)
            joint_action = joint_action + (ACTIONS[local_action],)

        best_action = vector_action_to_integer(joint_action)
        self.policy_cache[joint_state] = best_action
        return best_action


def best_response(policy: MultiagentRtdpPolicy, joint_state: int, agent: int, forbidden_states, stochastic=True):
    action_values = [policy.get_q(agent, joint_state, local_action)
                     for local_action in range(len(ACTIONS))]

    # Make sure the chosen local action is not forbidden
    locations = policy.env.state_to_locations(joint_state)
    local_state = policy.local_env_aux.locations_to_state((locations[agent],))
    for local_action in range(len(action_values)):
        for _, next_state, prob in policy.env.single_agent_movements(local_state, local_action):
            if next_state in forbidden_states and prob > 0:
                action_values[local_action] = -math.inf

    max_value = np.max(action_values)
    if stochastic:
        best_action = np.random.choice(np.argwhere(action_values == max_value).flatten())
    else:
        best_action = np.argmax(action_values)

    # Forbid the possible states from the chosen action to the next agents
    for _, next_state, prob in policy.env.single_agent_movements(local_state, best_action):
        if prob > 0:
            forbidden_states.add(next_state)

    return best_action


def multi_agent_turn_based_rtdp_single_iteration(policy: MultiagentRtdpPolicy, info: Dict):
    s = policy.env.reset()
    done = False
    start = time.time()
    path = [s]
    total_reward = 0

    # # debug
    # print('--------start iteration---------------')

    while not done:
        trajectory_actions = []
        forbidden_states = set()
        joint_action_vector = (STAY,) * policy.env.n_agents

        # Calculate local action
        for agent in range(policy.env.n_agents):
            local_action = best_response(policy, s, agent, forbidden_states)
            trajectory_actions.append(local_action)
            joint_action_vector = joint_action_vector[:agent] + (ACTIONS[local_action],) + joint_action_vector[
                                                                                           agent + 1:]

        # # debug
        # policy.env.render()
        # print(f'selected action: {joint_action_vector}')
        # time.sleep(1)

        # Compose the joint action
        joint_action = vector_action_to_integer(joint_action_vector)

        # update the current state, TODO: maybe I should update the same way as RTDP (which updates the last state as well)?
        for agent in reversed(range(policy.env.n_agents)):
            # update q(s, agent, action) based on the last state
            policy.q_update(agent, s, trajectory_actions[agent])
            policy.v_update(s)

        # step
        s, r, done, _ = policy.env.step(joint_action)
        total_reward += r
        path.append(s)

    # # debug
    # policy.env.render()

    # Backward update
    while path:
        s = path.pop()
        policy.v_update(s)

    # TODO: There is a bug here(total_reward is always 0), will think about it later

    # # debug
    # print('--------end iteration---------------')
    return total_reward


def multi_agent_turn_based_rtdp_iterations_generator(policy, info: Dict):
    info['iterations'] = []

    while True:
        iter_info = {}
        info['iterations'].append(iter_info)
        start = time.time()
        iter_reward = multi_agent_turn_based_rtdp_single_iteration(policy, {})
        iter_info['time'] = time.time() - start
        iter_info['reward'] = iter_reward
        yield iter_reward


def ma_rtdp(heuristic_function: Callable[[MapfEnv], Callable[[int], float]],
            gamma: float,
            iterations_batch_size: int,
            max_iterations: int,
            env: MapfEnv,
            info: Dict):
    def no_improvement_from_last_batch(policy: RtdpPolicy, iter_count: int):
        if iter_count % iterations_batch_size != 0:
            return False

        policy.policy_cache.clear()
        reward, _ = evaluate_policy(policy, 100, 1000)
        if reward == policy.env.reward_of_living * 1000:
            return False

        if not hasattr(policy, 'last_eval'):
            policy.last_eval = reward
            return False
        else:
            prev_eval = policy.last_eval
            policy.last_eval = reward
            return abs(policy.last_eval - prev_eval) / abs(prev_eval) <= 0.01

    # initialize V to an upper bound
    start = time.time()
    policy = MultiagentRtdpPolicy(env, gamma, heuristic_function(env))
    info['initialization_time'] = time.time() - start
    info['total_evaluation_time'] = 0

    # Run RTDP iterations
    iter_count = 0
    for iter_count, reward in enumerate(multi_agent_turn_based_rtdp_iterations_generator(policy, info),
                                        start=1):
        # Stop when no improvement or when we have exceeded maximum number of iterations
        eval_start = time.time()
        no_improvement = no_improvement_from_last_batch(policy, iter_count)
        info['total_evaluation_time'] += time.time() - eval_start
        if no_improvement or iter_count >= max_iterations:
            break

    info['n_iterations'] = iter_count
    info['total_time'] = time.time() - start

    return policy
