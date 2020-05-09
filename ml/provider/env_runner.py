from collections import deque

import gym
import logging
import numpy as np

from ml.experiment.agent.base import Agent
from ml.model.replay_memory import ReplayMemory

LOGGER = logging.getLogger(__name__)


class EnvironmentRunner:
    def __init__(self, agent: Agent, replay_memory: ReplayMemory, num_episodes: int, env: gym.Env, target_reward: int):
        self._num_episodes = num_episodes
        self._env = env

        self._target_reward = target_reward
        self._agent = agent

        self._replay_memory = replay_memory
        self._batch_size = 50
        self._rewards = deque(maxlen=100)

    def run(self):
        for i in range(self._num_episodes):
            LOGGER.info(f"Starting episode {i}")
            reward = self._run_episode()
            self._rewards.append(reward)

            if self._enviroment_is_cleared(self._rewards):
                return self._rewards

        return self._rewards

    def _run_episode(self) -> float:
        state_current = self._env.reset()
        done = False
        total_reward = 0

        while not done:
            action_current = self._agent.get_action(state_current)
            state_next, reward_current, done, info_current = self._env.step(action_current)

            total_reward += reward_current

            if done:
                reward_current = min(-1, -reward_current)

            self._remember_observation(
                state_current=state_current,
                action_current=action_current,
                state_next=state_next,
                reward_current=reward_current,
                done=done
            )

            if self._completed_initial_exploration():
                self._train_agent()

            state_current = state_next

        LOGGER.info(f"Finished episode with reward {total_reward}")
        return total_reward

    def _train_agent(self):
        transitions = self._replay_memory.pop(50)
        LOGGER.debug(f"Training agent with {len(transitions)} steps")
        self._agent.train(transitions)

    def _enviroment_is_cleared(self, rewards):
        if len(rewards) == rewards.maxlen and (np.mean(rewards) >= 200):
            return True

        return False

    def _remember_observation(self, state_current, action_current, state_next, reward_current, done):
        state_current = self._agent.preprocess_state(state_current)
        state_next = self._agent.preprocess_state(state_next)

        self._replay_memory.push(
            current_state=state_current,
            current_action=action_current,
            next_state=state_next,
            current_reward=reward_current,
            done=done
        )

    def _completed_initial_exploration(self):
        return len(self._replay_memory) > self._batch_size
