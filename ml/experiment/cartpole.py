import gym
import logging

import numpy as np
from typing import Tuple

from ml.config import ENVIRONMENT_NAME_CARTPOLE
from ml.experiment.agent.dqn import DQNAgent
from ml.model.replay_memory import ReplayMemory
from ml.network.dqn.dqn import DQN
from ml.provider.env_runner import EnvironmentRunner

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment (CartPole-v0)
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    return input_dim, output_dim


def main():
    env = gym.make(ENVIRONMENT_NAME_CARTPOLE)
    env = gym.wrappers.Monitor(env, directory="monitors", force=True)
    input_dim, output_dim = get_env_dim(env)

    replay_memory = ReplayMemory(10000)
    target_reward = 200
    num_episodes = 1000
    hidden_dim = 10
    epsilon = 1.0
    gamma = 0.99
    epsilon_decay = 0.95
    min_epsilon = 0.05

    dqn = DQN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    actions = np.array([ix for ix in range(output_dim)])

    agent = DQNAgent(
        dqn=dqn,
        actions=actions,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        gamma=gamma
    )

    runner = EnvironmentRunner(agent=agent, replay_memory=replay_memory, num_episodes=num_episodes, env=env,
                               target_reward=target_reward)
    rewards = runner.run()
    LOGGER.info(f"Rewards {rewards}")


if __name__ == "__main__":
    main()
