from typing import List

import numpy as np
import torch

from ml.agent import Agent
from ml.model.transition import Transition
from ml.network.dqn.dqn import DQN


class DQNAgent(Agent):
    def __init__(self, dqn: DQN, actions: np.ndarray, input_dim: int, output_dim: int, hidden_dim: int,
                 epsilon: float, epsilon_decay: float, min_epsilon: float, gamma: float) -> None:
        self._dqn = dqn

        self._actions = actions

        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._gamma = gamma

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

        self._loss_fn = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._dqn.parameters())

    def get_action(self, state: np.ndarray):
        if np.random.rand() < self._epsilon:
            self._epsilon = self._epsilon * self._epsilon_decay
            return np.random.choice(self._actions)
        else:
            state = np.vstack([state])
            pred = self._get_Q(state)
            _, argmax = torch.max(pred.data, 1)
            action_ix = argmax.data.item()
            return self._actions[action_ix]

    def preprocess_state(self, state: np.ndarray):
        state = self._to_variable(state)
        state = state.float()
        return state

    def train(self, transitions: List[Transition]) -> float:
        states = np.vstack([x.current_state for x in transitions])
        current_actions = np.array([x.current_action for x in transitions])
        current_rewards = np.array([x.current_reward for x in transitions])
        next_states = np.vstack([x.next_state for x in transitions])
        done = np.array([x.done for x in transitions])

        Q_predict = self._get_Q(states)
        Q_target = Q_predict.clone().data.numpy()

        for q_target, next_state, done, reward, action in zip(Q_target, next_states, done, current_rewards, current_actions):
            next_state = np.array(next_state)
            next_state = np.reshape(next_state, (-1, self._input_dim))  # reshapes to batch size of 1

            if done:
                q_value = reward  # this will be a negative number (penalty)
            else:
                preds = self._get_Q(next_state)  # predicted Q for each action from DQN
                q_value = np.max(preds.data.numpy())  # predicted Q for best action
                q_value = reward + (self._gamma * q_value) # add reward to PDV of action

            q_target[action] = q_value # teach DQN value of that action (we are peaking into the future using the next state)

        """
        Can do this in one line but is too much going on here
        Q_target[np.arange(len(Q_target)), current_actions] = current_rewards + self._gamma * np.max(
            self._get_Q(next_states).data.numpy(),
            axis=1) * ~done
        """
        Q_target = self._to_variable(Q_target).float()

        loss = self._train(Q_predict, Q_target)
        return loss

    def _train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self._dqn.train(mode=True)
        self._optimizer.zero_grad()

        loss = self._loss_fn(input=Q_pred, target=Q_true)
        loss.backward()

        self._optimizer.step()

        return loss.data.item()

    def _get_Q(self, states) -> torch.FloatTensor:
        states = self._to_variable(states.reshape(-1, self._input_dim))
        self._dqn.train(mode=False)
        preds = self._dqn(states)
        return preds

    def _to_variable(self, x: np.ndarray):
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))
