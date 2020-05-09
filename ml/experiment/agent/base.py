from typing import List

import numpy as np

from abc import ABC, abstractmethod

from ml.model.transition import Transition


class Agent(ABC):

    @abstractmethod
    def get_action(self, states: np.ndarray):
        raise NotImplementedError()

    def preprocess_state(self, state: np.ndarray):
        return state

    @abstractmethod
    def train(self, transitions: List[Transition]):
        raise NotImplementedError()
