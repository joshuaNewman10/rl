import numpy as np

from dataclasses import dataclass


@dataclass
class Transition:
    current_state: np.ndarray
    next_state: np.ndarray
    current_action: int
    current_reward: int
    done: bool
