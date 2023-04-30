import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass
