from dataclasses import dataclass, field

import numpy as np
from markgent.base.agent import Agent


@dataclass
class GridAgent(Agent):
    num_states: int
    num_actions: int

    gamma: float = 0.95
    alpha: float = 0.1
    epsilon: float = 0.05

    __q_table: np.ndarray[np.float64] = field(init=False)

    def __post_init__(self):
        self.__q_table = np.zeros((self.num_states, self.num_actions))

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            # Take a random action
            action = np.random.randint(self.num_actions)
        else:
            # Choose the best action according to the Q-table
            action_values = self.__q_table[state]
            action = np.argmax(action_values)
        return action

    def learn(self, state, action, reward, next_state, done):
        td_error = (
            reward
            + self.gamma * np.max(self.__q_table[next_state])
            - self.__q_table[state, action]
        )
        self.__q_table[state, action] += self.alpha * td_error
