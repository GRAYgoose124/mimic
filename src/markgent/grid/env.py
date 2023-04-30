import numpy as np


class GridWorld:
    def __init__(self, grid_size=3):
        self.size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        self.reward_point = (grid_size - 1, grid_size - 1)
        self.actor_pos = (0, 0)

    def randomize_reward_point(self):
        self.reward_point = (
            np.random.randint(self.size),
            np.random.randint(self.size),
        )

    def reset(self):
        self.actor_pos = (0, 0)
        self.grid = np.zeros((self.size, self.size))
        return self.actor_pos

    def step(self, action):
        x, y = self.actor_pos

        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.size - 1, y + 1)

        next_state = (x, y)
        if next_state == self.reward_point:
            reward = self.size**2
            self.randomize_reward_point()
        else:
            reward = -1
        done = next_state == self.reward_point

        self.grid[self.actor_pos] = 0
        self.grid[next_state] = 1

        self.actor_pos = next_state

        return next_state, reward, done

    def render(self):
        """render the grid world"""
        print(self)

    def __str__(self) -> str:
        """render but build a str instead of print"""
        s = ""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:
                    s += ". "
                elif (i, j) == self.reward_point:
                    s += "R "
                else:
                    s += "X "
            s += "\n"
        return s
