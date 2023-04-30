from markgent.base.trainer import Trainer
from markgent.grid.env import GridWorld


class GridWorldTrainer(Trainer):
    def __init__(self, agent, num_episodes, grid_size=3):
        super().__init__(agent, num_episodes)
        self.world = GridWorld(grid_size=grid_size)
        self.goal = 0

    def reset_environment(self):
        return (0, 0)

    def take_action(self, state, action):
        x, y = state

        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.world.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.world.size - 1, y + 1)

        next_state = (x, y)
        reward = -1 if next_state != self.world.reward_point else self.world.size**2

        # shoudl be negative/error
        reward -= (x - self.world.reward_point[0]) ** 2 + (
            y - self.world.reward_point[1]
        ) ** 2

        # done when the agent reaches the goal
        done = reward >= self.goal
        return next_state, reward, done

    def episode_finished(self, episode, total_reward):
        print(f"Episode {episode}: Total reward = {total_reward}")
