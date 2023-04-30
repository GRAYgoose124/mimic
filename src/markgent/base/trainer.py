class Trainer:
    def __init__(self, agent, num_episodes):
        self.agent = agent
        self.num_episodes = num_episodes

    def train(self):
        for i in range(self.num_episodes):
            state = self.reset_environment()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.take_action(state, action)
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            self.episode_finished(i, total_reward)

    def reset_environment(self):
        raise NotImplementedError()

    def take_action(self, state, action):
        raise NotImplementedError()

    def episode_finished(self, episode, total_reward):
        raise NotImplementedError()
