from markgent.grid import *


def main():
    grid_size = 4
    num_states = grid_size**2
    num_actions = 4
    gamma = 0.9
    alpha = 0.15

    q_agent = GridAgent(num_actions, num_states, gamma=gamma, alpha=alpha)
    trainer = GridWorldTrainer(q_agent, num_episodes=100, grid_size=grid_size)
    trainer.train()

    # Test the trained agent in the grid world environment
    env = GridWorld(grid_size=grid_size)
    state = env.reset()
    done = False
    total_reward = 0
    counter = 0

    while not done:
        if counter % 100 == 0:
            env.render()

        action = q_agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        counter += 1

    print(f"Total reward = {total_reward}")


if __name__ == "__main__":
    main()
