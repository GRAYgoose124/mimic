import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from .cbp import ContinualPPO

# CartPole-specific parameters
input_dim = 4  # CartPole has 4 observation values
output_dim = 2  # CartPole has 2 possible actions
hidden_dim = 64
lr = 1e-3
gamma = 0.99
epsilon = 0.2
alpha = 1e-4
rho = 1e-4
eta = 0.99
m = 100

# Training parameters
num_episodes = 1000
max_steps = 1000

# Initialize environment and agent
env = gym.make('CartPole-v1')
agent = ContinualPPO(input_dim, hidden_dim, output_dim, lr, gamma, epsilon, alpha, rho, eta, m)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()  # Unpack the initial state
    episode_reward = 0
    states, actions, rewards, next_states, log_probs, dones = [], [], [], [], [], []

    for step in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        log_probs.append(log_prob)
        dones.append(done)

        state = next_state
        episode_reward += reward

        if done:
            break

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    log_probs = np.array(log_probs)
    dones = np.array(dones)

    # Update the agent
    total_loss, policy_loss, value_loss, entropy = agent.update(states, actions, rewards, next_states, log_probs, dones)

    # Print episode results
    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Steps: {step+1}, "
          f"Loss: {total_loss:.4f}, Policy Loss: {policy_loss:.4f}, "
          f"Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

    # Optional: Early stopping condition
    if episode_reward >= 495:  # CartPole is considered solved at 495 points
        print(f"Environment solved in {episode+1} episodes!")
        break

env.close()
print("Training completed!")

# Optional: Test the trained agent
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
total_reward = 0

for _ in range(max_steps*2):
    action, _ = agent.get_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

    if done:
        break

env.close()
print(f"Test episode reward: {total_reward}")