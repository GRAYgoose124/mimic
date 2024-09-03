"""Continuous Backpropagation

Algorithm 1: Continual Backprop (CBP) for a feed
From: https://arxiv.org/pdf/2108.06325
```
forward neural network with L hidden layers
Set: step-size α, replacement rate ρ, decay rate η, and
maturity threshold m (e.g. 10−4, 10−4, 0.99, and 100)
Initialize: Initialize the weights w0, ..., wL. Let, wl be
    sampled from a distribution dl
Initialize: Utilities u1, ..., uL, average feature
    activation f1, ..., fl, and ages a1, ..., aL to 0
for each input xt do
    Forward pass: pass input through the network, get the prediction, ˆyt
    Evaluate: Receive loss l(xt, ˆyt)
    Backward pass: update the weights using stochastic gradient descent
    for layer l in 1 : L do
        Update age: al + = 1
        Update feature utility: Using Equations 4, 5,
        and 6
        Find eligible features: Features with age more
        than m
        Features to replace: nl ∗ρ of eligible features
        with smallest utility, let their indices be r
        Initialize input weights: Reset the input
        weights wl−1[r] using samples from dl
        Initialize output weights: Set wl[r] to zero
        Initialize utility, feature activation, and age:
        Set ul,r,t, fl,r,t, and al,r,t to 0
        Continual Backprop in semi-stationa
```
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        self.fc1.output = x
        x = torch.tanh(self.fc2(x))
        self.fc2.output = x
        x = self.fc3(x)
        self.fc3.output = x
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        self.fc1.output = x
        x = torch.tanh(self.fc2(x))
        self.fc2.output = x
        x = self.fc3(x)
        self.fc3.output = x
        return x

class ContinualPPO:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, gamma, epsilon, alpha, rho, eta, m):
        self.policy = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.value = ValueNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value.parameters(), 'lr': lr}
        ])
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.rho = rho
        self.eta = eta
        self.m = m
        self.value_coef = 0.5  # New: coefficient for value loss
        self.entropy_coef = 0.01  # New: coefficient for entropy regularization

        # Initialize utilities, average activations, and ages
        self.utilities = [torch.zeros(hidden_dim) for _ in range(2)]  # for hidden layers
        self.avg_activations = [torch.zeros(hidden_dim) for _ in range(2)]
        self.ages = [torch.zeros(hidden_dim) for _ in range(2)]

    def get_action(self, state):
        if isinstance(state, (list, np.ndarray)):
            state = torch.FloatTensor(state)
        elif isinstance(state, tuple):
            state = torch.FloatTensor(state[0])
        else:
            state = torch.FloatTensor([state])

        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).detach()

    def update(self, states, actions, rewards, next_states, log_probs, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute advantages
        values = self.value(states).squeeze()
        next_values = self.value(next_states).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        new_logits = self.policy(states)
        new_probs = F.softmax(new_logits, dim=1)
        new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(1))).squeeze()
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, rewards + self.gamma * next_values * (1 - dones))
        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Clip gradients
        nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)  # Clip gradients
        self.optimizer.step()

        # Apply CBP
        self.apply_cbp(self.policy)
        self.apply_cbp(self.value)

        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

    def apply_cbp(self, network):
        linear_layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
        
        for l in range(len(linear_layers) - 1):
            # Update utilities
            self.update_utilities(l, linear_layers[l], linear_layers[l+1])
            
            # Find eligible features
            eligible = self.ages[l] > self.m
            num_replace = int(self.rho * eligible.sum().item())
            
            if num_replace > 0:
                # Replace features
                _, indices = torch.topk(self.utilities[l] * eligible, k=num_replace, largest=False)
                self.replace_features(l, indices, linear_layers[l], linear_layers[l+1])

    def update_utilities(self, l, current_layer, next_layer):
        # Get the output of the current layer (h_l)
        h_l = current_layer.output

        # Ensure dimensions match
        if h_l.size() != self.avg_activations[l].size():
            self.avg_activations[l] = torch.zeros_like(h_l)
            self.utilities[l] = torch.zeros_like(h_l)
            self.ages[l] = torch.zeros_like(h_l)

        # Update average feature activation (f_l)
        self.avg_activations[l] = (1 - self.eta) * h_l + self.eta * self.avg_activations[l]

        # Bias-corrected estimate of average feature activation (f_hat_l)
        f_hat_l = self.avg_activations[l] / (1 - self.eta ** self.ages[l])

        # Update feature utility (z_l)
        weight_magnitude = torch.sum(torch.abs(next_layer.weight.data), dim=0)
        z_l = (1 - self.eta) * torch.abs(h_l - f_hat_l) * weight_magnitude + self.eta * self.utilities[l]

        # Update overall utility (u_l)
        input_weight_magnitude = torch.sum(torch.abs(current_layer.weight.data), dim=1)
        y_l = torch.abs(h_l - f_hat_l) * weight_magnitude / input_weight_magnitude
        self.utilities[l] = (1 - self.eta) * y_l + self.eta * self.utilities[l]

        # Bias-corrected estimate of overall utility (u_hat_l)
        u_hat_l = self.utilities[l] / (1 - self.eta ** self.ages[l])

        # Update ages
        self.ages[l] += 1

        # Store updated utilities
        self.utilities[l] = u_hat_l

    def replace_features(self, l, indices, current_layer, next_layer):
        # Reset input weights
        current_layer.weight.data[:, indices] = torch.randn_like(current_layer.weight.data[:, indices]) * 0.01
        
        # Reset output weights
        next_layer.weight.data[indices, :] = 0
        
        # Reset utility, average activation, and age
        self.utilities[l][indices] = 0
        self.avg_activations[l][indices] = 0
        self.ages[l][indices] = 0