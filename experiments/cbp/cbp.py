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
        
        utilities = [torch.zeros(layer.out_features, device=layer.weight.device) for layer in linear_layers]
        feature_activations = [torch.zeros(layer.out_features, device=layer.weight.device) for layer in linear_layers]
        ages = [torch.zeros(layer.out_features, device=layer.weight.device) for layer in linear_layers]
        z_values = [torch.zeros(layer.out_features, device=layer.weight.device) for layer in linear_layers]
        y_values = [torch.zeros(layer.out_features, device=layer.weight.device) for layer in linear_layers]

        self.update_utilities(linear_layers, utilities, feature_activations, ages, z_values, y_values)
        self.replace_features(linear_layers, utilities, ages)

    def update_utilities(self, linear_layers, utilities, feature_activations, ages, z_values, y_values):
        for l in range(len(linear_layers) - 1):
            h_l = linear_layers[l].output.mean(dim=0)
            w_l = linear_layers[l+1].weight
            w_l_prev = linear_layers[l].weight if l > 0 else None

            f_l = (1 - self.eta) * h_l + self.eta * feature_activations[l]
            f_l_hat = feature_activations[l] / (1 - self.eta ** ages[l].clamp(min=1e-8))

            z_l = (1 - self.eta) * torch.abs(h_l - f_l_hat) * torch.sum(torch.abs(w_l), dim=0) + self.eta * z_values[l]

            if w_l_prev is not None:
                y_l = torch.abs(h_l - f_l_hat) * torch.sum(torch.abs(w_l), dim=0) / torch.sum(torch.abs(w_l_prev), dim=1).clamp(min=1e-8)
            else:
                y_l = torch.abs(h_l - f_l_hat) * torch.sum(torch.abs(w_l), dim=0)

            u_l = (1 - self.eta) * y_l + self.eta * utilities[l]
            u_l_hat = u_l / (1 - self.eta ** ages[l].clamp(min=1e-8))

            feature_activations[l] = f_l
            z_values[l] = z_l
            y_values[l] = y_l
            utilities[l] = u_l_hat
            ages[l] += 1

    def replace_features(self, linear_layers, utilities, ages):
        for l in range(len(linear_layers)):
            eligible_features = (ages[l] > self.m).float()
            num_replace = int(self.rho * eligible_features.sum().item())
            
            if num_replace > 0:
                _, indices = torch.topk(utilities[l] * eligible_features, k=num_replace, largest=False)
                
                # Reset input weights
                linear_layers[l].weight.data[:, indices] = torch.randn_like(linear_layers[l].weight.data[:, indices]) * 0.01
                
                # Reset output weights if not the last layer
                if l < len(linear_layers) - 1:
                    linear_layers[l+1].weight.data[indices, :] = 0
                
                # Reset utility, feature activation, and age
                utilities[l][indices] = 0
                ages[l][indices] = 0
