from architecture import PlaceholderNet
 
import numpy as np
 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
 
class ReinforceAgent:
  def __init__(self, lr, n_inputs, n_actions, gamma=0.99):
    self.gamma = gamma
    self.log_probs = []
    self.rewards = []
 
    self.policy = PlaceholderNet(lr=lr, n_inputs=n_inputs, n_hidden_1=128, n_hidden_2=128, n_outs=n_actions)
     
  def choose_action(self, obs):
    probs = F.softmax(self.policy.forward(obs), dim=-1)
    action_probs = T.distributions.Categorical(probs)
    action = action_probs.sample()
    self.log_probs.append(action_probs.log_prob(action))
   
    return action.item()
 
  def train(self):
    self.policy.optimizer.zero_grad()
 
    G = np.zeros_like(self.rewards, dtype=np.float64)
    for t in range(len(self.rewards)):
      G_sum = 0
      discount = 1
      for k in range(t, len(self.rewards)):
        G_sum += self.rewards[k] * discount
        discount *= self.gamma
      G[t] = G_sum

    G = (G - np.mean(G)) / np.std(G)
    G = T.tensor(G, dtype=T.float).to(self.policy.device)
    loss = 0
 
    for g, log_prob in zip(G, self.log_probs):
      loss += -g * log_prob
 
    loss.backward()
    self.policy.optimizer.step()
 
    self.rewards = []
    self.log_probs = []