from architecture import A2CNet
from torch.distributions import Categorical

import torch as T
import torch.nn.utils as nn_utils

class A2CAgent:
  def __init__(self, lr, n_inputs, n_outs, n_hidden, td_n, gamma=0.99):
    self.td_n = td_n
    self.gamma = gamma
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.masks = []
    self.entropy = 0

    self.actor_critic = A2CNet(lr=lr, n_inputs=n_inputs, n_hidden=n_hidden, n_outs=n_outs)

  def train(self, next_state):
    self.actor_critic.optimizer.zero_grad()

    next_state = T.FloatTensor(next_state).to(self.actor_critic.device)
    _, next_val = self.actor_critic(next_state)

    returns = self._compute_returns(next_val)

    log_probs = T.cat(self.log_probs)
    returns = T.cat(returns).detach()
    values = T.cat(self.values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    # Add support for batching
    loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

    loss.backward()
    # nn_utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1)
    self.actor_critic.optimizer.step()

  def step(self, obs):
    state = T.FloatTensor(obs).to(self.actor_critic.device)
    probs, val = self.actor_critic(state)
    dist = Categorical(probs)
    return dist, val 

  def append_log_prob(self, log_prob):
    self.log_probs.append(log_prob)

  def append_value(self, value):
    self.values.append(value)

  def append_reward(self, reward):
    self.rewards.append(T.FloatTensor(reward).unsqueeze(1).to(self.actor_critic.device))

  def append_done(self, done):
    self.masks.append(T.FloatTensor(1 - done).unsqueeze(1).to(self.actor_critic.device))

  def clear_mem(self):
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.masks = []
    self.entropy = 0
  
  def test_env(self, env):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = T.FloatTensor(state).unsqueeze(0).to(self.actor_critic.device)

        probs, _ = self.actor_critic(state)
        dist  = Categorical(probs)

        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        
        state = next_state
        total_reward += reward
    return total_reward

  def _compute_returns(self, next_val):
    R = next_val
    returns = []
    for i in reversed(range(len(self.rewards))):
      R = self.rewards[i] + self.gamma * R * self.masks[i]
      returns.insert(0, R)
    return returns
