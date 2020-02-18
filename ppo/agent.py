from architecture import Net
from torch.distributions import Categorical

import numpy as np
import torch as T

class PPOAgent:
  def __init__(self, lr, n_inputs, n_outs, n_hidden, td_n, mini_batch_size, ppo_epochs, gamma=0.99, tau=0.95, clip_param=0.2):
    self.td_n = td_n
    self.mini_batch_size = mini_batch_size
    self.ppo_epochs = ppo_epochs
    self.gamma = gamma
    self.tau = tau
    self.clip_param = clip_param
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.masks = []
    self.actions = []
    self.states = []
    self.entropy = 0

    self.net = Net(lr=lr, n_inputs=n_inputs, n_hidden=n_hidden, n_outs=n_outs)

  def train(self, next_state):
    next_state = T.FloatTensor(next_state).to(self.net.device)
    _, next_value = self.net(next_state)
    returns = self._compute_gae(next_value)

    returns = T.cat(returns).detach()
    self.log_probs = T.cat(self.log_probs).detach()
    self.values = T.cat(self.values).detach()
    self.states = T.cat(self.states)
    self.actions = T.cat(self.actions)
    advantage = returns - self.values

    self._ppo_update(returns, advantage)

  def _ppo_update(self, returns, advantages):
    for _ in range(self.ppo_epochs):
      for state, action, old_log_probs, return_, advantage in self._ppo_iter(returns, advantages):
        probs, value = self.net(state)
        dist  = Categorical(probs)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(action)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = T.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        actor_loss  = - T.min(surr1, surr2).mean()
        critic_loss = (return_ - value).pow(2).mean()

        loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()

  def _ppo_iter(self, returns, advantage):
    batch_size = self.states.size(0)
    for _ in range(batch_size // self .mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
        yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]

  def _compute_gae(self, next_value):
    values = self.values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(self.rewards))):
        delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
        gae = delta + self.gamma * self.tau * self.masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

  def step(self, obs):
    state = T.FloatTensor(obs).to(self.net.device)
    probs, val = self.net(state)
    dist = Categorical(probs)
    return dist, val

  def append_log_prob(self, log_prob):
    self.log_probs.append(log_prob)

  def append_value(self, value):
    self.values.append(value)

  def append_reward(self, reward):
    self.rewards.append(T.FloatTensor(reward).unsqueeze(1).to(self.net.device))

  def append_done(self, done):
    self.masks.append(T.FloatTensor(1 - done).unsqueeze(1).to(self.net.device))

  def append_state(self, state):
    self.states.append(T.FloatTensor(state).to(self.net.device))

  def append_action(self, action):
    self.actions.append(action)

  def clear_mem(self):
    self.log_probs = []
    self.values = []
    self.rewards = []
    self.masks = []
    self.states = []
    self.actions = []
    self.entropy = 0

  def test_env(self, env):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = T.FloatTensor(state).unsqueeze(0).to(self.net.device)

        probs, _ = self.net(state)
        dist  = Categorical(probs)

        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])

        state = next_state
        total_reward += reward
    return total_reward

  def save_model(self, path):
    T.save(self.net.state_dict(), path)

  def load_model(self, path):
    self.net.load_state_dict(T.load(path))
    self.net.eval()
