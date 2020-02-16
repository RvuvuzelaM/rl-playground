import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class A2CNet(nn.Module):
  def __init__(self, lr, n_inputs, n_hidden, n_outs):
    super(A2CNet, self).__init__()

    self.lr = lr

    self.critic = nn.Sequential(
      nn.Linear(n_inputs, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, 1)
    )

    self.actor = nn.Sequential(
      nn.Linear(n_inputs, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_hidden),
      nn.ReLU(),
      nn.Linear(n_hidden, n_outs),
      nn.Softmax(),
    )

    self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
    self.to(self.device)

  def forward(self, obs):
    x = T.FloatTensor(obs).to(self.device)

    value = self.critic(x)
    probs = self.actor(x)
    dist  = Categorical(probs)
    return dist, value
