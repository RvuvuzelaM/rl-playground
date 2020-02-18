import torch as T
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
  def __init__(self, lr, n_inputs, n_hidden, n_outs):
    super(Net, self).__init__()

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
        nn.Softmax(dim=1) # this is for training
        # nn.Softmax(dim=-1) # this is for evaluating
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr)

    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
    self.to(self.device)

  def forward(self, x):
    value = self.critic(x)
    probs = self.actor(x)
    return probs, value
