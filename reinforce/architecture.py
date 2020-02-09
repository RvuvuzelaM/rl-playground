import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
class PlaceholderNet(nn.Module):
  def __init__(self, lr, n_inputs, n_hidden_1, n_hidden_2, n_outs):
    super(PlaceholderNet, self).__init__()
 
    self.lr = lr
    self.n_outs = n_outs
 
    self.fc1 = nn.Linear(n_inputs, n_hidden_1)
    self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
    self.outs = nn.Linear(n_hidden_2, n_outs)
 
    self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
     
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
    self.to(self.device)
     
  def forward(self, obs):
    state = T.Tensor(obs).to(self.device)
    x = F.relu(self.fc1(state)) 
    x = F.relu(self.fc2(x))
    return self.outs(x)