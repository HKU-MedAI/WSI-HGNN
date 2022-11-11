import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP4Layers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, softmax=False):
        super(MLP4Layers, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim * 2)
        self.linear2 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.linear3 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.linear4 = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.softmax = softmax

    def forward(self, x):
        h0 = F.relu(self.linear1(x))
        h1 = F.relu(self.linear2(h0))
        h2 = F.relu(self.linear3(h1))
        out = self.linear4(h2)
        if self.softmax:
            out = F.softmax(out, dim=-1)
        return out


class MLP2Layers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, softmax=False):
        super(MLP2Layers, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim * 2)
        self.linear2 = torch.nn.Linear(hidden_dim * 2, out_dim)
        self.softmax = softmax

    def forward(self, x):
        h0 = F.relu(self.linear1(x))
        out = self.linear2(h0)
        if self.softmax:
            out = F.log_softmax(out, dim=-1)
        return out