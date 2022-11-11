"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 graph_pooling_type="att"):
        super(GCN, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(in_dim, out_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, out_dim))

            if graph_pooling_type == 'sum':
                self.pools.append(SumPooling())
            elif graph_pooling_type == 'mean':
                self.pools.append(AvgPooling())
            elif graph_pooling_type == 'max':
                self.pools.append(MaxPooling())
            elif graph_pooling_type == 'att':
                if layer == 0:
                    gate_nn = torch.nn.Linear(in_dim, 1)
                else:
                    gate_nn = torch.nn.Linear(hidden_dim, 1)
                self.pools.append(GlobalAttentionPooling(gate_nn))
            else:
                raise NotImplementedError

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata['feat']

        h_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h_list.append(self.linears_prediction[i](self.pools[i](g, h)))
            h = layer(g, h)

        h_list.append(self.classify(self.pools[-1](g, h)))

        out = torch.stack(h_list).mean(0)

        return out
