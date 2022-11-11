"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling
from pooling import AvgPooling, SumPooling, MaxPooling


class NTPoolGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 node_dict,
                 n_layers,
                 activation,
                 dropout,
                 graph_pooling_type="att"):
        super(NTPoolGCN, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.node_dict = node_dict
        self.num_node_types = len(node_dict)
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # self.linears_prediction = nn.ModuleList()
        self.linears_prediction = nn.ModuleDict(
            {
                k: nn.ModuleList()
                for k, _ in node_dict.items()
            }
        )
        self.pools = nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                for k, molst in self.linears_prediction.items():
                        self.linears_prediction[k].append(
                            nn.Linear(in_dim, out_dim))
            else:
                for k, molst in self.linears_prediction.items():
                        self.linears_prediction[k].append(
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

    def alloc_features(self, g, h):
        h_dict = {}
        indices = g.ndata['_ID']

        for k, v in indices.items():
            v = v.squeeze()
            if h[v].ndim == 1:
                h_dict[k] = h[v].unsqueeze(0)
            else:
                h_dict[k] = h[v]

        return h_dict

    def forward(self, g):
        g_homo = dgl.to_homogeneous(g, ndata=['feat', '_ID'])
        g_homo = dgl.add_self_loop(g_homo)
        h_homo = g_homo.ndata['feat']

        h_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                h_homo = self.dropout(h_homo)

            h = self.alloc_features(g, h_homo)
            out_h = {}
            for k, v in h.items():
                if h[k].shape[0] > 0 and h[k].ndim > 1:
                    out_h[k] = self.linears_prediction[k][i](self.pools[i](g, h, ntype=k))
                else:
                    out_h[k] = h[k]

            h_list.append(out_h)

            h_homo = layer(g_homo, h_homo)

        # Taking the sum of predictions scores
        with g.local_scope():
            hg = 0
            count = 0
            # for h in h_list:
            for h in h_list:
                for ntype in g.ntypes:
                    if h[ntype].shape[0] > 0:
                        hg = hg + h[ntype]
                        count += 1
            hg = hg / count

        return hg
