import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling

from pooling import AvgPooling, SumPooling


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etype_dict):
        super(HeteroRGCNLayer, self).__init__()
        self.etype_dict = etype_dict
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etype_dict.values()
        })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        new_feat_dict = {k: [] for k in feat_dict.keys()}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Get etype
            local_etype = self.etype_dict[(srctype, etype, dsttype)]
            # Compute W_r * h
            # TODO: Temporary solution for edges that do not have a destination type
            if self.weight[local_etype].in_features == feat_dict[srctype].shape[1]:
                Wh = self.weight[local_etype](feat_dict[srctype])
            else:
                # feat_dict[srctype] = torch.zeros([1, self.weight[local_etype].out_features]).to(G.device)
                # G.nodes[srctype].data['h'] = torch.zeros([1, self.weight[local_etype].out_features]).to(G.device)
                Wh = torch.zeros([1, self.weight[local_etype].out_features]).to(G.device)

            new_feat_dict[srctype].append(Wh)

        for tp, tensors in new_feat_dict.items():
            if tensors == []:
                new_feat_dict[tp] = feat_dict[tp]
            else:
                new_feat_dict[tp] = torch.stack(tensors).mean(0)

        # return the updated node feature dictionary
        return new_feat_dict


class HeteroRGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, etypes, node_dict, graph_pooling_type="sum"):
        super(HeteroRGCN, self).__init__()

        self.node_dict = node_dict
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(HeteroRGCNLayer(hidden_dim, hidden_dim, etypes))
        self.out = nn.Linear(hidden_dim, out_dim)

        self.pools = nn.ModuleList()
        self.linears_prediction = nn.ModuleDict({
            k: nn.ModuleList()
            for k, _ in node_dict.items()
        })

        # Define pooling readout layers
        for layer in range(n_layers + 1):
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

    def forward(self, G, h=None):

        # Read features
        if h is None:
            h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['feat']))
        else:
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](h[ntype]))

        # Propagate and Collect features for pooling
        h_list = []
        for i in range(self.n_layers):
            out_h = {}
            # Perform pooling
            for k, v in h.items():
                if h[k].shape[0] > 0:
                    out_h[k] = self.linears_prediction[k][i](self.pools[i](G, h, ntype=k))
                else:
                    out_h[k] = h[k]
            h_list.append(out_h)

            h = self.layers[i](G, h)

        # Taking the sum of predictions scores
        hg = 0
        for h in h_list:
            for ntype in G.ntypes:
                if h[ntype].shape[0] > 0:
                    hg = hg + h[ntype]

        return hg
