import torch
import torch.nn as nn

import dgl

"""
Gem explainer from https://arxiv.org/pdf/2104.06643.pdf
Causal contribution can be computed by the loss difference excluding the node
"""


class HetGemExplainer:
    def __init__(self, graph: dgl.DGLHeteroGraph, model: nn.Module, label):

        temp = dgl.to_homogeneous(graph, ndata=['feat', 'patches_coords'], edata=['sim'])
        temp.edata['_TYPE'] *= int(0)
        ntypes = [str(t.item()) for t in torch.unique(temp.ndata['_TYPE'])]
        self.graph = dgl.to_heterogeneous(temp, ntypes=ntypes, etypes=['pos'])

        self.label = label
        self.gnn = model

        self.loss_fcn = nn.CrossEntropyLoss()

    def explain_node(self):
        pred = self.gnn(self.graph)
        loss = self.loss_fcn(pred, self.label)
        node_mask = {t: torch.zeros(self.graph.ndata['feat'][t].shape[0]) for t in self.graph.ntypes}

        for ntp, ts in self.graph.ndata['feat'].items():
            for nidx in range(ts.shape[0]):
                # Temporary solution for not none batch_num_nodes
                self.graph._batch_num_nodes = None

                alt_g = dgl.remove_nodes(self.graph, torch.tensor([nidx]).to(self.graph.device), ntype=ntp)
                pred_alt = self.gnn(alt_g)
                loss_alt = self.loss_fcn(pred_alt, self.label)
                delta = loss - loss_alt
                node_mask[ntp][nidx] = delta

        return node_mask