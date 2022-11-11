from tqdm import tqdm

import torch
import torch.nn as nn

import dgl

"""
Gem explainer from https://arxiv.org/pdf/2104.06643.pdf
Causal contribution can be computed by the loss difference excluding the node
"""


class GemExplainer:
    def __init__(self, graph: dgl.DGLGraph, model: nn.Module, label):

        self.graph = graph
        self.label = label
        self.gnn = model

        self.loss_fcn = nn.CrossEntropyLoss()

    def explain_node(self):
        # Apply temperature scaling
        temp = 40
        pred = self.gnn(self.graph)
        loss = self.loss_fcn(pred / temp, self.label)
        node_mask = torch.zeros(self.graph.num_nodes())
        batch_size = 10

        for b in tqdm(range(0, self.graph.num_nodes(), batch_size)):
            b = b // batch_size
            start = b * batch_size
            end = min((b+1) * batch_size, self.graph.num_nodes())

            # Temporary solution for not none batch_num_nodes
            self.graph._batch_num_nodes = None
            alt_graphs = [dgl.remove_nodes(self.graph, torch.tensor([nid]).to(self.graph.device))
                          for nid in range(start, end)]
            bg = dgl.batch(alt_graphs)

            with torch.no_grad():
                pred_alt = self.gnn(bg)

            lf = nn.CrossEntropyLoss(reduction='none')
            lb = torch.ones((end - start,), dtype=torch.int).to(self.graph.device) * self.label

            delta = lf((pred - pred_alt), lb)

            node_mask[start:end] = delta

        node_mask = node_mask.detach().cpu().numpy()
        node_mask = (node_mask - node_mask.min()) / (node_mask.max() - node_mask.min())
        return node_mask
