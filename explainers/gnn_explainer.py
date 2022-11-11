import copy
from math import sqrt

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import dgl


class ExplainerTags:
    ORIGINAL_ID = '_explainer_original_id'
    EDGE_MASK = '_explainer_edge_mast'
    NODE_FEATURES = 'feat'


# UDF for message passing. this will execute after all the previous messages, and mask the final massage.
def mask_message(edges):
    edata = edges.data['m']
    emask = edges.data[ExplainerTags.EDGE_MASK].sigmoid().view(-1, 1)
    masked_message = (edata.view(edata.shape[0], -1) * emask).view(edata.shape)
    return {'m': masked_message}


# workaround to hijack the "update_all" function of DGL
class ExplainGraph(dgl.DGLGraph):
    def update_all(self, message_func, reduce_func, apply_node_func=None, etype=None):
        super().apply_edges(message_func)
        super().update_all(mask_message, reduce_func, apply_node_func, etype)


class GNNExplainer:
    # hyper parameters, taken from the original paper
    params = {
        'edge_size': 0.005,
        'feat_size': 0.5,
        'edge_ent': 1.0,
        'feat_ent': 0.1,
        'eps': 1e-15
    }

    def __init__(self, graph: dgl.DGLGraph, model: nn.Module, num_hops: int,
                 epochs: int = 100, lr: float = 0.01,
                 mask_threshold: float = 0.5, edge_size: float = 0.005, feat_size: float = 0.1):
        """
        :param graph: dgl base graph
        :param model: model to explain
        :param num_hops: number of message passing layers in model
        :param epochs: number of epochs to optimize explainer
        :param lr: learning rate
        :param mask_threshold: threshold for hard-mask at return
        """
        self.g: dgl.DGLGraph = graph
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.threshold = mask_threshold
        self.num_hops = num_hops
        self.node_mask = None
        self.params['edge_size'] = edge_size
        self.params['feat_size'] = feat_size
        self.nfeat = ExplainerTags.NODE_FEATURES
        for module in self.model.modules():
            if hasattr(module, '_allow_zero_in_degree'):
                module._allow_zero_in_degree = True

    def __set_masks__(self, g: dgl.DGLGraph):
        """ set masks for edges and nodes """
        num_nodes = g.num_nodes()
        self.node_mask = nn.Parameter(torch.randn(num_nodes).to(self.g.device) * 0.1)

        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * g.num_nodes()))
        g.edata[ExplainerTags.EDGE_MASK] = nn.Parameter(torch.randn(g.num_edges()).to(self.g.device) * std)

    @staticmethod
    def __apply_feature_mask__(feat, mask):
        mask_sig = mask.view(-1, 1).sigmoid()
        return feat * mask_sig

    def __loss__(self, g, node_idx, log_logits, pred_label):
        # prediction loss
        if node_idx is None:  # graph classification
            loss = -log_logits.view(-1)[pred_label]
        else:
            loss = -log_logits[node_idx, pred_label[node_idx]]

        # edge loss
        me = g.edata[ExplainerTags.EDGE_MASK].sigmoid()
        loss = loss + torch.sum(me) * self.params['edge_size']  # edge regularization - subgraph size
        entropy = -me * torch.log(me + self.params['eps']) - (1 - me) * torch.log(1 - me + self.params['eps'])
        loss = loss + self.params['edge_ent'] * entropy.mean()  # edge los: entropy + regularization

        # node features loss
        mn = self.node_mask.sigmoid()
        loss = loss + torch.mean(mn) * self.params['feat_size']  # node feature regularization
        entropy = -mn * torch.log(mn + self.params['eps']) - (1 - mn) * torch.log(1 - mn + self.params['eps'])
        loss = loss + self.params['feat_ent'] * entropy.mean()  # node feature los: entropy + regularization
        # print(log_logits, loss)
        return loss

    def _predict(self, graph, model, node_id, feat_mask=None):
        model.eval()
        feat = graph.ndata[self.nfeat]
        if feat_mask is not None:
            feat = feat * feat_mask
        with torch.no_grad():
            log_logits = model(graph, feat)
            pred_label = log_logits.argmax(dim=-1)

        if node_id is None:   # graph classification
            return log_logits, pred_label
        else:    # node classification
            return log_logits[node_id], pred_label[node_id]

    def _create_subgraph(self, node_idx):
        """ get all nodes that contribute to the computation of node's embedding """
        if node_idx is None:  # graph classification
            sub_g = copy.deepcopy(self.g)
            sub_g.ndata[ExplainerTags.ORIGINAL_ID] = torch.range(0, self.g.num_nodes() - 1, dtype=torch.int).to(self.g.device)
        else:
            nodes = torch.tensor([node_idx])
            eid_list = []
            for _ in range(self.num_hops):
                predecessors, _, eid = self.g.in_edges(nodes, form='all')
                eid_list.extend(eid)
                predecessors = torch.flatten(predecessors).unique()
                nodes = torch.cat([nodes, predecessors])
                nodes = torch.unique(nodes)
            eid_list = list(np.unique(np.array([eid_list])))
            sub_g = dgl.edge_subgraph(self.g, eid_list)  # TODO - handle heterogeneous graphs
            sub_g.ndata[ExplainerTags.ORIGINAL_ID] = sub_g.ndata[dgl.NID]
        return sub_g

    def explain_node(self, node_idx):
        """ main function - calculate explanation """
        # get prediction label
        self.model.eval()
        with torch.no_grad():
            log_logits = self.model(self.g)
            pred_label = log_logits.argmax(dim=-1)

        # create initial subgraph (all nodes and edges that contribute to the explanation)
        subgraph = self._create_subgraph(node_idx)
        if node_idx is None:  # graph classification
            new_node_id = None
            pred_label = pred_label
        else:  # node classification
            new_node_id = np.where(subgraph.ndata[ExplainerTags.ORIGINAL_ID] == node_idx)[0][0]
            pred_label = pred_label[subgraph.ndata[ExplainerTags.ORIGINAL_ID]]

        # "trick" the graph so we can hijack its calls
        original_graph_class = subgraph.__class__
        subgraph.__class__ = ExplainGraph  # super hacky, but i find it elegant in it's own way.

        # set feature and edge masks
        self.__set_masks__(subgraph)
        feat = subgraph.ndata[self.nfeat]
        # move to device
        self.node_mask.to(self.g.device)
        subgraph.to(self.g.device)

        # start optimizing
        optimizer = torch.optim.Adam([self.node_mask, subgraph.edata[ExplainerTags.EDGE_MASK]], lr=self.lr)

        pbar = tqdm(total=self.epochs)
        pbar.set_description('Explaining node {}'.format(node_idx))
        # training loop
        for epoch in range(1, self.epochs + 1):
            h = self.__apply_feature_mask__(feat, self.node_mask)  # soft mask features
            log_logits = self.model(subgraph, h)         # get prediction (will mask edges inside dgl.graph.update_all)
            loss = self.__loss__(subgraph, new_node_id, log_logits, pred_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.detach())
            pbar.update(1)
        pbar.close()
        subgraph.__class__ = original_graph_class

        # # get hard node feature mask and edge mask
        # node_feat_mask = self.feature_mask.detach().sigmoid() > self.threshold
        # edge_mask = subgraph.edata[ExplainerTags.EDGE_MASK].detach().sigmoid().cpu()
        # subgraph.remove_edges(np.where(edge_mask < self.threshold)[0])
        #
        # # remove isolated nodes from subgraph
        # isolated_nodes = np.where((subgraph.in_degrees() == 0).cpu() & (subgraph.out_degrees().cpu() == 0))[0]
        # if node_idx is not None:  # node classification
        #     # don't delete our node in any case..
        #     isolated_nodes = isolated_nodes[isolated_nodes != new_node_id]
        # if sum(isolated_nodes) != subgraph.number_of_nodes():
        #     subgraph.remove_nodes(isolated_nodes)
        # # return subgraph, node_feat_mask

        # Return the feature mask as importance scores
        node_mask = self.node_mask.detach().sigmoid().cpu().numpy()
        return subgraph, node_mask

    def test_explanation(self, node_id, subgraph, feat_mask):
        """ print explanation results- reduced size, and accuracy
        :param node_id: original node id which we explained
        :param subgraph: result from GNNExplainer.explain_node
        :param feat_mask: result from GNNExplainer.explain_node
        """
        # get node prediction for original task
        log_logit_original, label_original = self._predict(self.g, self.model, node_id)
        # mapping to new subgraph node id
        if node_id is None:  # graph classification
            new_node_id = None
        else:   # node classification
            new_node_id = np.where(subgraph.ndata[ExplainerTags.ORIGINAL_ID] == node_id)[0][0]
        # current prediction results
        log_logit, label = self._predict(subgraph, self.model, new_node_id, feat_mask)
        # create subgraph needed for computation, for size reference
        origin_sub_g = self._create_subgraph(node_id)
        # print results
        print("subgraph size before masking (V,E)=({}, {})".format(origin_sub_g.num_nodes(), origin_sub_g.num_edges()))
        print("subgraph size after masking  (V,E)=({}, {})".format(subgraph.num_nodes(), subgraph.num_edges()))
        print("num features before masking: {}".format(torch.numel(feat_mask)))
        print("num features after masking: {}".format(feat_mask.sum()))
        print("log_logits before maseking: {}".format(log_logit_original))
        print("log_logits after masking: {}".format(log_logit))
        print("label before masking: {}".format(label_original))
        print("label after masking: {}".format(label))

    def _visualize(self, subgraph, nlabel_mapping=None, title="", path=None):
        num_classes = int(torch.max(self.g.ndata['label']).item()) + 1
        nx_g = dgl.to_networkx(subgraph, node_attrs=['label', ExplainerTags.ORIGINAL_ID])
        mapping = {i: nx_g.nodes[i][ExplainerTags.ORIGINAL_ID].item() for i in nx_g.nodes.keys()}
        node_labels = [nx_g.nodes[i]['label'].item() for i in nx_g.nodes.keys()]

        cmap = plt.get_cmap('cool', num_classes)
        cmap.set_under('gray')

        node_kwargs = {'node_size': 400, 'cmap': cmap, 'node_color': node_labels, 'vmin': 0, 'vmax': num_classes-1}
        label_kwargs = {'labels': mapping, 'font_size': 10}

        pos = nx.spring_layout(nx_g)
        ax = plt.gca()
        for source, target, data in nx_g.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=1.0,  # max(data['att'], 0.1), # TODO - change for transparent visualization mode
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        nx.draw_networkx_labels(nx_g, pos, **label_kwargs)
        nx.draw_networkx_nodes(nx_g, pos, **node_kwargs)
        if nlabel_mapping is None:
            nlabel_mapping = {i: i for i in range(num_classes)}
        patch_list = [mpatches.Patch(color=cmap(i), label=nlabel_mapping[i]) for i in range(num_classes)]
        ax.legend(handles=patch_list, title="Label", loc=1)
        if title is not None:
            plt.title(title)

        if path is None:
            plt.show()
        else:
            ax.savefig(path)

    def visualize(self, subgraph, node_id, nlabel_mapping=None, title="", path=None):
        """
        visualize explanation
        :param subgraph: result from GNNExplainer.explain_node
        :param node_id: node id used to explain
        :param nlabel_mapping: mapping between node label to text, to be printed in legend
        :param title: optional title for graph
        """
        self._visualize(self._create_subgraph(node_id), nlabel_mapping, title, path)
        self._visualize(subgraph, nlabel_mapping, title, path)
