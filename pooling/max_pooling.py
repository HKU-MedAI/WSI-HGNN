from torch import nn

from dgl.readout import max_nodes


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()


    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            graph.ndata['h'] = feat
            if ntype is None:
                readout = max_nodes(graph, 'h')
            else:
                readout = max_nodes(graph, 'h', ntype=ntype)

            return readout