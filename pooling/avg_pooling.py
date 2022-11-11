from torch import nn

from dgl.readout import mean_nodes


class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()


    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            graph.ndata['h'] = feat
            if ntype is None:
                readout = mean_nodes(graph, 'h')
            else:
                readout = mean_nodes(graph, 'h', ntype=ntype)

            return readout