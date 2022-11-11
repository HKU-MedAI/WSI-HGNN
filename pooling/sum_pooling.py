from torch import nn

from dgl.readout import sum_nodes


class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            graph.ndata['h'] = feat
            if ntype is None:
                readout = sum_nodes(graph, 'h')
            else:
                readout = sum_nodes(graph, 'h', ntype=ntype)

            return readout