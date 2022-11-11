import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling, GlobalAttentionPooling

# from utils import readout


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, input_dim, hidden_dim,
                 out_dim, num_layers, num_mlp_layers,
                 final_dropout, graph_pooling_type="sum",
                 neighbor_pooling_type="mean", learn_eps=True):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.layers = torch.nn.ModuleList()

        self.graph_pooling_type = graph_pooling_type

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.layers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))

        self.drop = nn.Dropout(final_dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(num_layers + 1):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, out_dim))
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
                    gate_nn = torch.nn.Linear(input_dim, 1)
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

        # with g.local_scope():
        #     # Pool the readouts
        #
        #     g.ndata['h'] = h
        #
        #     # Calculate graph representation by average readout.
        #     hg = dgl.mean_nodes(g, 'h')
        #     out = self.classify(hg)

        out = torch.stack(h_list).sum(0)

        return out