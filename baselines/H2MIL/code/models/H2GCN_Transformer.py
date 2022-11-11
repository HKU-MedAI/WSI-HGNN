from torch import nn
import torch

from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import LayerNorm




class H2GCN_Transformer(nn.Module):
    def __init__(self, in_feats, n_hidden, out_classes, drop_out_ratio=0.2, pool1_ratio=0.2, pool2_ratio=4,
                 pool3_ratio=3, mpool_method="global_mean_pool"):
        super(H2GCN_Transformer, self).__init__()
        self.conv1 = RAConv(in_channels=in_feats, out_channels=n_hidden)
        self.conv2 = RAConv(in_channels=n_hidden, out_channels=out_classes)

        self.pool_1 = IHPool(in_channels=out_classes, ratio=pool1_ratio, select='inter', dis='ou')
        self.pool_2 = IHPool(in_channels=out_classes, ratio=pool2_ratio, select='inter', dis='ou')

        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool
        elif mpool_method == "global_att_pool":
            att_net = nn.Sequential(nn.Linear(out_classes, out_classes // 2), nn.ReLU(), nn.Linear(out_classes // 2, 1))
            self.mpool = GlobalAttention(att_net)

        self.lin1 = torch.nn.Linear(out_classes, out_classes // 2)
        self.lin2 = torch.nn.Linear(out_classes // 2, 6)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = LayerNorm(in_feats)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index, node_type, data_id, tree, x_y_index = data.edge_index_tree_8nb, data.node_type, data.data_id, data.node_tree, data.x_y_index
        x_y_index = x_y_index * 2 - 1

        x = self.norm(x)

        x = self.conv1(x, edge_index, node_type)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        x, edge_index_1, edge_weight, batch, cluster_1, node_type_1, tree_1, score_1, x_y_index_1 = self.pool_1(x,
                                                                                                                edge_index,
                                                                                                                node_type=node_type,
                                                                                                                tree=tree,
                                                                                                                x_y_index=x_y_index)
        batch = edge_index_1.new_zeros(x.size(0))
        x1 = self.mpool(x, batch)

        x = self.conv2(x, edge_index_1, node_type_1)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)

        x, edge_index_2, edge_weight, batch, cluster_2, node_type_2, tree_2, score_2, x_y_index_2 = self.pool_2(x,
                                                                                                                edge_index_1,
                                                                                                                node_type=node_type_1,
                                                                                                                tree=tree_1,
                                                                                                                x_y_index=x_y_index_1)
        batch = edge_index_2.new_zeros(x.size(0))
        x2 = self.mpool(x, batch)

        x = x1 + x2

        x = self.lin1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.softmax(x)

        return x, (cluster_1, cluster_2), (node_type_1, node_type_2), (score_1, score_2), (tree_1, tree_2), (
        x_y_index_1, x_y_index_2), (edge_index_1, edge_index_2)