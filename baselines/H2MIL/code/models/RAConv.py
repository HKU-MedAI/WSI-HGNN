import time
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter
from torch.nn import Parameter, Linear
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,OptTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
class RAConv(MessagePassing):

    _alpha: OptTensor
    t_alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RAConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)
            
        if isinstance(in_channels, int):
            self.t_lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.t_lin_r = self.t_lin_l
            
        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        self.t_att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.t_att_r = Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.t_alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.t_lin_l.weight)
        glorot(self.t_lin_r.weight)
        glorot(self.t_att_l)
        glorot(self.t_att_r)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,node_type,
                size: Size = None,return_attention_weights=None):
        
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None



        # create resolution-level graph
        start_node_type = node_type[edge_index[0]]
        start_x = x[edge_index[0]]   
        
        new_index = start_node_type+edge_index[1]*3

        t_x = scatter(start_x, new_index, dim=0, reduce='mean')
        t_x = torch.cat((x,t_x),0)
        t_x_l = t_x_r = self.t_lin_l(t_x).view(-1, H, C)
        t_alpha_l = (t_x_l * self.t_att_l).sum(dim=-1)
        t_alpha_r = (t_x_r * self.t_att_r).sum(dim=-1)

        start = torch.tensor(sorted(list(set(new_index.cpu().numpy()))),dtype=torch.long)+len(node_type)
        end = torch.tensor(sorted(list(set(new_index.cpu().numpy()))),dtype=torch.long)//3
        new_edge = torch.stack([start, end], dim=0).to(device)

        # resolution-level attention
        t_out = self.propagate(new_edge, x=(t_x_l, t_x_l),soft_index=None, type_edge=None,node_size=None,
                             alpha=(t_alpha_l, t_alpha_r), size=size)
      
        # node-level attention
        out = self.propagate(edge_index, x=(x_l, x_r), soft_index=new_index,type_edge=new_edge[0],node_size=len(node_type),
                             alpha=(alpha_l, alpha_r), size=size)
    
        alpha = self._alpha
        self._alpha = None
        self.t_alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,soft_index:Tensor,type_edge,node_size,
                index: Tensor, ptr: OptTensor,size_i: Optional[int]) -> Tensor:

        
        if self.t_alpha == None:
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, index, ptr, size_i)
            self.t_alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        else:
            size_i = torch.max(soft_index)+1
    
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, soft_index, ptr, size_i)
            self._alpha = alpha

            try:
                alpha = self.t_alpha[torch.where((type_edge-node_size)==soft_index[:,None])[-1]]*alpha
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            except:
                print("shit, something wrong with node num")

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)