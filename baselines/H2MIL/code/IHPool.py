import os
import torch
import time
import math
import numpy as np
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter
from sklearn.cluster import KMeans
from torch_sparse import SparseTensor
from torch_geometric.nn import LEConv
from torch_geometric.utils import softmax
from scipy.spatial.distance import cdist
from typing import Union, Optional, Callable
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import remove_self_loops
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import add_remaining_self_loops,add_self_loops,sort_edge_index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def euclidean_dist(x, y):
    
    # spatial distance
    x_xy = x[:,0:2]
    y_xy = y[:,0:2]
    
    m = x_xy.size(0)
    n = y_xy.size(0)
    e = x_xy.size(1)  
    
    x1 = x_xy.unsqueeze(1).expand(m, n, e)
    y1 = y_xy.expand(m, n, e)
    dist_xy = (x1 - y1).pow(2).sum(2).float().sqrt()    
    
    # fitness difference
    x_f = x[:,2].unsqueeze(1)
    y_f = y[:,2].unsqueeze(1)
    
    m = x_f.size(0)
    n = y_f.size(0)
    e = x_f.size(1)  
    
    x2 = x_f.unsqueeze(1).expand(m, n, e)
    y2 = y_f.expand(m, n, e)
    dist_f = (x2 - y2).pow(2).sum(2).float().sqrt() 

    return dist_xy+dist_f  


class IHPool(torch.nn.Module):

    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.1,
                 GNN: Optional[Callable] = None, dropout: float = 0.0,
                 select='inter',dis='ou',
                 **kwargs):
        super(IHPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.dropout = dropout
        self.GNN = GNN
        self.select = select
        self.dis = dis

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,**kwargs)
            
        self.weight_1 = Parameter(torch.Tensor(1, in_channels))    
        self.weight_2 = Parameter(torch.Tensor(1, in_channels)) 
        self.nonlinearity = torch.tanh
            
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        size = self.in_channels
        uniform(size, self.weight_1)
        uniform(size, self.weight_2)
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index,node_type,tree,x_y_index,edge_weight=None, batch=None):
        r"""
        x : node feature;
        edge_index :  edge;
        node_type : the resolution-level of each node;
        tree : Correspondence between different level nodes;
        x_y_index : Space coordinates of each node;
        """

        N = x.size(0)
        N_1 = len(torch.where(node_type==1)[0])
        N_2 = len(torch.where(node_type==2)[0])

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)  
      
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            
        x = x.unsqueeze(-1) if x.dim() == 1 else x
           

        #fitness of second resolution-level node
        fitness_1 = (x[torch.where(node_type==1)] * self.weight_1).sum(dim=-1)
        fitness_1 = self.nonlinearity(fitness_1 / self.weight_1.norm(p=2, dim=-1))
        
        #concat spatial information
        x_y_fitness_1 = torch.cat((x_y_index[torch.where(node_type==1)],fitness_1.unsqueeze(1)),-1).to(device)    
        sort_fitness_1,sort_fitness_1_index = torch.sort(fitness_1)

        #select nodes at intervals according to fitness
        if self.select == 'inter': 
            if self.ratio < 1: 
                index_of_threshold_fitness_1 = sort_fitness_1_index[range(0,N_1,int(torch.ceil(torch.tensor(N_1/(N_1*self.ratio)))))].to(device) 
            else:
                if N_1 < self.ratio:
                    index_of_threshold_fitness_1 = sort_fitness_1_index[range(0,N_1,int(torch.ceil(torch.tensor(N_1/N_1))))]  
                else:    
                    index_of_threshold_fitness_1 = sort_fitness_1_index[range(0,N_1,int(torch.ceil(torch.tensor(N_1/(self.ratio)))))]  
                    
        threshold_x_y_fitness_1 = x_y_fitness_1[index_of_threshold_fitness_1].to(device)

        #Clustering according to Euclidean distance
        if self.dis == 'ou': 
            cosine_dis_1 = euclidean_dist(threshold_x_y_fitness_1,x_y_fitness_1).to(device)
            _,cosine_sort_index = torch.sort(cosine_dis_1,0)
            cluster_1 = cosine_sort_index[0]        

        # Calculate the coordinates of the nodes after clustering    
        new_x_y_index_1 = scatter(x_y_index[torch.where(node_type==1)], cluster_1, dim=0, reduce='mean').to(device) 
        new_x_y_index = torch.cat((torch.tensor([[0,0]]).to(device),new_x_y_index_1),0).to(device)        

        #fitness of third resolution-level node
        fitness_2 = (x[torch.where(node_type==2)] * self.weight_2).sum(dim=-1).to(device)
        fitness_2 = self.nonlinearity(fitness_2 / self.weight_2.norm(p=2, dim=-1))
        #concat spatial information
        x_y_fitness_2 = torch.cat((x_y_index[torch.where(node_type==2)],fitness_2.unsqueeze(1)),-1) 
        x_y_index_2 = x_y_index[torch.where(node_type==2)]

        #the nodes to be pooled are depend on the pooling results of corresponding low-resolution nodes
        cluster_2 = torch.tensor([0]*N_2,dtype=torch.long).to(device)
        cluster2_from_1 = cluster_1[tree[torch.where(node_type==2)]-torch.min(tree[torch.where(node_type==2)])]

        new_tree = torch.tensor([-1]).to(device)
        new_tree = torch.cat((new_tree,torch.tensor([0]*len(set(cluster_1.cpu().numpy()))).to(device)),0).to(device)

        #Clustering of each substructure
        for k in range(len(set(cluster_1.cpu().numpy()))):
            #Get the index of each substructure
            index_of_after_cluster = torch.where(cluster2_from_1==torch.tensor(sorted(list(set(cluster_1.cpu().numpy()))))[:,None][k].to(device))
            N_k = len(index_of_after_cluster[0])

            after_cluster_fitness_2 = fitness_2[index_of_after_cluster].to(device)
            after_cluster_x_y_fitness_2 = x_y_fitness_2[index_of_after_cluster].to(device)
            t_x_y_index_2 = x_y_index_2[index_of_after_cluster].to(device)
            
            sort_fitness_2,sort_fitness_2_index = torch.sort(after_cluster_fitness_2)

            #select nodes at intervals according to fitness
            if self.select == 'inter': 
                if self.ratio < 1:
                    index_of_threshold_fitness_2 = sort_fitness_2_index[range(0,N_k,int(torch.ceil(torch.tensor(N_k/(N_k*self.ratio)))))].to(device) 
                else:
                    if N_k == 1:
                        index_of_threshold_fitness_2 = sort_fitness_2_index[range(0,N_k,N_k)].to(device)
                    else:
                        index_of_threshold_fitness_2 = sort_fitness_2_index[range(0,N_k,N_k-1)].to(device)                         
            threshold_x_y_fitness_2 = after_cluster_x_y_fitness_2[index_of_threshold_fitness_2].to(device)

            #Clustering according to Euclidean distance
            if self.dis == 'ou':
                cosine_dis_2 = euclidean_dist(threshold_x_y_fitness_2,after_cluster_x_y_fitness_2).to(device)
                _,cosine_sort_index = torch.sort(cosine_dis_2,0)
                t_cluster_2 = cosine_sort_index[0].to(device)
            
            new_x_y_index = torch.cat((new_x_y_index,scatter(t_x_y_index_2, t_cluster_2, dim=0, reduce='mean'))).to(device)   
            t_cluster_2 += len(set(cluster_2.cpu().numpy()))*2
            
            cluster_2[torch.where(cluster2_from_1==k)] = t_cluster_2 
            new_tree = torch.cat((new_tree,torch.tensor([k+1]*len(set(t_cluster_2.cpu().numpy()))).to(device)))
            
        # Make the clustering results of different levels not repeated
        cluster = torch.tensor(range(N),dtype=torch.long).to(device)
        cluster[torch.where(node_type==0)] = 0
        cluster[torch.where(node_type==1)] = cluster_1+1
        cluster[torch.where(node_type==2)] = cluster_2+len(cluster_1)+100
        # Remove invalid clusters
        cluster = torch.where(cluster[:,None] == torch.tensor(sorted(list(set(cluster.cpu().numpy())))).to(device))[-1].to(device)

        # new node's type
        node_type_0 = torch.tensor([0])
        node_type_1 = torch.tensor([1]*len(set(cluster_1.cpu().numpy())))
        node_type_2 = torch.tensor([2]*len(set(cluster_2.cpu().numpy())))
        node_type = torch.cat((node_type_0,node_type_1,node_type_2),0).to(device)
        
        # X← S^T* X
        x = scatter(x, cluster, dim=0, reduce='mean')        

        # A← S^T* A* S
        A = 0 * torch.ones((N,N)).to(device)
        A[edge_index[0], edge_index[1]] = 1
        A = scatter(A, cluster, dim=0, reduce='add') 
        A = scatter(A, cluster, dim=1, reduce='add') 
        row, col = torch.where(A!=0)     
        edge_index = torch.stack([row, col], dim=0)        
        
        batch = edge_index.new_zeros(x.size(0))
        fitness = torch.cat((torch.tensor([0]).to(device),fitness_1,fitness_2),0)

        return x, edge_index, edge_weight, batch, cluster, node_type,new_tree,fitness,new_x_y_index

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)