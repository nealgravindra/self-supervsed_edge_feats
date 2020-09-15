import os
import sys
sys.path.append('/home/ngr4/project/edge_feat/scripts')
import pickle
import time
import random
import glob
import math
import numpy as np
import pandas as pd
from scipy import sparse
from node2vec import Node2Vec
import networkx as nx

from typing import List
import copy
import os.path as osp

import torch
import torch.utils.data
from torch_sparse import matmul as torch_sparse_matmul # non-standard 
from torch_sparse import SparseTensor, cat
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.init import xavier_uniform_ as glorot
from torch.nn.init import zeros_ as zeros
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add

from typing import Union, Tuple, Callable
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor



# transformer 
class Encoder(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, s_max, d_model, num_heads, ln=False, skip=True):
        super(Encoder, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.skip = skip
        self.s_max = s_max
        #Maximum set size
        self.d_model = d_model
        self.fc_q = nn.Linear(dim_Q, d_model)
        self.fc_k = nn.Linear(dim_K, d_model)
        self.fc_v = nn.Linear(dim_K, d_model)
        if ln:
            self.ln0 = nn.LayerNorm(d_model)
            self.ln1 = nn.LayerNorm(d_model)
        #This is the classic pointwise feedforward in "Attention is All you need"
        self.ff = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.ReLU(),
        nn.Linear(4 * d_model, d_model))
        # I have experimented with just a smaller version of this
       # self.fc_o = nn.Linear(d_model,d_model)
        self.fc_rep = nn.Linear(s_max, 1)
        
    # number of heads must divide output size = d_model
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.d_model // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(-2,-1))/math.sqrt(self.d_model), dim=-1)
        A_1 = A.bmm(V_) 
        O = torch.cat((A_1).split(Q.size(0), 0), 2) # return THIS O, e.g., transformer_attn
        O = torch.cat((Q_ + A_1).split(Q.size(0), 0), 2) if getattr(self, 'skip', True) else \
             torch.cat((A_1).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)

        # For the classic transformers paper it is
        O = O + self.ff(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        O = O.transpose(-2,-1)
        O = F.pad(O, (0, self.s_max- O.shape[-1]), 'constant', 0)
        O = self.fc_rep(O)
        O = O.squeeze() 
        return O

class SelfAttention(nn.Module):
    def __init__(self, s_max, dim_in=10, dim_out=6, num_heads=2, ln=True, skip=True):
        super(SelfAttention, self).__init__()
        self.Encoder = Encoder(dim_in, dim_in, dim_in, s_max, dim_out, num_heads, ln=ln, skip=skip)
    def forward(self, X):
        return self.Encoder(X, X)
    
# original
# class SelfAttention_batch(nn.Module):
#     def __init__(self, s_max, dim_in=18, dim_out=8, num_heads=2, ln=True, skip=True):
#         super(SelfAttention_batch, self).__init__()
#         self.Encoder = Encoder(dim_in, dim_in, dim_in, s_max, dim_out, num_heads, ln=ln, skip=skip)
#     def forward(self, X):
#         return self.Encoder(X, X)

class SelfAttention_batch(nn.Module):
    def __init__(self, s_max, dim_in, 
                 dim_out=8, 
                 num_heads=2, 
                 ln=True, 
                 skip=True):
        super(SelfAttention_batch, self).__init__()
        self.Encoder = Encoder(dim_in, dim_in, dim_in, s_max, dim_out, num_heads, ln=ln, skip=skip)
    def forward(self, X):
        return self.Encoder(X, X)
    


# full model (GATConv from PyTorch Geometric)
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None,
                return_attention_weights=False):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x,
                             return_attention_weights=return_attention_weights)

        if return_attention_weights:
            alpha, self.alpha = self.alpha, None
            return out, alpha
        else:
            return out


    def message(self, edge_index_i, x_i, x_j, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
    
# class Set2Set(torch.nn.Module):
#     r"""The global pooling operator based on iterative content-based attention
#     from the `"Order Matters: Sequence to sequence for sets"
#     <https://arxiv.org/abs/1511.06391>`_ paper

#     .. math::
#         \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

#         \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

#         \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

#         \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

#     where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
#     the dimensionality as the input.

#     Args:
#         in_channels (int): Size of each input sample.
#         processing_steps (int): Number of iterations :math:`T`.
#         num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
#             :obj:`num_layers=2` would mean stacking two LSTMs together to form
#             a stacked LSTM, with the second LSTM taking in outputs of the first
#             LSTM and computing the final results. (default: :obj:`1`)
#     """

#     def __init__(self, in_channels=18, processing_steps=10, num_layers=1):
#         super(Set2Set, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = 2 * in_channels
#         self.processing_steps = processing_steps
#         self.num_layers = num_layers

#         self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
#                                   num_layers)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lstm.reset_parameters()


#     def forward(self, x, batch):
#         """"""
#         batch_size = s_max
#         #batch_size = batch.max().item() + 1

#         h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
#              x.new_zeros((self.num_layers, batch_size, self.in_channels)))
#         q_star = x.new_zeros(batch_size, self.out_channels)

#         for i in range(self.processing_steps):
#             q, h = self.lstm(q_star.unsqueeze(0), h)
#             q = q.view(batch_size, self.in_channels)
#             batch_len_idx = torch.LongTensor(list(range(x.shape[1])))
#             print(x.shape)
#             print(q.shape)
#             e = (x * q[batch_len_idx]).sum(dim=-1, keepdim=True)
#             print(e.shape)
#             print(batch)
#             a = softmax(e, batch_len_idx, num_nodes=batch_size)
#             r = scatter_add(a * x, batch_len_idx, dim=0, dim_size=batch_size)
# #             e = (x * q[batch]).sum(dim=-1, keepdim=True)
# #             a = softmax(e, batch, num_nodes=batch_size)
# #             r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
#             q_star = torch.cat([q, r], dim=-1)

#         return q_star

class Set2Set(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_fn=nn.ReLU, num_layers=1):
        '''
        Args:
            input_dim: input dim of Set2Set. 
            hidden_dim: the dim of set representation, which is also the INPUT dimension of 
                the LSTM in Set2Set. 
                This is a concatenation of weighted sum of embedding (dim input_dim), and the LSTM
                hidden/output (dim: self.lstm_output_dim).
        '''
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if hidden_dim <= input_dim:
            print('ERROR: Set2Set output_dim should be larger than input_dim')
        # the hidden is a concatenation of weighted sum of embedding and LSTM output
        self.lstm_output_dim = hidden_dim - input_dim
        self.lstm = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)
        # convert back to dim of input_dim
       # self.pred = nn.Linear(hidden_dim, input_dim)
        self.pred = nn.Linear(hidden_dim,8)
        self.act = act_fn()
        
    def forward(self, embedding):
        '''
        Args:
            embedding: [batch_size x n x d] embedding matrix
        Returns:
            aggregated: [batch_size x d] vector representation of all embeddings
        '''
        batch_size = embedding.size()[0]
        n = embedding.size()[1]
        hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_output_dim),
                  torch.zeros(self.num_layers, batch_size, self.lstm_output_dim))
        q_star = torch.zeros(batch_size, 1, self.hidden_dim)
        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, hidden = self.lstm(q_star, hidden)
            # e: batch_size x n x 1
            e = embedding @ torch.transpose(q, 1, 2)
            a = nn.Softmax(dim=1)(e)
            r = torch.sum(a * embedding, dim=1, keepdim=True)
            q_star = torch.cat((q, r), dim=2)
        q_star = torch.squeeze(q_star, dim=1)
        out = self.act(self.pred(q_star))
        return out


class GAT_transformer(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(70, d.y.unique().size()[0],
                            heads=nHeads, concat=False, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention(s_max)

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.transformer(edge_feats)
        x = self.gat2(torch.cat((x,x_t),dim=1), edge_index)
        return F.log_softmax(x, dim=1)
    
class GAT_transformer_batch(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer_batch, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(72, d.y.unique().size()[0],
                            heads=nHeads, concat=False, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention_batch(s_max)

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.transformer(edge_feats)
        x = self.gat2(torch.cat((x,x_t),dim=1), edge_index)
        return F.log_softmax(x, dim=1)

    
class GAT_transformer_mlp(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer_mlp, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention(s_max)
        self.linear = nn.Linear(70, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.transformer(edge_feats)
        x = self.gat2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
# class GAT_transformer_mlp_batch(torch.nn.Module):
#     def __init__(self):
#         super(GAT_transformer_mlp_batch, self).__init__()
#         self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
#                             heads=nHeads, concat=True, negative_slope=alpha,
#                             dropout=dropout, bias=True)
#         self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
#                             heads=nHeads, concat=True, negative_slope=alpha,
#                             dropout=dropout, bias=True)
#         self.transformer = SelfAttention_batch(s_max)
#         self.linear = nn.Linear(72, d.y.unique().size()[0])

#     def forward(self, data, edge_feats):
#         x, edge_index = data.x, data.edge_index
#         x = self.gat1(x, edge_index)
#         x = F.elu(x)
#         x_t = self.transformer(edge_feats)
#         x = self.gat2(x, edge_index)
#         x = self.linear(torch.cat((x,x_t),dim=1))
#         return F.log_softmax(x, dim=1)

class GAT_transformer_mlp_batch(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer_mlp_batch, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention_batch(s_max, d.edge_attr.shape[1])
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.transformer(edge_feats)
        x = self.gat2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHiddenUnits*nHeads, d.y.unique().size()[0],
                            heads=nHeads, concat=False, negative_slope=alpha,
                            dropout=dropout, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64, d.y.unique().size()[0])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GCN_transformer(torch.nn.Module):
    def __init__(self):
        super(GCN_transformer, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64+6, d.y.unique().size()[0])
        self.transformer = SelfAttention(s_max)

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.transformer(edge_feats)
        x = F.dropout(x, training=self.training)
        x = self.conv2(torch.cat((x,x_t),dim=1), edge_index)
        return F.log_softmax(x, dim=1)
    
class GCN_transformer_mlp(torch.nn.Module):
    def __init__(self):
        super(GCN_transformer_mlp, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.transformer = SelfAttention(s_max)
        self.linear = nn.Linear(70, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.transformer(edge_feats)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
class GCN_transformer_mlp_batch(torch.nn.Module):
    def __init__(self):
        super(GCN_transformer_mlp_batch, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.transformer = SelfAttention_batch(s_max)
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.transformer(edge_feats)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
class GCN_set2set(torch.nn.Module):
    def __init__(self):
        super(GCN_set2set, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.set2set = Set2Set(18, 36)
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.set2set(edge_feats)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
class DeepSet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepSet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi = nn.Sequential(
                nn.Linear(input_dim, 2*input_dim),
                nn.Linear(2*input_dim, 4*input_dim))
        self.rho = nn.Sequential(
                nn.Linear(4*input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32,output_dim))
    def forward(self,x):
        x = self.phi(x)
        x = torch.sum(x,axis=1)
        x = self.rho(x)
        return x
    
class GCN_deepset(torch.nn.Module):
    def __init__(self):
        super(GCN_deepset, self).__init__()
        self.conv1 = GCNConv(d.num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.deepset = DeepSet(18, 8)
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.deepset(edge_feats)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)

class GAT_deepset(torch.nn.Module):
    def __init__(self):
        super(GAT_deepset, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.deepset = DeepSet(18, 8)
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.deepset(edge_feats)
        x = self.gat2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
class GAT_set2set(torch.nn.Module):
    def __init__(self):
        super(GAT_set2set, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.set2set = Set2Set(18, 36)
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.set2set(edge_feats)
        x = self.gat2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = True,
                 **kwargs):
        super(GINEConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
    
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
    


class GINE(torch.nn.Module):
    def __init__(self):
        super(GINE, self).__init__()
        self.linear = nn.Linear(d.x.shape[1], d.edge_attr.shape[1])
        self.gine_conv1 = GINEConv(torch.nn.Linear(d.edge_attr.shape[1], d.edge_attr.shape[1]))
        self.gine_conv2 = GINEConv(torch.nn.Linear(d.edge_attr.shape[1], d.y.unique().size()[0]))

    def forward(self, data):
        x, edge_index, edge_feats = data.x, data.edge_index, data.edge_attr
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)
        x = self.linear(x)
        x = F.relu(x)
        x = self.gine_conv1(x, edge_index, edge_feats)
        x = F.relu(x)
        x = self.gine_conv2(x, edge_index, edge_feats)        
        return F.log_softmax(x, dim=1)
    
class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    
class edge_cond_conv(torch.nn.Module):
    def __init__(self):
        super(edge_cond_conv, self).__init__()
        self.linear1 = nn.Linear(d.x.shape[1], 128)
        self.nn_conv1 = NNConv(128, 64, nn.Linear(d.edge_attr.shape[1], 128*64))
        self.nn_conv2 = NNConv(64, d.y.unique().size()[0], nn.Linear(d.edge_attr.shape[1], 64*d.y.unique().size()[0]))
        

    def forward(self, data):
        x, edge_index, edge_feats = data.x, data.edge_index, data.edge_attr
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.nn_conv1(x, edge_index, edge_feats)
        x = F.relu(x)
        x = self.nn_conv2(x, edge_index, edge_feats)
        return F.log_softmax(x, dim=1)
    
    
# transformer 
class Encoder_averaged(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, d_model, num_heads, ln=False, skip=True):
        super(Encoder_averaged, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.skip = skip
        #Maximum set size
        self.d_model = d_model
        self.fc_q = nn.Linear(dim_Q, d_model)
        self.fc_k = nn.Linear(dim_K, d_model)
        self.fc_v = nn.Linear(dim_K, d_model)
        if ln:
            self.ln0 = nn.LayerNorm(d_model)
            self.ln1 = nn.LayerNorm(d_model)
        #This is the classic pointwise feedforward in "Attention is All you need"
        self.ff = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.ReLU(),
        nn.Linear(4 * d_model, d_model))
        # I have experimented with just a smaller version of this
       # self.fc_o = nn.Linear(d_model,d_model)
#         self.fc_rep = nn.Linear(s_max, 1)
        
    # number of heads must divide output size = d_model
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.d_model // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(-2,-1))/math.sqrt(self.d_model), dim=-1)
        A_1 = A.bmm(V_) 
        O = torch.cat((A_1).split(Q.size(0), 0), 2) # return THIS O, e.g., transformer_attn
        O = torch.cat((Q_ + A_1).split(Q.size(0), 0), 2) if getattr(self, 'skip', True) else \
             torch.cat((A_1).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)

        # For the classic transformers paper it is
        O = O + self.ff(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
#         O = O.transpose(-2,-1)
#         O = F.pad(O, (0, self.s_max- O.shape[-1]), 'constant', 0)
#         O = self.fc_rep(O)
#         O = O.squeeze() 
        O = torch.mean(O, dim=1)
        return O

class SelfAttention_averaged(nn.Module):
    def __init__(self, dim_in, dim_out=8, num_heads=2, ln=True, skip=True):
        super(SelfAttention_averaged, self).__init__()
        self.Encoder = Encoder_averaged(dim_in, dim_in, dim_in, dim_out, num_heads, ln=ln, skip=skip)
    def forward(self, X):
        return self.Encoder(X, X)
    
# original
# class SelfAttention_batch(nn.Module):
#     def __init__(self, s_max, dim_in=18, dim_out=8, num_heads=2, ln=True, skip=True):
#         super(SelfAttention_batch, self).__init__()
#         self.Encoder = Encoder(dim_in, dim_in, dim_in, s_max, dim_out, num_heads, ln=ln, skip=skip)
#     def forward(self, X):
#         return self.Encoder(X, X)

class GAT_transformer_averaged(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer_averaged, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention_averaged(d.edge_attr.shape[1])
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x_t = self.transformer(edge_feats)
        x = self.gat2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)