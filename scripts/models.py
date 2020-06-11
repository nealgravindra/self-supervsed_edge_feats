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
    
class SelfAttention_batch(nn.Module):
    def __init__(self, s_max, dim_in=18, dim_out=8, num_heads=2, ln=True, skip=True):
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
    
    
class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels=18, processing_steps=10, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()


    def forward(self, x, batch):
        """"""
        batch_size = s_max
        #batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            batch_len_idx = torch.LongTensor(list(range(x.shape[1])))
            print(x.shape)
            print(q.shape)
            e = (x * q[batch_len_idx]).sum(dim=-1, keepdim=True)
            print(e.shape)
            print(batch)
            a = softmax(e, batch_len_idx, num_nodes=batch_size)
            r = scatter_add(a * x, batch_len_idx, dim=0, dim_size=batch_size)
#             e = (x * q[batch]).sum(dim=-1, keepdim=True)
#             a = softmax(e, batch, num_nodes=batch_size)
#             r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star


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
    
class GAT_transformer_mlp_batch(torch.nn.Module):
    def __init__(self):
        super(GAT_transformer_mlp_batch, self).__init__()
        self.gat1 = GATConv(d.num_node_features, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.gat2 = GATConv(nHeads*nHiddenUnits, out_channels=nHiddenUnits,
                            heads=nHeads, concat=True, negative_slope=alpha,
                            dropout=dropout, bias=True)
        self.transformer = SelfAttention_batch(s_max)
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
        self.set2set = Set2Set()
        self.linear = nn.Linear(72, d.y.unique().size()[0])

    def forward(self, data, edge_feats):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_t = self.set2set(edge_feats, data)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(torch.cat((x,x_t),dim=1))
        return F.log_softmax(x, dim=1)
    