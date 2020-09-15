import os
import sys
sys.path.append('/home/ngr4/project/edge_feat/scripts')
import utils
import pickle
import glob
import numpy as np
import pandas as pd
from scipy import sparse
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


# for edge_feat loading, return attn weights
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
                return_attention_weights=True):
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
    
# REF: pytorch geometric for below data loaders
class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
    """
    def __init__(self, data, num_parts, recursive=False, save_dir=None):
        assert (data.edge_index is not None)

        self.num_parts = num_parts
        self.recursive = recursive
        self.save_dir = save_dir

        self.process(data)

    def process(self, data):
        recursive = '_recursive' if self.recursive else ''
        filename = f'part_data_{self.num_parts}{recursive}.pt'

        path = osp.join(self.save_dir or '', filename)
        if self.save_dir is not None and osp.exists(path):
            data, partptr, perm = torch.load(path)
        else:
            data = copy.copy(data)
            num_nodes = data.num_nodes

            (row, col), edge_attr = data.edge_index, data.edge_attr
            adj = SparseTensor(row=row, col=col, value=edge_attr)
            adj, partptr, perm = adj.partition(self.num_parts, self.recursive)

            for key, item in data:
                if item.size(0) == num_nodes:
                    data[key] = item[perm]

            data.edge_index = None
            data.edge_attr = None
            data.adj = adj

            if self.save_dir is not None:
                torch.save((data, partptr, perm), path)

        self.data = data
        self.perm = perm
        self.partptr = partptr


    def __len__(self):
        return self.partptr.numel() - 1


    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        data = copy.copy(self.data)
        num_nodes = data.num_nodes

        for key, item in data:
            if item.size(0) == num_nodes:
                data[key] = item.narrow(0, start, length)

        data.adj = data.adj.narrow(1, start, length)

        row, col, value = data.adj.coo()
        data.adj = None
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        return data


    def __repr__(self):
        return (f'{self.__class__.__name__}({self.data}, '
                f'num_parts={self.num_parts})')



class ClusterLoader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    Args:
        cluster_data (torch_geometric.data.ClusterData): The already
            partioned data object.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
    """
    def __init__(self, cluster_data, batch_size=1, shuffle=False, **kwargs):
        class HelperDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(cluster_data)

            def __getitem__(self, idx):
                start = int(cluster_data.partptr[idx])
                length = int(cluster_data.partptr[idx + 1]) - start

                data = copy.copy(cluster_data.data)
                num_nodes = data.num_nodes
                for key, item in data:
                    if item.size(0) == num_nodes:
                        data[key] = item.narrow(0, start, length)

                return data, idx

        def collate(batch):
            data_list = [data[0] for data in batch]
            parts: List[int] = [data[1] for data in batch]
            partptr = cluster_data.partptr

            adj = cat([data.adj for data in data_list], dim=0)

            adj = adj.t()
            adjs = []
            for part in parts:
                start = partptr[part]
                length = partptr[part + 1] - start
                adjs.append(adj.narrow(0, start, length))
            adj = cat(adjs, dim=0).t()
            row, col, value = adj.coo()

            data = cluster_data.data.__class__()
            data.num_nodes = adj.size(0)
            data.edge_index = torch.stack([row, col], dim=0)
            data.edge_attr = value

            ref = data_list[0]
            keys = ref.keys
            keys.remove('adj')

            for key in keys:
                if ref[key].size(0) != ref.adj.size(0):
                    data[key] = ref[key]
                else:
                    data[key] = torch.cat([d[key] for d in data_list],
                                          dim=ref.__cat_dim__(key, ref[key]))

            return data

        super(ClusterLoader,
              self).__init__(HelperDataset(), batch_size, shuffle,
                             collate_fn=collate, **kwargs)


# key execution
def get_data(pkl_fname, label, sample, replicate, 
             incl_curvature=False,
             load_attn1=None, load_attn2=None, 
             modelpkl_fname1=None, modelpkl_fname2=None,
             preloadn2v=False,
             out_channels=8, heads=8, negative_slope=0.2, dropout=0.4, 
             verbose=True):
    """From pkl to Pytorch Geometric data object.
    
    Apply to both train/test. 
    
    Arguments:
        pickle path (str): pkl is dict type with requirements, 'k':value desc:
            'X': features (can be sparse)
            'adj': adjacency matrix (can be scipy.sparse)
            'label' (pd.Series): name for y where y is 1d, length X, etc. and pre-encoded
        sample (str): coming from first argument along with job submission to save 
            time consuming features to, grabbing pkl_fnames' path 
        replicate (str): concatenate to sample to load model, not necessary if load_attn is none
        load_attn (str): name of attn from {}{}_{}.format(sample,replicate,load_attn) model. Should match label in datapkl
        preloadn2v (bool): if load_attn is not None, should preloaded node2vec edge attr be loaded?
        modelpkl_fname (str): if load_attn is not None, point to model_pkl file
        (default) args to GAT to load back up edge feats. Only used if load_attn is not None
        
        
    Returns:
        (torch_geometric.Data)
    
    """
    pdfp = os.path.split(pkl_fname)[0]
    
    with open(pkl_fname,'rb') as f :
        datapkl = pickle.load(f)
        f.close()
        
    if load_attn1 is None and load_attn2 is None and not incl_curvature and preloadn2v is None:

        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[label]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
        
    # load all edge_feat
    elif load_attn1 is not None and load_attn2 is not None and incl_curvature and preloadn2v is not None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model

        # second attention
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn2]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname2
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn2 = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(attn2, dtype=float),
                               torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
                               torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
    # only load attn1
    elif load_attn1 is not None and load_attn2 is None and not incl_curvature and preloadn2v is None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
#         F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
#         n2v = utils.node2vec_dot2edge(datapkl['adj'], 
#                                       os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
#                                       preloaded=preloadn2v)
#         edge_attr = torch.cat((torch.tensor(attn, dtype=float),
#                                torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
#                                torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        edge_attr = torch.tensor(attn, dtype=float)    
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')

    # attn2 
    elif load_attn1 is None and load_attn2 is not None and not incl_curvature and preloadn2v is None:
        # second attention
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn2]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname2
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn2 = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
#         F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
#         n2v = utils.node2vec_dot2edge(datapkl['adj'], 
#                                       os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
#                                       preloaded=preloadn2v)
#         edge_attr = torch.cat((torch.tensor(attn, dtype=float),
#                                torch.tensor(attn2, dtype=float),
#                                torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
#                                torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        edge_attr = torch.tensor(attn2, dtype=float)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean 
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
    
    # curvature
    elif load_attn1 is None and load_attn2 is None and incl_curvature and preloadn2v is None:
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[label]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
#         n2v = utils.node2vec_dot2edge(datapkl['adj'], 
#                                       os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
#                                       preloaded=preloadn2v)
#         edge_attr = torch.cat((torch.tensor(attn, dtype=float),
#                                torch.tensor(attn2, dtype=float),
#                                torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
#                                torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        edge_attr = torch.tensor(utils.range_scale(F_e)).reshape(-1,1)
        d = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        del node_features,edge_index,labels,edge_attr
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
        
    # n2v
    elif load_attn1 is None and load_attn2 is None and not incl_curvature and preloadn2v is not None:
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[label]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        # add other edge feats
#         F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
#         edge_attr = torch.cat((torch.tensor(attn, dtype=float),
#                                torch.tensor(attn2, dtype=float),
#                                torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
#                                torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        edge_attr = torch.tensor(utils.range_scale(n2v)).reshape(-1,1)
        d = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        del node_features,edge_index,labels,edge_attr
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
    # attn1 + attn2
    elif load_attn1 is not None and load_attn2 is not None and not incl_curvature and preloadn2v is None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model

        # second attention
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn2]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname2
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn2 = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(attn2, dtype=float)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
    # attn1 + attn2 + n2v
    elif load_attn1 is not None and load_attn2 is not None and not incl_curvature and preloadn2v is not None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model

        # second attention
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn2]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname2
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn2 = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(attn2, dtype=float),
                               torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
          
    # attn1 + attn2 + curvature
    elif load_attn1 is not None and load_attn2 is not None and incl_curvature and preloadn2v is None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model

        # second attention
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn2]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])
        del datapkl # clear space

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname2
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn2 = model(d)

        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(attn2, dtype=float),
                               torch.tensor(utils.range_scale(F_e)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        del model # extra clean
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
    # n2v + curvature
    elif load_attn1 is None and load_attn2 is None and incl_curvature and preloadn2v is not None:
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[label]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
        edge_attr = torch.cat((torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
                               torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        d = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
        del node_features,edge_index,labels,edge_attr
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
            
    # attn1 + curvature
    elif load_attn1 is not None and load_attn2 is None and incl_curvature and preloadn2v is None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model


        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(utils.range_scale(F_e)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)

        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
            
    # attn1 + n2v
    elif load_attn1 is not None and load_attn2 is None and not incl_curvature and preloadn2v is not None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model


        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)
        
        if verbose:
            print('\nData shapes:')
            print(d)
            print('')
            
    # attn1 + n2v + curvature
    elif load_attn1 is not None and load_attn2 is None and incl_curvature and preloadn2v is not None:
        # model for DATA EXTRACTION
        ## TODO: clean this up in some other script or fx

        # load proper label
        node_features = datapkl['X']
        if isinstance(node_features, sparse.csr_matrix):
            node_features = torch.from_numpy(node_features.todense()).float()
        else:
            node_features = torch.from_numpy(node_features).float()
        labels = datapkl[load_attn1]
        if False:
            # assume label_encoding is done in pre-processing steps
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'ctype_labels_encoding.csv'))
        if False:
            # labels as pd.Series
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels) # assumes labels as list
        edge_index,_ = utils.scipysparse2torchsparse(datapkl['adj'])

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        del node_features,edge_index,labels

        # model to grab attn
        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.gat1 = GATConv(d.num_node_features, out_channels=out_channels,
                                    heads=heads, concat=True, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)
                self.gat2 = GATConv(out_channels*heads, d.y.unique().size()[0],
                                    heads=heads, concat=False, negative_slope=negative_slope,
                                    dropout=dropout, bias=True)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x,attn1 = self.gat1(x, edge_index)
                x = F.elu(x)
                x,attn2 = self.gat2(x, edge_index)
                return F.log_softmax(x, dim=1),attn1


        # load edge_feature 
        model = GAT()
        if False:
            # general fname loading?
            model_pkl = glob.glob(os.path.join(pdfp,'*{}{}_{}*.pkl'.format(sample,replicate,load_attn)))[0]
        else:
            model_pkl = modelpkl_fname1
        model.load_state_dict(torch.load(model_pkl, map_location=torch.device('cpu')))
        model.eval()

        logsoftmax_out, attn = model(d)

        del model


        # update labels
        with open(pkl_fname,'rb') as f :
            datapkl = pickle.load(f)
            f.close()
        labels = datapkl[label]
        if False:
            label_encoder = {v:i for i,v in enumerate(labels.unique())}
            labels = labels.map(label_encoder)
            pd.DataFrame(label_encoder,index=[0]).T.to_csv(os.path.join(pdfp,'cond_labels_encoding.csv'))
        if False:
            labels = torch.LongTensor(labels.to_numpy())
        else:
            labels = torch.LongTensor(labels)

        # add other edge feats
        F_e = utils.forman_curvature(datapkl['adj'], verbose=True, plot=False)
        n2v = utils.node2vec_dot2edge(datapkl['adj'], 
                                      os.path.join(pdfp,'{}_n2v_{}.txt'.format(sample.split('_')[0], os.path.split(pkl_fname)[1].split('.p')[0])),
                                      preloaded=preloadn2v)
        edge_attr = torch.cat((torch.tensor(attn, dtype=float),
                               torch.tensor(utils.range_scale(F_e)).reshape(-1,1), 
                               torch.tensor(utils.range_scale(n2v)).reshape(-1,1)),dim=1)
        d = Data(x=d.x, edge_index=d.edge_index, edge_attr=edge_attr, y=labels)

        if verbose:
            print('\nData shapes:')
            print(d)
            print('')

    else:
        print('Can only load edge feats of a specific entry set type. Exiting.')
        exit()
        
    return d
