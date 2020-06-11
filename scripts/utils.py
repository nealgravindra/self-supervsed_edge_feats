import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
from node2vec import Node2Vec
import networkx as nx
import torch


def scipysparse2torchsparse(x) :
    '''
    Input: scipy csr_matrix
    Returns: torch tensor in experimental sparse format

    REF: Code adatped from [PyTorch discussion forum](https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915>)
    '''
    samples=x.shape[0]
    features=x.shape[1]
    values=x.data
    coo_data=x.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col]) # OR transpose list of index tuples
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return indices,t

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def forman_curvature(adj, verbose=True, plot=False, vectorize=True):
    """Caclualte Forman curvature according to Arijit formula.
    
    NOTE: 4 - \sqrt{w_e}*\sum{\frac{1}{\sqrt{w_{e_{v_1}}}} + \frac{1}{\sqrt{w_{e_{v_2}}}}}.
    NOTE2: Summation is over two indices. If adj is None, simulate on graphs from networkx.
    
    Arguments:
        adj (scipy.sparse.csr)
    
    """
    if adj is not None:
        i_e,j_e = adj.nonzero()
        w_e = np.squeeze(np.array(adj[i_e, j_e]))
        w_v1 = 1
        w_v2 = 1
        F_e = []
        e_counter = 0
        if verbose:
            tic = time.time()
            print('Forman curvature per {} edges'.format(w_e.shape[0]))
                
        if vectorize:
            temp = pd.DataFrame({'w_e':w_e,'i':i_e,'j':j_e})
            w_e_v1 = temp.groupby('i').apply(lambda x: np.sum(1/np.sqrt(x['w_e'])))
            w_e_v2 = temp.groupby('j').apply(lambda x: np.sum(1/np.sqrt(x['w_e'])))
            temp = temp.merge(w_e_v1.rename('w_e_v1'), left_on='i',right_on='i')
            temp = temp.merge(w_e_v2.rename('w_e_v2'), left_on='j', right_on='j')
            F_e = 4 - np.sqrt(temp['w_e']) * (temp['w_e_v1'] + temp['w_e_v2'])

        if not vectorize:
            for i,j in zip(i_e,j_e):
                if verbose:
                    if e_counter % 10000 == 0 and e_counter != 0:
                        toc = time.time() - tic
                        print('  through {:.1f}-% edges in {:.2f}-s'.format(100*e_counter/w_e.shape[0], toc))
                w_e_v1 = w_e[i_e==i]
                w_e_v2 = w_e[j_e==j]

                tshoot = False
                if verbose and e_counter==0 and tshoot:
                    print(i)
                    print(j)
                    print(w_e_v1)
                    print(w_e_v2)

                F_e_ij = 4 - np.sqrt(w_e[e_counter])*(np.sum((w_v1/np.sqrt(w_e_v1))) + np.sum(w_v2/np.sqrt(w_e_v2)))

                F_e.append(F_e_ij)
                e_counter += 1

        if plot:
            fig,ax=plt.subplots(1,1,figsize=(3,2))
            sns.distplot(F_e, bins=10, color='#A6B7BF', ax=ax)
            ax.set_xlabel('Forman curvature')
            ax.set_ylabel('Frequency')
        
    else:
        sims = [nx.adjacency_matrix(nx.erdos_renyi_graph(n=1000, p=0.003, seed=None, directed=False)),
                nx.adjacency_matrix(nx.barabasi_albert_graph(n=1000, m=3)),
                nx.adjacency_matrix(nx.watts_strogatz_graph(n=1000,k=5,p=0.5))
               ]
        for adj in sims:
            F_e = forman_curvature(adj, verbose=True, plot=True)
        F_e = 'Simulated.'
    return F_e

def node2vec_dot2edge(adj, fname, verbose=True, vectorized=True, preloaded=False):
    
    if not preloaded:

        if verbose:
            print('node2vec model fitting...')
            tic = time.time()
            
        # node2vec then dot product 
        node2vec = Node2Vec(
                    graph=nx.from_scipy_sparse_matrix(adj),
                    dimensions=16,
                    walk_length=80,
                    num_walks=10,
                    workers=1)

        n2v = node2vec.fit()

        n2v.wv.save_word2vec_format(fname)

    vectors = pd.read_csv(fname, sep=' ', skiprows=1, header=None)
    vectors = vectors.sort_values(by=0)
    
    if verbose and not preloaded:
        print('  embeddings calculated in {:.1f}-s'.format(time.time() - tic))
    i,j = adj.nonzero()
    edge_attr = []
    v_i = vectors.iloc[i,1:]
    v_j = vectors.iloc[j,1:]
    

    if verbose:
        tic = time.time()
        print('Dot prod per {} edges'.format(v_i.shape[0]))
        
    if vectorized:
        edge_attr = np.einsum('ij,ij->i', v_i, v_j)
        
    else:
        e_counter = 0
        for k in range(v_i.shape[0]):
            if verbose:
                if e_counter % 10000 == 0 and e_counter != 0:
                    toc = time.time() - tic
                    print('  through {:.1f}-% edges in {:.2f}-s'.format(100*e_counter/v_i.shape[0], toc))
            edge_attr.append(np.dot(v_i.iloc[k,:], v_j.iloc[k,:]))
            e_counter += 1
      
    return edge_attr

def range_scale(X, a=-1, b=1):
    # range scale
    return (((b-a)*(X - np.min(X))) / (np.max(X) - np.min(X))) + a

def clip_range(X, a=-1, b=1, min_clip=-50, max_clip=2):
    X[X<min_clip] = min_clip
    X[X>max_clip] = max_clip
    return range_scale(X)


def edge_set_reshape(batch, device='cpu'):
    """Reshape with... pandas :(

    Arguments:
        batch (pytorch geometric data object): must contain edge attributes.

    """
    if device=='cpu':
        temp = pd.DataFrame(batch.edge_attr.detach().numpy(),
                            columns=['e{}'.format(i) for i in range(batch.edge_attr.shape[1])])
    else:
        temp = pd.DataFrame(batch.edge_attr.detach().cpu().numpy(),
                            columns=['e{}'.format(i) for i in range(batch.edge_attr.shape[1])])
    temp['i'] = batch.edge_index[0,:].cpu().numpy() # always just specify cpu?
    j_idx = []
    for i in temp['i'].unique():
        j_idx += list(range((temp['i']==i).sum()))
    temp['j_idx'] = j_idx
    del j_idx
    temp = temp.melt(id_vars=['i','j_idx'],
                     value_vars=['e{}'.format(i) for i in range(batch.edge_attr.shape[1])])
    temp['variable'] = temp['j_idx'].astype(str) + '_' + temp['variable'].astype(str)
    temp = temp.drop(columns='j_idx')
    temp = temp.pivot(index='i',columns='variable',values='value')
    temp.fillna(0,inplace=True)
    new_e = torch.tensor(temp.to_numpy().reshape((temp.shape[0],-1,batch.edge_attr.shape[1])), requires_grad=True)
    return new_e



