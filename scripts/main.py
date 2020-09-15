import os
import sys
sys.path.append('/home/ngr4/project/edge_feat/scripts')
import utils
import datetime
import load_data as data
import models
import train
import pickle
import time
import random
import glob
import numpy as np
import pandas as pd
import torch


def main(**kwargs):
    
    # load data
    d = data.get_data(os.path.join(kwargs['pdfp'],kwargs['data_train_pkl']), 
                      label=kwargs['label'], 
                      sample=kwargs['sample'], 
                      replicate=kwargs['replicate'], 
                      incl_curvature=kwargs['incl_curvature'],
                      load_attn1=kwargs['load_attn1'], 
                      load_attn2=kwargs['load_attn2'],
                      modelpkl_fname1=os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname1']),
                      modelpkl_fname2=os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname2']),
                      preloadn2v=kwargs['preloadn2v'],
                      out_channels=8, heads=8, negative_slope=0.2, dropout=0.4)

    if not kwargs['fastmode'] :
        d_val = data.get_data(os.path.join(kwargs['pdfp'],kwargs['data_val_pkl']), 
                              label=kwargs['label'], 
                              sample=kwargs['sample'], 
                              replicate=kwargs['replicate'], 
                              incl_curvature=kwargs['incl_curvature'],
                              load_attn1=kwargs['load_attn1'], 
                              load_attn2=kwargs['load_attn2'],
                              modelpkl_fname1=os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname1']),
                              modelpkl_fname2=os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname2']),
                              preloadn2v=kwargs['preloadn2v'],
                              out_channels=8, heads=8, negative_slope=0.2, dropout=0.4)
        
        
    # data loader for mini-batching
    cd = data.ClusterData(d,num_parts=kwargs['NumParts'])
    cl = data.ClusterLoader(cd,batch_size=kwargs['BatchSize'],shuffle=True)

    if not kwargs['fastmode']:
        cd_val = data.ClusterData(d_val,num_parts=kwargs['NumParts'])
        cl_val = data.ClusterLoader(cd_val,batch_size=kwargs['BatchSize'],shuffle=True)
        
        
    # pick device
    if False : 
        # automate?
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    else :
        device = torch.device(kwargs['Device'])
        
        
    # import model
    models.nHiddenUnits = kwargs['nHiddenUnits']
    models.d = d
    models.nHeads = kwargs['nHeads']
    models.alpha = kwargs['alpha']
    models.dropout = kwargs['dropout']

    if 'transformer' in kwargs['model'] and kwargs['model']!='GAT_transformer_averaged':
        # need to define model based on max number of connections
        s_max = []
        for batch in cl:
            _, n = batch.edge_index[0].unique(return_counts=True)
            s_max.append(n.max().item())
        s_max = 5 + np.max(s_max)
        models.s_max = s_max
        print('s_max: {}'.format(s_max))
        
        if kwargs['model']=='GCN_transformer':
            model = models.GCN_transformer().to(device)
        elif kwargs['model']=='GCN_transformer_mlp':
            model = models.GCN_transformer_mlp().to(device)
        elif kwargs['model']=='GAT_transformer':
            model = models.GAT_transformer().to(device)
        elif kwargs['model']=='GAT_transformer_mlp':
            model = models.GAT_transformer_mlp().to(device)
        elif kwargs['model']=='GAT_transformer_batch':
                model = models.GAT_transformer_batch().to(device)
        elif kwargs['model']=='GAT_transformer_mlp_batch':
            model = models.GAT_transformer_mlp_batch().to(device)
        elif kwargs['model']=='GCN_transformer_mlp_batch':
            model = models.GCN_transformer_mlp_batch().to(device)
    
    # specific model names
    elif kwargs['model']=='GCN_deepset':
        model = models.GCN_deepset().to(device)
    elif kwargs['model']=='GCN_set2set':
        model = models.GCN_set2set().to(device)
    elif kwargs['model']=='GAT_deepset':
        model = models.GAT_deepset().to(device)
    elif kwargs['model']=='GAT_set2set':
        model = models.GAT_set2set().to(device)
    elif kwargs['model']=='GCN':
        model = models.GCN().to(device)
    elif kwargs['model']=='GAT':
        model = models.GAT().to(device)
    elif kwargs['model']=='GINE':
        model = models.GINE().to(device)
    elif kwargs['model']=='edge_cond_conv':
        model = models.edge_cond_conv().to(device)
    elif kwargs['model']=='GAT_transformer_averaged':
        model = models.GAT_transformer_averaged().to(device)
    else:
        print('Re-enter model name. Valid ones are (GAT/GCN)(_transformer)(_mlp)(_batch) for last two with transformer')
        exit()
        
    # set seeds 
    random.seed(kwargs['rs'])
    np.random.seed(kwargs['rs'])
    torch.manual_seed(kwargs['rs'])
    if kwargs['Device'] == 'cuda' :
        torch.cuda.manual_seed(kwargs['rs'])
    
    
    # pick optimizer 
    optimizer = torch.optim.Adagrad(model.parameters(),
                                    lr=kwargs['LR'],
                                    weight_decay=kwargs['WeightDecay'])
    
    
    
    # set train module values
    train.model = model
    train.cl = cl
    train.optimizer = optimizer
    train.device = device
    if not kwargs['fastmode']:
        train.cl_val = cl_val
    train.model_name = kwargs['model']
    train.clip = kwargs['clip']
    train.fastmode = kwargs['fastmode']
    
    # train scheme
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = kwargs['nEpochs'] + 1 # np.inf to avoid problems if small epoch number
    best_epoch = 0
    for epoch in range(kwargs['nEpochs']):
        loss_values.append(train.train(epoch))

        if not kwargs['fastmode']:
            torch.save(model.state_dict(), '{}-{}{}.pkl'.format(epoch,kwargs['sample'],kwargs['replicate']))
            
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == kwargs['patience']:
                break

            files = glob.glob('*-{}{}.pkl'.format(kwargs['sample'],kwargs['replicate']))
            for file in files:
                epoch_nb = int(file.split('-{}{}.pkl'.format(kwargs['sample'],kwargs['replicate']))[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        elif epoch==kwargs['nEpochs']:
            torch.save(model.state_dict(), '{}-{}{}.pkl'.format(epoch, kwargs['sample'],kwargs['replicate']))
            
    files = glob.glob('*-{}{}.pkl'.format(kwargs['sample'],kwargs['replicate']))
    for file in files:
        epoch_nb = int(file.split('-{}{}.pkl'.format(kwargs['sample'],kwargs['replicate']))[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print('\nOptimization Finished! Best epoch: {}'.format(best_epoch))
    print('Training time elapsed: {}-h:m:s'.format(str(datetime.timedelta(seconds=time.time()-t_total))))

    if True:
        # test
        print('\nLoading epoch #{}'.format(best_epoch))
        
        if True:
            # send model to cpu
            if kwargs['model']=='GCN_transformer':
                model = models.GCN_transformer().to(torch.device('cpu'))
            elif kwargs['model']=='GCN_transformer_mlp':
                model = models.GCN_transformer_mlp().to(torch.device('cpu'))
            elif kwargs['model']=='GAT_transformer':
                model = models.GAT_transformer().to(torch.device('cpu'))
            elif kwargs['model']=='GAT_transformer_mlp':
                model = models.GAT_transformer_mlp().to(torch.device('cpu'))
            elif kwargs['model']=='GCN':
                model = models.GCN().to(torch.device('cpu'))
            elif kwargs['model']=='GAT':
                model = models.GAT().to(torch.device('cpu'))
            elif kwargs['model']=='GAT_transformer_batch':
                model = models.GAT_transformer_batch().to(device)
            elif kwargs['model']=='GAT_transformer_mlp_batch':
                model = models.GAT_transformer_mlp_batch().to(device)
            elif kwargs['model']=='GCN_transformer_mlp_batch':
                model = models.GCN_transformer_mlp_batch().to(device)
            elif kwargs['model']=='GCN_deepset':
                model = models.GCN_deepset().to(device)
            elif kwargs['model']=='GCN_set2set':
                model = models.GCN_set2set().to(device)
            elif kwargs['model']=='GAT_deepset':
                model = models.GAT_deepset().to(device)
            elif kwargs['model']=='GAT_set2set':
                model = models.GAT_set2set().to(device)
            elif kwargs['model']=='GINE':
                model = models.GINE().to(device)
            elif kwargs['model']=='edge_cond_conv':
                model = models.edge_cond_conv().to(device)
            elif kwargs['model']=='GAT_transformer_averaged':
                model = models.GAT_transformer_averaged().to(device)
                
        model.load_state_dict(torch.load('{}-{}{}.pkl'.format(best_epoch,kwargs['sample'],kwargs['replicate']), 
                                         map_location=torch.device('cpu')))
        
        train.test_fname = os.path.join(kwargs['pdfp'],kwargs['data_test_pkl'])
        train.label = kwargs['label']
        train.sample = kwargs['sample']
        train.replicate = kwargs['replicate']
        train.incl_curvature = kwargs['incl_curvature']
        train.load_attn1 = kwargs['load_attn1']
        train.load_attn2 = kwargs['load_attn2']
        train.modelpkl_fname1 = os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname1'])
        train.modelpkl_fname2 = os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname2'])
        train.preloadn2v = kwargs['preloadn2v']
        train.model = model
        train.batch_size = kwargs['BatchSize']
        train.num_parts = kwargs['NumParts']
        
        train.compute_test()
    

if __name__ == '__main__':
    
    params = {

        ################################################################################
        # hyperparams
        ################################################################################
        'pdfp':'/home/ngr4/project/covid_lung/data/processed/', #'/home/ngr4/project/sccovid/data/processed/' #'/home/ngr4/project/covid_lung/data/processed/' #'/home/ngr4/project/scni/data/processed_200108/' #'/home/ngr4/project/perturb-seq/data/processed/'
        'data_train_pkl':'liao_train_200529.pkl', #'hbec_train_200529.pkl' #'liao_train_200529.pkl' #'scni_train_200604.pkl' # 'norman_train.pkl'
        'data_val_pkl':'liao_val_200529.pkl', #'hbec_val_200529.pkl' #'liao_val_200529.pkl' #'scni_val_200604.pkl', # 'norman_val.pkl'
        'data_test_pkl':'liao_test_200529.pkl', #'hbec_test_200529.pkl' #'liao_test_200529.pkl' #'scni_test_200604.pkl', # 'norman_test.pkl'
        'sample':'liao_cond', #'hbec_it' #'liao_cond' #'scni_ms',
        'replicate':sys.argv[1],
        'label':'ycondition', #'yinftime' #'ycondition' #'yms', #'yguide'
        'model':'GAT_transformer_averaged', # GAT/GCN(_transformer)(_mlp)(_batch)
        'incl_curvature':True, # (True/False) include curvature as an edge feature 
        'load_attn1':'yctype', # if load_attn is not None, give label for attn to load 
        'modelpkl_fname1':'887-liao_ctype_gat1.pkl', #'338-hbec_ctype_gat1.pkl' #'887-liao_ctype_gat1.pkl' #'1973-scni_ctype1.pkl', # if load_attn is not None, indicate name of model pkl in pdfp
        'load_attn2':'ybatch',
        'modelpkl_fname2':'1988-liao_batch_gat1.pkl', #'1148-hbec_batch_gat1.pkl' #'1988-liao_batch_gat1.pkl' #'1996-scni_batch_gat1.pkl',
        'preloadn2v':True, # (bool or None) True/False: pre-load, if None, don't include n2v as edge feat

        'BatchSize':256,
        'NumParts':5000, # num sub-graphs
        'Device':'cpu',
        'LR':0.001, # learning rate
        'WeightDecay':5e-4,
        'fastmode':False, # if `fastmode=False`, report validation
        'nHiddenUnits':8,
        'nHeads':8, # number of attention heads
        'nEpochs':2000,
        'dropout':0.4, # applied to all GAT layers
        'alpha':0.2, # alpha for leaky_relu
        'patience':100, # epochs to beat
        'clip':None, # set `clip=1` to turn on gradient clipping
        'rs':random.randint(1,1000000), # random_seed
        ################################################################################

    }

    
    tic = time.time()
    main(**params)
    print('\n\n... full run-through in {}-h:m:s'.format(str(datetime.timedelta(seconds=time.time()-tic))))