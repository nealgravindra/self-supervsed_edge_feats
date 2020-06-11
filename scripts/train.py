import os
import sys
sys.path.append('/home/ngr4/project/edge_feat/scripts')
import utils
import load_data as data
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(epoch):
    t = time.time()
    epoch_loss = []
    epoch_acc = []
    epoch_acc_val = []
    epoch_loss_val = []

    model.train()
    for batch in cl :
        batch = batch.to(device)
        optimizer.zero_grad()
        if 'transformer' in model_name:
            output = model(batch,utils.edge_set_reshape(batch).float().to(device))
        else:
            output = model(batch)
        # y_true = batch.y.to(device)
        loss = F.nll_loss(output, batch.y)
        loss.backward()
        if clip is not None :
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()
        epoch_loss.append(loss.item())
        epoch_acc.append(utils.accuracy(output, batch.y).item())
    if not fastmode :
        for batch_val in cl_val:
            batch_val = batch_val.to(device)
            model.eval()
            if 'transformer' in model_name:
                output = model(batch_val, utils.edge_set_reshape(batch_val).float().to(device))
            else:
                output = model(batch_val)
            loss_val = F.nll_loss(output, batch_val.y)
            epoch_acc_val.append(utils.accuracy(output,batch_val.y).item())
            epoch_loss_val.append(loss_val.item())
        print('Epoch {}\t<loss>={:.4f}\t<acc>={:.4f}\t<loss_val>={:.4f}\t<acc_val>={:.4f}\tin {:.2f} s'.format(epoch,np.mean(epoch_loss),np.mean(epoch_acc),np.mean(epoch_loss_val),np.mean(epoch_acc_val),time.time() - t))
        return np.mean(epoch_loss_val)
    else :
        print('Epoch {}\t<loss>={:.4f}\t<acc>={:.4f}\tin {:.2f}-s'.format(epoch,np.mean(epoch_loss),np.mean(epoch_acc),time.time()-t))
        return np.mean(epoch_loss)


# TODO: move this to separate eval script (or nb)
def compute_test():
    d_test = data.get_data(test_fname, label, 
                           sample, replicate, 
                           load_attn1, load_attn2,
                           modelpkl_fname1, modelpkl_fname2,
                           preloadn2v, out_channels=8, 
                           heads=8, negative_slope=0.2, 
                           dropout=0.4)
    
    if False:
        # batch it to keep on GPU
        model.to(device) # assumes cuda specified
        
        cd_test = data.ClusterData(d_test,num_parts)
        cl_test = data.ClusterLoader(cd_test,batch_size,shuffle=True)
        
        batch_loss_test = []
        batch_acc_test = []
        
        for batch_test in cl_test:
            batch_test = batch_test.to(device)
            model.eval()
            if 'transformer' in model_name:
                output = model(batch_test, utils.edge_set_reshape(batch_test).float().to(device))
            else:
                output = model(batch_test)
            loss_test = F.nll_loss(output, batch_test.y)
            batch_acc_test.append(utils.accuracy(output,batch_test.y).item())
            batch_loss_test.append(loss_test.item())
        print('Test set results:')
        print('  <loss>_bacth={:.4f}'.format(np.mean(batch_loss_test)))
        print('  <acc>_batch ={:.4f}'.format(np.mean(batch_acc_test)))
    else:
        # keep on cpu
        model.eval()
        if 'transformer' in model_name:
            output = model(d_test, utils.edge_set_reshape(d_test).float())
        else:
            output = model(d_test)
        loss_test = F.nll_loss(output, d_test.y).item()
        acc_test = utils.accuracy(output,d_test.y).item()
        print('Test set results:')
        print('  loss: {:.4f}'.format(loss_test))
        print('  accuracy: {:.4f}'.format(acc_test))
            
        
if __name__ == '__main__':
    # TODO: put in eval file where attn is also loaded
    print('\nLoading epoch #{}'.format(best_epoch))

    test_fname = os.path.join(kwargs['pdfp'],kwargs['data_test_pkl'])
    label = kwargs['label']
    sample = kwargs['sample']
    replicate = kwargs['replicate']
    load_attn = kwargs['load_attn']
    preloadn2v = kwargs['preloadn2v']
    modelpkl_fname = os.path.join(kwargs['pdfp'],kwargs['modelpkl_fname'])
    model = model
    batch_size = kwargs['BatchSize']
    num_parts = kwargs['NumParts']
    
    model.load_state_dict(torch.load('{}-{}{}.pkl'.format(best_epoch,kwargs['sample'],kwargs['replicate']), 
                                     map_location=torch.device('cpu')))

    compute_test()