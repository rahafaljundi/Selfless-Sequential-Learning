

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import scipy.stats as stats
import sys

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
#sys.path.append('../my_utils')
#from ImageFolderTrainVal import *
import pdb

#cov with zero mean assumptions


class SLNI_Module(nn.Module):
    def __init__(self, module, sparcified_layers):
        super(SLNI_Module, self).__init__()
        self.module = module
        self.sparcified_layers=sparcified_layers
        self.neuron_omega=False
        self.scale=0
        self.squash='exp'#squashing function exp or sigmoid
        self.min=1#min or multiply of two neuron importance
        self.abs=True
        self.divide_by_tasks=False
    def forward(self, x):
        decov_loss =0
        sub_index=0
        
        for name, module in self.module._modules.items():
        
                for namex, modulex in module._modules.items():
                    
                    x = modulex(x)
                    if namex in self.sparcified_layers[sub_index]:
                        
                        if hasattr(modulex, 'omega_val') :
                            
                            
                            
                            neuron_omega_val=modulex.omega_val
                            
                            if self.scale==0:
                                decov_loss +=deCov_loss_neuron_mega(x,neuron_omega_val,self.min,self.squash) 
                            else:
                                decov_loss += deCov_loss_neuron_mega_gaussian_weighted(x,neuron_omega_val,x.size(1)/self.scale,self.min,self.squash)
                        else:
                            if self.scale==0:
                                decov_loss +=deCov_loss(x)
                            else:
                                decov_loss += deCov_loss_gaussian_weighted(x,x.size(1)/self.scale)
                #for reshaping the fully connected layers
                #need to be changed for 
                if sub_index==0:
                    
                    try:
                        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
                    except:
                        pass
                sub_index+=1
        #pdb.set_trace()        
        return x,decov_loss
    

def deCov_loss(A):
  
    #e=torch.ones((A.size(0)),1).cuda()
    #e= Variable(e, requires_grad=False)
    #etA=torch.mm(torch.transpose(e,0,1),A)
    #X=A- (1/A.size(0))*etA
    
    cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
    F_cov_norm=cov.norm(1)#*cov.norm(2)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)#*cov_diag.norm(2)
    decov_loss=(F_cov_norm-cov_diag_norm)
    #divide by the number of neurons
    decov_loss=decov_loss#/A.size(1)
    #NORM 1 is HERE!
    if decov_loss.data.item()<0:
        print('DECOV LOSS L1 IS ZEROO')
        return 0
    return decov_loss

def deCov_loss_gaussian_weighted(A,scale=32):
  
    #e=torch.ones((A.size(0)),1).cuda()
    #e= Variable(e, requires_grad=False)
    #etA=torch.mm(torch.transpose(e,0,1),A)
    #X=A- (1/A.size(0))*etA
   
    cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
    normal_weights=np.fromfunction(lambda i, j: stats.norm.pdf(abs(i-j), loc=0, scale=scale)/stats.norm.pdf(0, loc=0, scale=scale), cov.size(), dtype=int)
    normal_weights=torch.Tensor(normal_weights).cuda()
    cov=cov*normal_weights
    F_cov_norm=cov.norm(1)#*cov.norm(2)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag

    cov_diag_norm=cov_diag.norm(1)#*cov_diag.norm(2)
    decov_loss=(F_cov_norm-cov_diag_norm)
    #divide by the number of neurons
    decov_loss=decov_loss#/A.size(1)
    #NORM 1 is HERE!
    if decov_loss.data.item()<0:
        print('DECOV LOSS L1 IS ZEROO')
        return 0
    return decov_loss

def deCov_loss_neuron_mega(A,neuron_omega_val,take_min=False,squash='exp'):
  
    #e=torch.ones((A.size(0)),1).cuda()
    #e= Variable(e, requires_grad=False)
    #etA=torch.mm(torch.transpose(e,0,1),A)
    #X=A- (1/A.size(0))*etA
    #neuron_omega_val=-self.reg_params[key]
    sigmoid=torch.nn.Sigmoid()
    if squash=='exp':
        y=torch.exp(-neuron_omega_val)
    else:
        pdb.set_trace()
        y=1- sigmoid(neuron_omega_val)
        y=(y-y.min())/(y.max()-y.min())
                                  
    if take_min:
        y=y.expand(y.size(0),y.size(0))
        yt=y.transpose(0,1)
        y=torch.min(y,yt)
        y=Variable(y.data, requires_grad=False)
        cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
        cov=cov*y
    else:
        y=Variable(y.data, requires_grad=False)

        Az=torch.mul(y,A)
        cov=(1/Az.size(0))*torch.mm(torch.transpose(Az,0,1),Az)
    F_cov_norm=cov.norm(1)#*cov.norm(2)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)#*cov_diag.norm(2)
    decov_loss=(F_cov_norm-cov_diag_norm)
    #divide by the number of neurons
    decov_loss=decov_loss#/A.size(1)
    if decov_loss.data.item()<0:
        print('DECOV LOSS L1 IS ZEROO')
        return 0
    return decov_loss

def deCov_loss_neuron_mega_gaussian_weighted(A,neuron_omega_val,scale,take_min=False,squash='exp'):
  
    sigmoid=torch.nn.Sigmoid()
    if squash=='exp':
        y=torch.exp(-neuron_omega_val)
    else:
        y=1- sigmoid(neuron_omega_val)
        y=(y-y.min())/(y.max()-y.min())
                                  
    if take_min:
        y=y.expand(y.size(0),y.size(0))
        yt=y.transpose(0,1)
        y=torch.min(y,yt)
        y=Variable(y.data, requires_grad=False)
        cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
        cov=cov*y
    else:
        y=Variable(y.data, requires_grad=False)

        Az=torch.mul(y,A)
        cov=(1/Az.size(0))*torch.mm(torch.transpose(Az,0,1),Az)
        
    normal_weights=np.fromfunction(lambda i, j: stats.norm.pdf(abs(i-j), loc=0, scale=scale)/stats.norm.pdf(0, loc=0, scale=scale), cov.size(), dtype=int)
    normal_weights=torch.Tensor(normal_weights).cuda()
    cov=cov*normal_weights
    F_cov_norm=cov.norm(1)#*cov.norm(2)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)#*cov_diag.norm(2)
    decov_loss=(F_cov_norm-cov_diag_norm)
    #divide by the number of neurons
    decov_loss=decov_loss#/A.size(1)
    if decov_loss.data.item()<0:
        print('DECOV LOSS L1 IS ZEROO')
        return 0
    return decov_loss