import argparse
import sys
import os
import pdb

from models.VGGSlim import *
def set_random(seed=7):
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
import random
import numpy

#GETTING INPUT PARAMS
#--------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, default=1e-8)
parser.add_argument('--num_epochs', type=int, default=80, help='training number of epochs')
parser.add_argument('--reg_lambda', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--b1', type=bool, default=False, help='online')
parser.add_argument('--neuron_omega', type=bool, default=False, help='wanna use neuron_omega?')
parser.add_argument('--normalize', type=bool, default=False, help='wanna use normalized neuron_omega?')
parser.add_argument('--dropout', type=bool, default=False, help='dropout enabled or disabled?')
parser.add_argument('--scale', type=float, default=4, help='scale of Gaussian')
opt = parser.parse_args()
lam=opt.lam
scale=opt.scale
normalize=opt.normalize
num_epochs=opt.num_epochs
reg_lambda=opt.reg_lambda
lr=opt.lr
neuron_omega=opt.neuron_omega
b1=opt.b1
dropout=opt.dropout
lr_decay_epoch=20
extra_str="tinyimagenet_exp"
for arg in vars(opt):
    extra_str=extra_str+str(arg,)+'_'+str(getattr(opt, arg))

print(extra_str)
in_layers=[[],['1','3']]
task_name='1'
#--------------------------------------
nb_tasks=10

#--------------------------------------
if 1:
    model_path='/yourpath//vgg11slim.pth.tar'
    parent_exp_dir='/yourexpdir/'
    dataset_parent_dir='/yourpath/tiny-imagenet-200/'
    from Finetune_SNI_ICLR import *

    exp_dir=os.path.join(parent_exp_dir,'1','SLNID/'+extra_str)
    dataset_path=os.path.join(dataset_parent_dir,task_name,'trainval_dataset.pth.tar')
    set_random()
    fine_tune_SGD_SLNI(dataset_path=dataset_path, num_epochs=num_epochs,exp_dir=exp_dir,model_path=model_path,batch_size=200,lr=lr,init_freeze=0,lam=lam,lr_decay_epoch=lr_decay_epoch,in_layers=in_layers,pretrained=False,weight_decay=0,scale=scale)



    from MAS_SNI_ICLR import *


    batch_size=200

    weight_decay=0
    lr=1e-2
    
    
    previous_model_path=os.path.join(exp_dir,'best_model.pth.tar')

    model=torch.load(previous_model_path)
    model.divide_by_tasks=True
    torch.save(model,previous_model_path)
    for task in range(2,nb_tasks+1):
        not_converging=True
        
        task_name =str(task)
        dataset_path=os.path.join(dataset_parent_dir,task_name,'trainval_dataset.pth.tar')
       
        exp_dir=os.path.join(parent_exp_dir,task_name,'yourexpdir')
        init_model_path=None
        reg_set=os.path.join(dataset_parent_dir,str(task-1),'Notransform_trainval_dataset.pth.tar')
        while not_converging:
            model,best_acc=fine_MAS_acuumelation_sparce(dataset_path=dataset_path,previous_task_model_path=previous_model_path,init_model_path=init_model_path,exp_dir=exp_dir,data_dir=None,reg_sets=[reg_set],reg_lambda=reg_lambda,batch_size=200,num_epochs=num_epochs,lr=lr,norm='L2',after_freeze=0,lam=lam,b1=b1,neuron_omega=neuron_omega,lr_decay_epoch=lr_decay_epoch,weight_decay=weight_decay)
            
            if best_acc<0.10:
                reg_lambda=reg_lambda/2  
                shutil.rmtree(exp_dir)
            else:
                not_converging=False     
                previous_model_path=os.path.join(exp_dir,'best_model.pth.tar')

