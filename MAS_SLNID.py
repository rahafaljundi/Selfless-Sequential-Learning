#Implements the main functions for using MAS regularizer
from __future__ import print_function, division

import torch
import torch.nn as nn
import sys
from ImageFolderTrainVal import *
import Regularized_Training

import pdb

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
   
    lr = init_lr* (0.1**(epoch // lr_decay_epoch))
    
    if epoch>0 and epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            
            param_group['lr'] =param_group['lr']* 0.1
            
    return optimizer

def replace_heads(previous_model_path,current_model_path):
    current_model_ft=torch.load(current_model_path)
        
    previous_model_ft=torch.load(previous_model_path)
    current_model_ft.classifier._modules['6']=previous_model_ft.classifier._modules['6']
    return current_model_ft
def fine_tune_objective_based_orthreg_acuumelation(dataset_path,previous_task_model_path,init_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='', num_epochs=100,lr=0.0008,batch_size=200,after_freeze=1,weight_decay=1e-3,b1=False,head_shared=False):
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=150,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
   

    model_ft = torch.load(previous_task_model_path)
    if  isinstance(model_ft, dict):
        model_ft=model_ft['model']
    model_ft=accumulate_objective_based_weights(data_dir,reg_sets,model_ft,batch_size,norm)
    model_ft.reg_params['lambda']=reg_lambda
    #get the number of features in this network and add a new task head
    last_layer_index=str(len(model_ft.classifier._modules)-1)
##############
    if not head_shared:
        last_layer_index=str(len(model_ft.classifier._modules)-1)
        if not init_model_path is None:
            init_model = torch.load(init_model_path) 
            model_ft.classifier._modules[last_layer_index] = init_model.classifier._modules[last_layer_index]

        else:
            num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
   

 #************************************************   

    
    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    
    if use_gpu:
        model_ft = model_ft.cuda()
    

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    
    optimizer_ft = Regularized_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,weight_decay=weight_decay,orth_reg=True)
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not init_model_path is None:    
        del init_model
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    model_ft = Regularized_Training.train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume)
    
    return model_ft
def fine_MAS_acuumelation_sparce(dataset_path,previous_task_model_path,init_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='', num_epochs=100,lr=0.0008,batch_size=200,after_freeze=1,lam=0,b1=True,L1_decay=False,weight_decay=1e-5,lr_multiplier=None,head_shared=False,neuron_omega=False,metric='avg',augment_omega=False,lr_decay_epoch=45):
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)
    if  isinstance(model_ft, dict):
        model_ft=model_ft['model']    
    if b1:
        #compute the importance with batch size of 1
        update_batch_size=1
    else:
        update_batch_size=batch_size
    #moved after adding the backward hook, so neurons importance is based on their gradients.  
    handles=[]
    if neuron_omega:
        
        model_ft,handles=compute_neurons_omega(model_ft)
        model_ft.neuron_omega=True  
        
    model_ft=accumulate_objective_based_weights_sparce(data_dir,reg_sets,model_ft,update_batch_size,norm,augment_omega,metric)
    for handel in handles:
        handel.remove()
    
    set_neurons_omega_val(model_ft)
    model_ft.reg_params['lambda']=reg_lambda
    #get the number of features in this network
    last_layer_index=str(len(model_ft.module.classifier._modules)-1)
    if not head_shared:
        last_layer_index=str(len(model_ft.module.classifier._modules)-1)
        if not init_model_path is None:
            init_model = torch.load(init_model_path) 
            #hack
            if hasattr(init_model,'module'):
                init_model=init_model.module
            model_ft.module.classifier._modules[last_layer_index] = init_model.classifier._modules[last_layer_index]

        else:
            num_ftrs=model_ft.module.classifier._modules[last_layer_index].in_features 
            model_ft.module.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
   
    #if neuron_omega:
    #    model_ft=compute_neurons_omega(model_ft,metric)
    #    model_ft.neuron_omega=True 

    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    sanitycheck(model_ft) 
    if use_gpu:
        model_ft = model_ft.cuda()
    

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
    
    optimizer_ft = Regularized_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,weight_decay=weight_decay,L1_decay=L1_decay)

     
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not init_model_path is None:    
        del init_model
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    model_ft,best_acc  = Regularized_Training.train_model_sparce(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume,lam=lam)


    return model_ft,best_acc 
def fine_tune_objective_based_acuumelation_sparce_prob(dataset_pathes,task_index,probs,previous_probs,previous_task_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='', num_epochs=100,lr=0.0008,batch_size=200,after_freeze=1,lam=0,b1=True,L1_decay=False,weight_decay=1e-5,lr_multiplier=None,head_shared=True,neuron_omega=False,metric='avg',augment_omega=False,lr_decay_epoch=45):

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)
    if  isinstance(model_ft, dict):
        model_ft=model_ft['model']    
    if b1:
        #compute the importance with batch size of 1
        update_batch_size=1
    else:
        update_batch_size=batch_size
    #moved after adding the backward hook, so neurons importance is based on their gradients.  
    handles=[]
    if neuron_omega:
        
        model_ft,handles=compute_neurons_omega(model_ft)
        model_ft.neuron_omega=True  
        
    model_ft=accumulate_objective_based_weights_sparce_tasks_prob(dataset_pathes,task_index-1,previous_probs,model_ft,batch_size,norm,augment_omega,metric,b1)
    for handel in handles:
        handel.remove()
    
    set_neurons_omega_val(model_ft)
    model_ft.reg_params['lambda']=reg_lambda

    #if neuron_omega:
    #    model_ft=compute_neurons_omega(model_ft,metric)
    #    model_ft.neuron_omega=True 

    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    sanitycheck(model_ft) 
    if use_gpu:
        model_ft = model_ft.cuda()
    

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
    
    optimizer_ft = Regularized_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,weight_decay=weight_decay,L1_decay=L1_decay)

     
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    resume=os.path.join(exp_dir,'epoch.pth.tar')
    model_ft,best_acc  = Regularized_Training.train_model_sparce_tasks_probs(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,lr_decay_epoch,dataset_pathes,task_index,probs,batch_size,use_gpu,num_epochs,exp_dir,resume,lam=lam)
    
    
    return model_ft,best_acc 
def MAS_SprasePenalty_ES(dataset_path,previous_task_model_path,init_model_path,exp_dir,data_dir,reg_sets,reg_lambda=1,norm='', num_epochs=100,lr=0.0008,batch_size=200,after_freeze=1,lam=0,b1=True,L1_decay=False,weight_decay=1e-5,lr_multiplier=None,head_shared=False,neuron_omega=False,metric='avg',augment_omega=False,lr_decay_epoch=45):
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
   
    model_ft = torch.load(previous_task_model_path)
    if  isinstance(model_ft, dict):
        model_ft=model_ft['model']    
    if b1:
        #compute the importance with batch size of 1
        update_batch_size=1
    else:
        update_batch_size=batch_size
    #moved after adding the backward hook, so neurons importance is based on their gradients.  
    handles=[]
    if neuron_omega:
        
        model_ft,handles=compute_neurons_omega(model_ft)
        model_ft.neuron_omega=True  
        
    model_ft=accumulate_objective_based_weights_sparce(data_dir,reg_sets,model_ft,update_batch_size,norm,augment_omega,metric)
    for handel in handles:
        handel.remove()
    set_neurons_omega_val(model_ft)
    model_ft.reg_params['lambda']=reg_lambda
    #get the number of features in this network
    last_layer_index=str(len(model_ft.module.classifier._modules)-1)
    if not head_shared:
        last_layer_index=str(len(model_ft.module.classifier._modules)-1)
        if not init_model_path is None:
            init_model = torch.load(init_model_path) 
            #hack
            if hasattr(init_model,'module'):
                init_model=init_model.module
            model_ft.module.classifier._modules[last_layer_index] = init_model.classifier._modules[last_layer_index]

        else:
            num_ftrs=model_ft.module.classifier._modules[last_layer_index].in_features 
            model_ft.module.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
   
    #if neuron_omega:
    #    model_ft=compute_neurons_omega(model_ft,metric)
    #    model_ft.neuron_omega=True 

    criterion = nn.CrossEntropyLoss()
    #update the objective based params
    sanitycheck(model_ft) 
    if use_gpu:
        model_ft = model_ft.cuda()
    

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
    
    optimizer_ft = Regularized_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,weight_decay=weight_decay,L1_decay=L1_decay)

     
    #exp_dir='/esat/monkey/raljundi/pytorch/CUB11f_hebbian_finetuned'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not init_model_path is None:    
        del init_model
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    model_ft,best_acc = Regularized_Training.train_model_sparce_early_stopping(model_ft, criterion, optimizer_ft,exp_lr_scheduler, lr,lr_decay_epoch,dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume,lam=lam)
    
    
    return model_ft,best_acc
def accumulate_objective_based_weights(data_dir,reg_sets,model_ft,batch_size,norm=''):
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    
    #========================COPIED FROM FACTS*
   
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)
    #=============================================================================
    
    use_gpu = torch.cuda.is_available()
    #hack
    if not hasattr(model_ft,'reg_params'):
        
        reg_params=Regularized_Training.initialize_reg_params(model_ft)
        model_ft.reg_params=reg_params

    reg_params=Regularized_Training.initialize_store_reg_params(model_ft)
    model_ft.reg_params=reg_params
    
    optimizer_ft = Regularized_Training.MAS_OMEGA_ESTIMATE(model_ft.parameters(), lr=0.0001, momentum=0.9)
   
    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft = Regularized_Training.compute_importance_l2(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        model_ft = Regularized_Training.compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)

    reg_params=Regularized_Training.accumelate_reg_params(model_ft)
    model_ft.reg_params=reg_params
    sanitycheck(model_ft)   
    return model_ft
def accumulate_objective_based_weights_sparce(data_dir,reg_sets,model_ft,batch_size,norm='',augment_omega=False,metric='avg'):
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    
    #========================COPIED FROM FACTS*
   
    dset_loaders=[]
    for data_path in reg_sets:
    
        # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
        if data_dir is not None:
            dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
        else:
            dset=torch.load(data_path)
            dset=dset['train']
        
        dset_loader= torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)
        dset_loaders.append(dset_loader)
    #=============================================================================
    
    use_gpu = torch.cuda.is_available()


    if not hasattr(model_ft,'reg_params'):
       
        reg_params=Regularized_Training.initialize_reg_params(model_ft)
        model_ft.reg_params=reg_params
 
    reg_params=Regularized_Training.initialize_store_reg_params(model_ft)
    model_ft.reg_params=reg_params
    
    optimizer_ft = Regularized_Training.MAS_OMEGA_ESTIMATE(model_ft.parameters(), lr=0.0001, momentum=0.9)
   
    if norm=='L2':
        print('********************objective with L2 norm***************')
        model_ft = Regularized_Training.compute_importance_l2_sparce(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    else:
        model_ft = Regularized_Training.compute_importance(model_ft, optimizer_ft,exp_lr_scheduler, dset_loaders,use_gpu)
    if augment_omega:
        augment_importance_with_neron_omega(model_ft,metric)
    reg_params=Regularized_Training.accumelate_reg_params(model_ft)
    model_ft.reg_params=reg_params
    
    return model_ft
def accumulate_objective_based_weights_sparce_tasks_prob(dataset_pathes,task_index,probs,model_ft,batch_size,norm='',augment_omega=False,metric='avg',b1=False):
    data_transform =  transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    

    use_gpu = torch.cuda.is_available()


    if not hasattr(model_ft,'reg_params'):
       
        reg_params=Regularized_Training.initialize_reg_params(model_ft)
        model_ft.reg_params=reg_params
 
    reg_params=Regularized_Training.initialize_store_reg_params(model_ft)
    model_ft.reg_params=reg_params
    
    optimizer_ft = Regularized_Training.MAS_OMEGA_ESTIMATE(model_ft.parameters(), lr=0.0001, momentum=0.9)
   
    if b1:
        model_ft = Regularized_Training.compute_importance_l2_sparce_tasks_prb_onebatch(model_ft, optimizer_ft,exp_lr_scheduler, dataset_pathes,task_index,probs,batch_size,use_gpu)
    else:
        
        model_ft = Regularized_Training.compute_importance_l2_sparce_tasks_prb(model_ft, optimizer_ft,exp_lr_scheduler, dataset_pathes,task_index,probs,batch_size,use_gpu)

    if augment_omega:
        augment_importance_with_neron_omega(model_ft,metric)
    reg_params=Regularized_Training.accumelate_reg_params(model_ft)
    model_ft.reg_params=reg_params
    
    return model_ft

def sanitycheck(model):
    for name, param in model.named_parameters():
            #w=torch.FloatTensor(param.size()).zero_()
            print (name)
            if param in model.reg_params:
            
                reg_param=model.reg_params.get(param)
                omega=reg_param.get('omega')
                
                print('omega max is',omega.max().item())
                print('omega min is',omega.min().item())
                print('omega mean is',omega.mean().item())
#if omega was already computed based on another trial 
def move_omega(model1,model2):
    for name1, param1 in model1.named_parameters():
            #w=torch.FloatTensor(param.size()).zero_()
            print (name1)
            if param1 in model1.reg_params:
                for name2, param2 in model2.named_parameters():
                    if name1==name2 and param1.data.size()==param2.data.size() :
                        reg_param1=model1.reg_params.get(param1)
                        reg_param2=model2.reg_params.get(param2)
                        omega1=reg_param1.get('omega')
                        reg_param2['omega']=omega1.clone()
                    
    return model2      
def compute_neuron_importance(self, grad_input, grad_output):
    
    if 'ReLU' in self.__class__.__name__:
        
        if hasattr(self, "neuron_omega"):
            
            self.samples_size+=grad_input[0].size(0)
            if self.abs:
                self.neuron_omega+=torch.sum(torch.abs(grad_input[0]),0)
            else:
                self.neuron_omega+=torch.abs(torch.sum(grad_input[0],0))
            
        else:
            if self.abs:
                self.neuron_omega=torch.sum(torch.abs(grad_input[0]),0)#torch.abs(torch.sum((grad_input[0]),0))
            else:
                self.neuron_omega=torch.abs(torch.sum(grad_input[0],0))
            self.samples_size=grad_input[0].size(0)
            
            
def compute_neurons_omega(model):
    handels=[]
    for name, module in model.module._modules.items():

        for namex, modulex in module._modules.items():
            modulex.abs=model.abs
            handle=modulex.register_backward_hook(compute_neuron_importance)
            handels.append(handle)
            
    return model,handels
def check_neurons_omega(model):
    
    for name, module in model.module._modules.items():

        for namex, modulex in module._modules.items():
            
            if hasattr(modulex, "neuron_omega"):
                print(modulex.neuron_omega.size())
            
                  
            
    return model
def set_neurons_omega_val(model):
    
    for name, module in model.module._modules.items():

        for namex, modulex in module._modules.items():
            
            if hasattr(modulex, "neuron_omega"):
                if hasattr(modulex, "omega_val"):
                    if model.divide_by_tasks:

                        modulex.omega_val=(modulex.omega_val*modulex.task_nb +modulex.neuron_omega/modulex.samples_size)/(modulex.task_nb+1)
                        modulex.task_nb+=1
                    else:                    
                        modulex.omega_val+=modulex.neuron_omega/modulex.samples_size

                        
                else:
                    modulex.omega_val=modulex.neuron_omega/modulex.samples_size
                    modulex.task_nb=1
                del modulex.samples_size
                del modulex.neuron_omega
                  
            
    return model
