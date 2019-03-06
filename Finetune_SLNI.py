

import sys
from SGD_Training import *
from SLNID import *

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
   
    lr = init_lr* (0.1**(epoch // lr_decay_epoch))
    
    if epoch>0 and epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            
            param_group['lr'] =param_group['lr']* 0.1
            

    return optimizer


def fine_tune_SGD_SLNI(dataset_path,model_path,exp_dir,batch_size=100, num_epochs=100,lr=0.0004,init_freeze=1,lam=1e-8,lr_decay_epoch=45,\
pretrained=True,in_layers=[[],['2','5']],weight_decay=1e-5,lr_multiplier=None,L1_Decay=False,scale=4,squash='exp'):
    """
    scale for Gaussian Weighting, the size of the layer is diveded by this value to determine the scale of the Gaussiana
    and hence the affect of the cov loss between neurons.
    """
    print('lr is ' + str(lr))
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume=os.path.join(exp_dir,'epoch.pth.tar')

    if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model_ft = checkpoint['model']
    if not os.path.isfile(model_path):
        model_ft = models.alexnet(pretrained=pretrained)
       
    else:
        model_ft=torch.load(model_path)
    if not init_freeze:    
        
        last_layer_index=str(len(model_ft.classifier._modules)-1)
        num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
        model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()
    
    model_ft=SLNI_Module(model_ft,in_layers)
    model_ft.scale=scale
    model_ft.squash=squash
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
    if L1_Decay:
        
        optimizer_ft =  L1WDecay_SGD(lr_params, lr, momentum=0.9,weight_decay=weight_decay)
    else:    
        optimizer_ft =  optim.SGD(lr_params, lr, momentum=0.9,weight_decay=weight_decay)
    #if  hasattr(model_ft, 'reg_params'):
    #    reg_params=Elastic_Training.reassign_reg_params(model_ft,new_reg_params)
        
    
  
    model_ft = train_model_Sparce(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume,lam=lam,lr_decay_epoch=lr_decay_epoch)
    
    return model_ft


def fine_tune_SGD_Decov_zm_L1_Probs(dataset_pathes,task_index,probs,model_path,exp_dir,batch_size=100, num_epochs=100,lr=0.0004,init_freeze=1,lam=1e-8,lr_decay_epoch=45,pretrained=True,in_layers=[[],['2','5']],weight_decay=1e-5,lr_multiplier=None,L1_Decay=False,scale=4,squash='exp'):
    """
    scale for Gauassian Weighting, the size of the layer is diveded by this value to determine the scale of the Gaussian
    and hence the affect of the cov loss between neurons.
    """
    
    print('lr is ' + str(lr))
    print("********* THIS IS ICLR VERSION***********")


    use_gpu = torch.cuda.is_available()
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model_ft = checkpoint['model']
    if not os.path.isfile(model_path):
        model_ft = models.alexnet(pretrained=pretrained)
       
    else:
        model_ft=torch.load(model_path)
  
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()
    
    model_ft=AlexNet_DeCov_zm(model_ft,in_layers)
    model_ft.scale=scale
    model_ft.squash=squash
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
   
    optimizer_ft =  optim.SGD(lr_params, lr, momentum=0.9,weight_decay=weight_decay)
    #if  hasattr(model_ft, 'reg_params'):
    #    reg_params=Elastic_Training.reassign_reg_params(model_ft,new_reg_params)
        
    
  
    model_ft = train_model_Sparce_tasksprob(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dataset_pathes,task_index,probs,batch_size,use_gpu,num_epochs,exp_dir,resume,lam=lam,lr_decay_epoch=lr_decay_epoch)
    
    return model_ft


def fine_tune_SGD_Decov_zm_L1_earlyStopping(dataset_path,model_path,exp_dir,batch_size=100, num_epochs=100,lr=0.0004,init_freeze=1,lam=1e-8,lr_decay_epoch=45,pretrained=True,in_layers=[[],['2','5']],weight_decay=1e-5,lr_multiplier=None,L1_Decay=False,scale=4):
    """
    scale for Gauassian Weighting, the size of the layer is diveded by this value to determine the scale of the Gaussiana 
    and hence the affect of the cov loss between neurons.
    """
    print('lr is ' + str(lr))
    print("********* THIS IS ICLR VERSION***********")
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume=os.path.join(exp_dir,'epoch.pth.tar')
    if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            model_ft = checkpoint['model']
    if not os.path.isfile(model_path):
        model_ft = models.alexnet(pretrained=pretrained)
       
    else:
        model_ft=torch.load(model_path)
    if not init_freeze:    
        
        last_layer_index=str(len(model_ft.classifier._modules)-1)
        num_ftrs=model_ft.classifier._modules[last_layer_index].in_features 
        model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))  
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()
    
    model_ft=AlexNet_DeCov_zm(model_ft,in_layers)
    model_ft.scale=scale
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0008, momentum=0.9)
    if not lr_multiplier is None:
        #feature extraction run with lower learning rate
        lr_params=    [{'params': model_ft.module.features.parameters(), 'lr': lr_multiplier*lr},
        {'params': model_ft.module.classifier.parameters()}]
    else:   
        lr_params=   model_ft.parameters()
    if L1_Decay:
        
        optimizer_ft =  L1WDecay_SGD(lr_params, lr, momentum=0.9,weight_decay=weight_decay)
    else:    
        optimizer_ft =  optim.SGD(lr_params, lr, momentum=0.9,weight_decay=weight_decay)
    #if  hasattr(model_ft, 'reg_params'):
    #    reg_params=Elastic_Training.reassign_reg_params(model_ft,new_reg_params)
        
    
  
    model_ft = train_model_sparce_early_stopping(model_ft, criterion, optimizer_ft,exp_lr_scheduler,lr, dset_loaders,dset_sizes,use_gpu,num_epochs,exp_dir,resume,lam=lam,lr_decay_epoch=lr_decay_epoch)
    
    return model_ft