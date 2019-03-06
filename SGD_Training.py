
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import shutil
from torch.utils.data import DataLoader
#end of imports

def train_model(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,lr_decay_epoch=45,exp_dir='./',resume=''):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc=checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,lr,lr_decay_epoch=lr_decay_epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #print('step')
                    optimizer.step()

                # statistics
                
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'epoch_acc':epoch_acc,
            'best_acc':best_acc,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model,best_acc

def train_model_Sparce(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='', lam=5e-7,lr_decay_epoch=45):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()
   
    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        #modelx = checkpoint['model']
        #model.reg_params=modelx.reg_params
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        #pdb.
        #model.reg_params=reg_params
        #del model.reg_params
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,lr,lr_decay_epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs,norm = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss=loss+lam*norm
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #print('step')
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'epoch_acc':epoch_acc,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model
#tasks are sampled with probablities and not hard boundaries
def train_model_Sparce_tasksprob(model, criterion, optimizer, lr_scheduler,lr,dsets_pathes,task_index,probs,batch_size,use_gpu, num_epochs\
                                 ,exp_dir='./',resume='', lam=5e-7,lr_decay_epoch=45):
    
    tasks_dset_loaders=[]
    
    labels_bias=[]
    prev_last_class=0
    phases_dataloader_iterators={}
    
    phases_dataloader_iterators['train']=[]
    phases_dataloader_iterators['val']=[]
    for dataset_path in dsets_pathes:
        task=dsets_pathes.index(dataset_path)
        dsets = torch.load(dataset_path)
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size*probs[task],
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        
        dset_classes = dsets['train'].classes
        tasks_dset_loaders.append(dset_loaders)
       
        phases_dataloader_iterators['train'].append(iter(dset_loaders['train']))
        phases_dataloader_iterators['val'].append(iter(dset_loaders['val']))
        labels_bias.append(prev_last_class)
        
        prev_last_class+=len(dset_classes)
        
    since = time.time()
       
    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        #modelx = checkpoint['model']
        #model.reg_params=modelx.reg_params
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        #pdb.
        #model.reg_params=reg_params
        #del model.reg_params
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
            
                optimizer = lr_scheduler(optimizer, epoch,lr,lr_decay_epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            
            # Iterate over data.
            dest_size=0
            
            Flag=True
            while Flag:

                # get the inputs

                all_inputs=None
                all_labels=None
                
                for task in range(0,len(tasks_dset_loaders)):

                    this_label_bias=labels_bias[task]
                    try:
                        inputs, labels=next(phases_dataloader_iterators[phase][task])
                    except StopIteration:
                        #pdb.set_trace()
                        dataloader_iterator = iter(tasks_dset_loaders[task][phase])
                        phases_dataloader_iterators[phase][task]=dataloader_iterator
                        Flag=False
                    if Flag:
                        inputs=inputs.squeeze()
                        if task==0:
                            all_inputs,all_labels=inputs, labels
                        else:
                            all_inputs=torch.cat((all_inputs,inputs),0)
                            all_labels=torch.cat((all_labels,labels+this_label_bias),0)
                        del inputs,labels

                # wrap them in Variable

                if Flag:
                    if use_gpu:
                        all_inputs, all_labels = Variable(all_inputs.cuda()), \
                            Variable(all_labels.cuda())
                
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs,norm = model(all_inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, all_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss=loss+lam*norm
                        loss.backward()
                        #print('step')
                        optimizer.step()

                    # statistics
                    #pdb.set_trace() 
                    running_loss += loss.data.item()
                    running_corrects += torch.sum(preds == all_labels.data).item()
                    dest_size+=all_labels.size(0)
                   
            epoch_loss = running_loss / dest_size
            epoch_acc = running_corrects / dest_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del all_labels
                del all_inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))



            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del all_labels
                del all_inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'epoch_acc':epoch_acc,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model

def set_lr(optimizer, lr,count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training=True
    if count>10:
        continue_training=False
        print("training terminated")
    if count==5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer,lr,continue_training

def traminate_protocol(since,best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    

def train_model_sparce_early_stopping(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume='',lam=0,lr_decay_epoch=45):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()
    val_beat_counts=0#number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc=checkpoint['best_acc']
        lr=checkpoint['lr']
        print("lr is ",lr)
        val_beat_counts=checkpoint['val_beat_counts']
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer,lr,continue_training = set_lr(optimizer,lr,count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since,best_acc)
                    return model,best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs=inputs.squeeze()
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs,norm = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
        
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss=loss+lam*norm
                    loss.backward()
                    #print('step')
                    optimizer.step()

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' :
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    #best_model = copy.deepcopy(model)
                    torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts=0
                else:
                    val_beat_counts+=1
                    print("val_beat_counts is",str(val_beat_counts))
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'lr':lr,
            'val_beat_counts':val_beat_counts,
            'model': model,
            'epoch_acc':epoch_acc,
            'best_acc':best_acc,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model,best_acc
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)
   