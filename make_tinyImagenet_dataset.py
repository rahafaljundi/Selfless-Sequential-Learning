import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.ion()
from ImageFolderTrainVal import *
from VGGSlim import *
import shutil
import pdb
import os

# Data loading code
data_dir='./TINYIMAGNET/'    
for task in range(1,11):
    traindir = os.path.join(data_dir, str(task),'train')
    valdir = os.path.join(data_dir,str(task), 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(64),
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
    ]))

    val_dataset=datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                normalize,
    ]))
    dsets={}
    dsets['train']=train_dataset
    dsets['val']=val_dataset    
    torch.save(dsets,os.path.join(data_dir,str(task),'trainval_dataset.pth.tar'))    


# Data loading code
data_dir='./TINYIMAGNET/'  
for task in range(1,10):
    traindir = os.path.join(data_dir, str(task),'train')
    valdir = os.path.join(data_dir,str(task), 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_dataset =datasets.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                normalize,
    ]))

    val_dataset=datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                normalize,
    ]))
    dsets={}
    dsets['train']=train_dataset
    dsets['val']=val_dataset    
    torch.save(dsets,os.path.join(data_dir,str(task),'Notransform_trainval_dataset.pth.tar'))    
