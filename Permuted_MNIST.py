
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys

import codecs

import random
import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):
    
    def __init__(self, root="../data/", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        self.classes= [0,1,2,3,4,5,6,7,8,9] 
        if self.train:
            self.train_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                           for img in self.train_data])
        else:
            self.test_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                          for img in self.test_data])

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        
        return [img for img in self.train_data[sample_idx]]


def get_permute_mnist(num_task=3,batch_size=100):
    all_dsets=[]
    idx = list(range(28 * 28))
    for i in range(num_task):
        dsets={}
        dsets['train'] =PermutedMNIST(train=True, permute_idx=idx)
        dsets['val'] = PermutedMNIST(train=False, permute_idx=idx)
        all_dsets.append(dsets)
        random.shuffle(idx)
    return all_dsets
