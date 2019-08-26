import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.ion()


import shutil
import pdb
import os


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    '16Slim': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    '11Slim': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
    
}

class VGGSlim(torchvision.models.VGG):

    def __init__(self, config='11Slim', num_classes=50, init_weights=True):
        features=make_layers(cfg[config])
        super(VGGSlim2, self).__init__(features)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes),
        )
        if init_weights:
            self._initialize_weights()
