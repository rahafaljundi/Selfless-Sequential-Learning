#Create different backbone networks
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
class MNIST_Net(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256):
        super(MNIST_Net, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
class MNIST_Net_Drop(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256):
        super(MNIST_Net_Drop, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
from LWTA import *
class MNIST_LWTA_Net(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256,window_size=2):
        super(MNIST_LWTA_Net, self).__init__()

        self.classifier = nn.Sequential(
           
            nn.Linear( 28 * 28, hidden_size),
            LWTA(hidden_size,hidden_size,window_size),
            nn.Linear(hidden_size, hidden_size),
            LWTA(hidden_size,hidden_size,window_size),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        return x
import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
from LWTA import *
class MNIST_MAXOUT_Net(nn.Module):
    def __init__(self,num_classes=10,hidden_size=256,window_size=2):
        super(MNIST_MAXOUT_Net, self).__init__()

        
           
        self.lin1=nn.Linear( 28 * 28, hidden_size)
        self.maxout=nn.MaxPool1d(kernel_size=window_size)
        self.lin2=nn.Linear(int(hidden_size/2), int(hidden_size/2))

        self.lin3=nn.Linear(int(hidden_size/4), num_classes)

    def forward(self, x):
        
        x = self.lin1(x)
        x=x.view(1,x.size(0),x.size(1))
        x=self.maxout(x)
        x=x.view(x.size(1),x.size(2))
        x = self.lin2(x)   
        x=x.view(1,x.size(0),x.size(1))
        x=self.maxout(x)
        x=x.view(x.size(1),x.size(2))
        x = self.lin3(x)   
        return x
