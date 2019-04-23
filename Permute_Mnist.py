from Permuted_MNIST import *
import os
def create_datasets(path='./Datasets'):
    if not os.path.isdir(path):
        os.mkdir(path)
    num_task=5
    all_dsets=get_permute_mnist(num_task=num_task)
    for t in range(0,num_task):
        torch.save(all_dsets[t],path+'/permuted_t'+str(t)+'_dataset.pth.tar')