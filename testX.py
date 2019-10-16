

import argparse
import sys
import os
import pdb
os.chdir('/users/visics/raljundi/Code/MyOwnCode/Pytorch/Object_recognition')
print(os.getcwd())
sys.path.append('/users/visics/raljundi/Code/MyOwnCode/Pytorch/Object_recognition')
sys.path.append('/users/visics/raljundi/Code/MyOwnCode/Pytorch/Object_recognition/SNI_ICLR')
sys.path.append('/users/visics/raljundi/Code/MyOwnCode/Pytorch/my_utils')
sys.path.append('/users/visics/raljundi/Code/MyOwnCode/Pytorch/survey/tiny_imagenet')
from VGGSlim import *
from Test_sequntial import *
import traceback
results={}#torch.load("tinyImagenet.pth")
lams=[5e-6,2e-6,1e-6,5e-7,1e-7,5e-7,8e-7,5e-8,1e-8]

scales=[6.0,0.0,2.0,4.0,8.0]
lambdas=[4.0,2.0,1.0]

task_model_path="/esat/monkey/raljundi/pytorch/object_recognition_exp/selfless_sequential/10tasks/VGG11Slim2/CES/1/MAS_DecovZML/lam_2e-06num_epochs_60reg_lambda_4.0lr_0.01b1_Falseneuron_omega_Truenormalize_Falsedropout_Falsescale_6.0/best_model.pth.tar"
data_dir='/esat/monkey/raljundi/tiny-imagenet-200-tasks//tiny-imagenet-200/'
dataset_path=os.path.join(data_dir,'1','trainval_dataset.pth.tar')
model_ft=torch.load(task_model_path)
task=1
acc=1
for i in [0,3,6,8,11,13]:
    model_ft.module.features[i].padding_mode='zeros'
print(model_ft)
torch.save(model_ft,task_model_path)
accorg=test_model_sparce(task_model_path,dataset_path) 
print('task number ',task,' accuracy is %.2f'%acc, 'compared to %.2f'%accorg) 

task_model_path="/esat/monkey/raljundi/pytorch/object_recognition_exp/selfless_sequential/10tasks/VGG11Slim2/CES/TESTGIHUB/1/SLNID/tinyimagenet_explam_2e-06num_epochs_60reg_lambda_4lr_0.01b1_Falseneuron_omega_Truenormalize_Falsedropout_Falsescale_6/best_model.pth.tar"

data_dir='/esat/monkey/raljundi/tiny-imagenet-200-tasks//tiny-imagenet-200/'
dataset_path=os.path.join(data_dir,'1','trainval_dataset.pth.tar')
model_ft=torch.load(task_model_path)
for i in [0,3,6,8,11,13]:
    model_ft.module.features[i].padding_mode='zeros'
print(model_ft)
torch.save(model_ft,task_model_path)
accorg=test_model_sparce(task_model_path,dataset_path) 
print('task number ',task,' accuracy is %.2f'%acc, 'compared to %.2f'%accorg) 

