

import argparse
import sys
import os
import pdb

from MAS_SLNID import *
from VGGSlim import *
from Test_Utils import *
import traceback

import numpy as np
import matplotlib.pyplot as plt
import pylab
def plot_multibar( seqacc,keys,colors,labels,hatches,save_img=False,ylim=(88, 97),bar_widthT= 0.08,bar_width = 0.1,legend="out"):
    # data to plot
    n_groups = len(keys)

    # create plot
    fig, ax = plt.subplots()
    index = np.array([0,1.3,2.5,3.6,4.7])

    index2 = np.array([0,1.3,2.5,3.6,4.7,5.9,7.1,8.2,9.5])
    
    
    opacity = 0.8

    plt.grid(True, alpha=0.3)
    xx=0
    methindex=0
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 14
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    X = np.arange(len(seqacc[keys[0]]))
    xticks=['T'+str(x+1) for x in X]

    for key in keys:
        
        pylab.bar(X+methindex* bar_width, seqacc[key], width=bar_widthT,color= colors[methindex],label=labels[methindex],hatch=hatches[methindex],alpha=0.9)
        methindex+=1
    
    

    pylab.xticks(X+(methindex/3)* bar_width,xticks, fontsize=14, color='black')
    #pylab.legend(loc='best', prop={'size': 13})
    if legend=="out":
        pylab.legend(bbox_to_anchor=(1.0,1), loc="upper left",prop={'size': 14.5})
    else:
        pylab.legend(loc='best', prop={'size': 13})
    pylab.ylabel('Accuracy % after learning all tasks',fontsize=14)
    
    pylab.ylim(ylim)
    #pylab.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    pylab.tight_layout()
    pylab.tick_params(axis='both', which='major', labelsize=14)
    if save_img:
        pylab.axis('on')
        pylab.tight_layout()
        #plt.gca().set_position([0, 0, 1, 1])
        #pylab.savefig('{}.svg'.format(save_img) , bbox_inches='tight')
        pylab.savefig('{}.png'.format(save_img) , bbox_inches='tight')
        pylab.clf()
    else:
        pylab.show()



###############################################

results={}#torch.load("tinyImagenet.pth")
lams=[1e-6,0]

scales=[6]
lambdas=[4]
parent_exp_dir='./TINYIMAGNET_exp_dir/'#Change to yours
for normalize in [False]:
    for scale in scales:
        print("***** scale is ",scale,"**********")

        for lam in lams:
            print("***** lam is ",lam,"**********")


            for reg_lambda in lambdas:
                print("***** reg_lambda is ",reg_lambda,"**********")
                accs=[]
                forgettings=[]
                extra_str="tinyimagenet_exp"+"num_epochs_60reg_lambda_"+str(reg_lambda)+"lr_0.01b1_Falseneuron_omega_Truenormalize_"+str(normalize)+"dropout_Falsescale_"+str(scale)+"_lam"+str(lam)
               
                model_path=parent_exp_dir+'/10/SLNID/'+extra_str+'/best_model.pth.tar'
                data_dir='./TINYIMAGNET'
                for task in range(1,11):
                    try:
                        model=torch.load(model_path)     
                        dataset_path=os.path.join(data_dir,str(task),'trainval_dataset.pth.tar')
                        task_model_path=parent_exp_dir+str(task)+'/SLNID/'+extra_str+'/best_model.pth.tar'

                        acc=test_seq_task_performance(task_model_path,model_path,dataset_path)   
                        accorg=test_model(task_model_path,dataset_path) 
                        print('task number ',task,' accuracy is %.2f'%acc, 'compared to %.2f'%accorg)  
                        accs.append(acc)
                        forgettings.append(accorg-acc)
                    except:
                        traceback.print_exc()
                results["SLNID"+str(lam)+"lambda_"+str(reg_lambda)+"scale_"+str(scale)]=[accs,forgettings]





seqacc={}
keys=list(results.keys())
avg={}
for key in keys:
    seqacc[key]=results[key][0]
    avg[key]=sum(seqacc[key])/len(seqacc[key])
	
hatches=["",""] 
colors=['C0','C2']
labels=['SNID: '+str(round(avg[keys[0]],2)),"No-Reg: "+str(round(avg[keys[1]],2))]

plot_multibar( seqacc,keys,colors,labels,hatches,save_img="tinyimagent_bars",ylim=(40, 66),bar_widthT= 0.12,bar_width = 0.14,legend="best")

torch.save(results,"SampleCode_tinyImagenet_results.pth")

