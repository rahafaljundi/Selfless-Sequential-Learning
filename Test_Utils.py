#Main test functions
from __future__ import print_function, division

import torch
from torch.autograd import Variable
import numpy as np


def test_model(model_path, dataset_path, batch_size=600, print_classes_acc=False):
    model = torch.load(model_path)
    if isinstance(model, dict):
        model = model['model']
    model = model.cuda()

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['val']:
        images, labels = data
        images = images.cuda()
        images = images.squeeze()
        labels = labels.cuda()
        outputs, x = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        # pdb.set_trace()
        for i in range(len(predicted)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        del images
        del labels
        del outputs
        del data
    if print_classes_acc:

        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
                dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy = np.sum(class_correct) * 100 / np.sum(class_total)
    print('Accuracy: ' + str(accuracy))
    return accuracy


def test_seq_task_performance(previous_model_path,current_model_path,dataset_path,check=0):
    current_model_ft=torch.load(current_model_path)
    if isinstance(current_model_ft, dict): 
        current_model_ft=current_model_ft['model']
    
    previous_model_ft=torch.load(previous_model_path)
    if isinstance(previous_model_ft, dict): 
        previous_model_ft=previous_model_ft['model']

    last_layer_index=str(len(previous_model_ft.module.classifier._modules)-1)
    
    current_model_ft.module.classifier._modules[last_layer_index] = previous_model_ft.module.classifier._modules[last_layer_index]
    
    temp_path='tobetested.pth.tar'
    torch.save(current_model_ft,temp_path)
    acc=test_model(temp_path,dataset_path)
    return acc

