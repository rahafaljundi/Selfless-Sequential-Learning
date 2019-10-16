
# coding: utf-8

# In[2]:

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import os
import os.path
import bisect
import warnings

from torch._utils import _accumulate
from torchvision import datasets, models, transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx,file_list):
    images = []
    #print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        #print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file=target+'/'+fname;
                    #print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolderTrainVal(datasets.ImageFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    #UPDATE,4thJul18:
            If calsses is given, other classes in the folder will be ignored
    """

    def __init__(self, root,files_list, transform=None, target_transform=None,
                 loader=default_loader,classes=None):
        if classes is  None:
            classes, class_to_idx = find_classes(root)
        else:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print(root)
        imgs = make_dataset(root, class_to_idx,files_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


# Split a dataset into train and val randomly:
from itertools import accumulate


class ImageFolder_Subset(ImageFolderTrainVal):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.classes=dataset.classes   
        self.indices = indices
        self.class_to_idx =dataset.class_to_idx 
        self.root = dataset.root
        self.loader = dataset.loader
        self.transform=dataset.transform
        self.target_transform =dataset.target_transform
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(sum(lengths))
    return [ImageFolder_Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]


class ConcatDatasetLabels( torch.utils.data.ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
        the output labels are shifted by the dataset index which differs from the pytorch implementation that return the original labels
    """

    

    def __init__(self, datasets,classes_len):
        super(ConcatDatasetLabels, self).__init__(datasets)
        self.cumulative_classes_len =list(accumulate (classes_len)  )
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            img,label= self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            img,label= self.datasets[dataset_idx][sample_idx]
            label=label+self.cumulative_classes_len[dataset_idx-1]
        return img,label



