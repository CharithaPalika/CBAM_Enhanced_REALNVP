
import torch
import torch.nn as nn
import argparse
import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils
import argparse
import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def load(dataset):
    """Load dataset.

    Args:
        dataset: name of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    if dataset == 'cifar10':    # 3 x 32 x 32
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        train_set = datasets.CIFAR10('../../data/CIFAR10', 
            train=True, download=True, transform=transform)
        [train_split, val_split] = data.random_split(train_set, [46000, 4000])

    elif dataset == 'celeba':   # 3 x 218 x 178
        data_info = DataInfo(dataset, 3, 64)
        def CelebACrop(images):
            return transforms.functional.crop(images, 40, 15, 148, 148)
        transform = transforms.Compose(
            [CelebACrop, 
             transforms.Resize(64),
             transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor()])
        print('celebA')
        train_set = datasets.ImageFolder('/kaggle/input/celeba-dataset/img_align_celeba', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [0.75, 0.25]) #[150000, 12770])


    elif dataset == 'imnet32':
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet32/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    elif dataset == 'imnet64':
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet64/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    return train_split, val_split, data_info
