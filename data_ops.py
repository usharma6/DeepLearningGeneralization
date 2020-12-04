import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import struct
import PIL
import torchvision.models as models



def AugmentedImageNet(train_transforms=[], val_transforms=[], test_transforms=[], batch_size=100):

    #!wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    #!unzip tiny-imagenet-200.zip

    target_folder = './tiny-imagenet-200/val/'
    test_folder   = './tiny-imagenet-200/test/'

    #os.mkdir(test_folder)
    val_dict = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob('./tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
        if not os.path.exists(test_folder + str(folder)):
            os.mkdir(test_folder + str(folder))
            os.mkdir(test_folder + str(folder) + '/images')


    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if len(glob.glob(target_folder + str(folder) + '/images/*')) <25:
            dest = target_folder + str(folder) + '/images/' + str(file)
        else:
            dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir('./tiny-imagenet-200/val/images')

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    data_dir = 'tiny-imagenet-200/'
    num_workers = {
        'train' : 100,
        'val'   : 0,
        'test'  : 0
    }
    image_datasets = {x: dataset.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                                 shuffle=True, pin_memory =True)
                  for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes
