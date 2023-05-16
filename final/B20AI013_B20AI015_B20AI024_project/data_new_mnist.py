import torch
import numpy as np
import os
from torch.utils.data import Dataset
from multiprocessing import Pool
from functools import partial
# from scipy.misc import imread, imresize
import time
import torchvision


class DefenseDataset(Dataset):
    def __init__(self, config, phase, attack):
        # assert phase in ['train', 'test']
        # assert attack in ['fgsm', 'pgd', 'jitter']
        # self.phase = phase
        # self.attack = attack

        # self.adv_imgs = torch.load(f'dataset/{self.attack}_samples_CIFAR10.pt')[phase]
        

        # self.config = config

    def __getitem__(self, idx):

        # orig_img, label = self.dataset[idx]
        # adv_img = self.adv_imgs[idx]
        # return orig_img, adv_img, label
        
    def __len__(self):
        # return len(self.dataset)


def read_labels(filename='dev_dataset.csv'):
    f = open(filename, 'r')
    data = f.read()
    f.close()

    data = data.split('\n')
    data = data[1:-1]
    labels = dict()
    for line in data:
        line = line.split(',')
        labels[line[0]] = int(line[6])
    return labels


def augment(imgs, config, train):
    if train:
        np.random.seed(int(time.time() * 1000000) % 1000000000)

    if config.has_key('flip') and config['flip'] and train:
        stride = np.random.randint(2) * 2 - 1
        for i in range(len(imgs)):
            imgs[i] = imgs[i][:, ::stride, :]

    if config.has_key('crop_size'):
        crop_size = config['crop_size']
        if train:
            h = np.random.randint(imgs[0].shape[0] - crop_size[0] + 1)
            w = np.random.randint(imgs[0].shape[1] - crop_size[1] + 1)
        else:
            h = int(imgs[0].shape[0] - crop_size[0]) / 2
            w = int(imgs[0].shape[1] - crop_size[1]) / 2
        for i in range(len(imgs)):
            imgs[i] = imgs[i][h:h + crop_size[0], w:w + crop_size[1], :]

    return imgs


def normalize(imgs, net_type):
    if net_type == 'inceptionresnetv2':
        for i in range(len(imgs)):
            imgs[i] = imgs[i].astype(np.float32)
            imgs[i] = 2 * (imgs[i] / 255.0) - 1.0

    else:
        mean = np.asarray([0.485, 0.456, 0.406], np.float32).reshape((1, 1, 3))
        std = np.asarray([0.229, 0.224, 0.225], np.float32).reshape((1, 1, 3))
        for i in range(len(imgs)):
            imgs[i] = imgs[i].astype(np.float32)
            imgs[i] /= 255
            imgs[i] -= mean
            imgs[i] /= std

    return imgs