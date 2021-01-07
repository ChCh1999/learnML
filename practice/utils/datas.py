# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   datas.py
# @Time   :   2021/1/3 20:49
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import practice
import os

data_root = os.path.join(os.path.dirname(practice.__file__), 'data')


class CIFAR:
    def __init__(self, img_size=224, batch_size=4, is_download=False):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        data_path = os.path.join(data_root, 'cifar')
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                 download=is_download, transform=transform)
        self.trainLoader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=2)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                download=is_download, transform=transform)
        self.testLoader = DataLoader(test_set, batch_size=batch_size,
                                     shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VocDetection:
    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize([500, 500]),
            transforms.ToTensor()
        ])
        data_path = os.path.join(data_root, 'voc')
        train_data = torchvision.datasets.VOCDetection(
            root=data_path,
            year='2007',
            image_set='train',
            download=False,
            transform=transform
        )

        self.train_loader = DataLoader(train_data)


if __name__ == '__main__':
    # c = CIFAR()
    pass
