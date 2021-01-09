# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   Train_resnet.py
# @Time   :   2021/1/8 18:26
import practice
import practice.config as config
from practice.utils import train, datas
import os

data = datas.CIFAR(img_size=224)
res = practice.models.resnet18(pretrained=True, num_classes=10)
train.EPOCH = 50
train.df_device = 'cuda'
train.train_classifier(res, data.trainLoader, os.path.join(config.model_root, 'resnet.pt'))
