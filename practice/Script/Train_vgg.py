# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   Train_vgg.py
# @Time   :   2021/1/7 22:09
from practice import models, config
from practice.utils import *
import os

data = datas.CIFAR(img_size=224, batch_size=25)

vgg = models.VGG16(n_classes=10)

train.df_device = 'cuda'
train.train_classifier(vgg, data.trainLoader, os.path.join(config.model_root, 'vgg.pt'))
