# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   Train_vgg.py
# @Time   :   2021/1/7 22:09
from practice import models
from practice.utils import *

data = datas.CIFAR(img_size=224, batch_size=25)

vgg = models.VGG16(n_classes=10)
models.vgg.df_device = 'cuda:1'
models.vgg.train_vgg(data.trainLoader)
