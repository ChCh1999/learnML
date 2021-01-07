# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   Train_Alex.py
# @Time   :   2021/1/7 16:49
from practice import models
from practice.utils import *

data = datas.CIFAR(img_size=227)

Alex = models.AlexNet(num_classes=10)
Alex.load_pretrained_model()

train.EPOCH = 50
train.train_classifier(Alex, data.trainLoader, 'model/alex.pt')
