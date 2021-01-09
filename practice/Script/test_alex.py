# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   test_alex.py
# @Time   :   2021/1/7 21:19
from practice import models
from practice import utils
import sys
import torch

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('please given model path')
    path = sys.argv[1]
    data = utils.datas.CIFAR().testLoader
    alex = models.AlexNet(num_classes=10)
    alex.load_state_dict(torch.load(path))
    utils.train.test_classifier(alex, data)
