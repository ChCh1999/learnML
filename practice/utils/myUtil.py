# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   myUtil.py
# @Time   :   2020/12/17 9:41
import datetime
import torch


def save_model(model, msg='', accuracy=0):
    model_save_path = './' + \
                      type(model).__name__ + \
                      ('_' + msg if msg else '') + \
                      datetime.datetime.now().strftime('_%m%d_%H%M') + \
                      ('_%.3f' % accuracy if accuracy else '') + '.pt'

    torch.save(model.state_dict(), model_save_path)
    return model_save_path
