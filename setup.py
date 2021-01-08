# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   setup.py
# @Time   :   2021/1/3 21:00
from setuptools import setup, find_packages

requirements = ["torch", "torchvision"]
setup(
    name='practice',  # 包名字
    version='1.0',  # 包版本
    description='ml practice',
    packages=find_packages(exclude=("configs", "tests", "model", "data")),  # 包
)
