#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 22:45
# @File  : util.py
# @Author: Ch
# @Date  : 2020/12/14
def normalize(raw_data):
    dim = len(raw_data[0])
    res = [[] for _ in range(len(raw_data))]
    for i in range(dim):
        data_dim_i = [d[i] for d in raw_data]
        maximum = max(data_dim_i)
        minimum = min(data_dim_i)
        length = maximum - minimum
        for j in range(len(raw_data)):
            res[j].append((data_dim_i[j] - minimum) / length)
    return res
