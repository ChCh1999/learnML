#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 21:20
# @File  : linear_reg.py
# @Author: Ch
# @Date  : 2020/12/13
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import datasets

import util


def pred(w, x, b=torch.tensor([0])):
    res = torch.matmul(w.t(), x.t())
    res = res.add(b)
    return res


def linear_reg(data: list, res, batch_size=64, lr=0.00001, max_round=5000, threshold=0.0000001):
    step = len(data) // batch_size
    batches = [data[i::step] for i in range(step)]
    batches_res = [res[i::step] for i in range(step)]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    w = torch.randn(len(data[0]), 1, device=device, dtype=torch.float64, requires_grad=True, )
    b = torch.tensor([0], device=device, dtype=torch.float64, requires_grad=True)

    loss = torch.tensor([0.], device=device, requires_grad=True)
    optimiser = torch.optim.SGD([w, b], lr=lr)
    optimiser.zero_grad()
    creation = nn.MSELoss()
    epoch = 0
    last_loss = 1

    while (abs(loss.item() - last_loss) > threshold) and epoch < max_round:
        last_loss = loss.item()
        batch = torch.tensor(batches[epoch % step], device=device, )
        batch_res = torch.tensor([batches_res[epoch % step]], device=device, )
        pred_res = pred(w, batch, b)
        loss = creation(batch_res, pred_res)
        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()
        print("epoch", epoch, loss.item()/batch_size)
        epoch += 1
    return w.to('cpu'), b.to('cpu')


boston = datasets.load_boston(return_X_y=True)

data = util.normalize([d[0:13] for d in boston[0]])
train_data = data[::2]
train_res = boston[1][::2]
test_data = data[1::2]
test_res = boston[1][1::2]

w, b = linear_reg(train_data, train_res, batch_size=len(train_data), lr=0.1, max_round=20000)
print(w, b)


def pred_scatter(pred_res, res, title=''):
    plt.scatter(res, pred_res, color='red')
    plt.xlabel('real price')
    plt.ylabel('predicate price')
    plt.title(title)
    max_val = max(max(pred_res), max(res))
    plt.plot([0, max_val], [0, max_val])
    plt.show()


pred_train = pred(w, torch.tensor(train_data), b).detach().numpy()[0]
pred_scatter(pred_train, train_res, 'train')
pred_test = pred(w, torch.tensor(test_data), b).detach().numpy()[0]
pred_scatter(pred_test, test_res, 'test')
creation = nn.MSELoss()
print(creation(torch.tensor(pred_test), torch.tensor(test_res)))
# [[ -8.8190], [  4.4083], [ -1.2132], [  2.2977], [ -9.0558],[ 18.5883], [ -0.8607],[-18.3859],[6.7886],[-5.7185],[-8.5996],[4.2438],[-17.2118]]
