#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 22:35
# @File  : linear_modal.py
# @Author: Ch
# @Date  : 2020/12/14
import torch
from sklearn import datasets as sk_dataset
from torch import nn
import torch.utils.data as torch_data
import matplotlib.pyplot as plt


class LinearReg(nn.Module):
    """
    linear eg model
    """

    def __init__(self, feature_count, lr=0.1):
        super(LinearReg, self).__init__()
        self.fc = nn.Linear(feature_count, 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.loss_function = torch.nn.MSELoss()
        nn.init.normal_(self.fc.weight, mean=0, std=1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


# confirm device
# df_device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
df_device = torch.device('cpu')
# read dataset
boston_dataset = sk_dataset.load_boston(return_X_y=True)


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


data = normalize([d[0:13] for d in boston_dataset[0]])
y = boston_dataset[1]
train_data = data[::2]
train_res = y[::2]
test_data = data[1::2]
test_res = y[1::2]
# create dataLoader
Boston_dataset = torch_data.TensorDataset(
    torch.tensor(train_data, dtype=torch.float, device=df_device),
    torch.tensor(train_res, dtype=torch.float, device=df_device)
)
batch_size = 80
data_loader = torch_data.DataLoader(Boston_dataset, batch_size=batch_size, shuffle=True)
# train
net = LinearReg(13)
net.to(df_device)
num_epochs = 10
l = 0
for epoch in range(1, num_epochs + 1):
    for X, y in data_loader:
        output = net(X)
        l = net.loss_function(output, y.view(-1, 1))
        net.optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        net.optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item() / batch_size))

# predicate
pred_res = net(torch.tensor(test_data, dtype=torch.float)).detach().numpy()
# plot
pred_res = pred_res.reshape(1, -1)[0]
plt.scatter(test_res, pred_res, c='r')
plt.xlabel('real price')
plt.ylabel('predicate price')
plt.title('test')
max_point = max(max(pred_res), max(test_res))
plt.plot([0, max_point], [0, max_point])
plt.show()
