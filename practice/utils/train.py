# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   train.py
# @Time   :   2021/1/3 21:04
import torch
import torch.nn as nn
import tqdm
import os

# df_device = torch.device('cpu')
df_device = torch.device('cuda')


def train_classifier(net, trainLoader, save_path='model/net.pkl',
                     LEARNING_RATE=0.01,
                     EPOCH=50):
    net.to(df_device)

    # Loss, Optimizer & Scheduler
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # init save path
    par_dir = os.path.dirname(save_path)
    if par_dir and not os.path.isdir(par_dir):
        os.makedirs(par_dir)

    # Train the model
    for epoch in range(EPOCH):

        avg_loss = 0
        cnt = 0
        for images, labels in tqdm.tqdm(trainLoader, desc="E%d" % epoch):
            images = images.to(df_device)
            labels = labels.to(df_device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            _, outputs = net(images)
            loss = cost(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            # print("[E: %d
            # ] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
            loss.backward()
            optimizer.step()
        scheduler.step(avg_loss)
        print('e%d: loss %f' % (epoch, avg_loss / cnt))
        # Save the Trained Model
        torch.save(net.state_dict(), save_path)
    return net


def test_classifier(net, testLoader):
    # Test the model
    net.eval()
    net.to(df_device)
    correct = 0
    total = 0

    for images, labels in testLoader:
        images = images.to(df_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        print('correct:%d acc:%.5f' % (correct, correct / total))
        # print(predicted, labels, correct, total)
    print("avg acc: %f" % (correct / total))
