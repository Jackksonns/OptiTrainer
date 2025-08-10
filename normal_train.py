import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import torchvision.transforms as transforms
import torchvision
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2
from scipy import stats
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
import ttach as tta

#可自定义模型架构
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((8,8))
        self.fc = nn.Linear(32*8*8, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_dl = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)

#只需在此处更换模型实例化即可
model = MyModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)


#training parameters
epochs = 200

def train():
    step = 0
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        print('epoch:', epoch)
        for data in  train_dl:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            # print('outputs shape:', outputs.shape)
            # print('labels shape:', labels.shape)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            end_time = time.time()
            if step % 500 == 0:
                print('spend', time.time() - start_time)
                print(f'step: {step}, train loss: {loss.item()}')

        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                total_loss += loss_fn(outputs, targets).item() * imgs.size(0)
                total_acc  += (outputs.argmax(1)==targets).sum().item()
                n += imgs.size(0)
                #就是先算求和，走完测试循环再除
        avg_loss = total_loss / n
        avg_acc  = total_acc  / n
        print(f'epoch {epoch} | test loss: {avg_loss:.4f}, test acc: {avg_acc:.4f}')
            
        # torch.save(model.state_dict(), f'model_{epoch}.pth')
        #目前不保存模型，仅看效果

train()
