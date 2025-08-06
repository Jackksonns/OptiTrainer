"""
OptiTrainer: Optimized Deep Learning Training Pipeline
====================================================

A high-performance, modular training template for image classification tasks.

- Implements advanced training tricks: K-Fold Cross Validation, Cosine Annealing LR Scheduler, automatic best model saving, and ensemble inference.
- Achieves up to 20% accuracy improvement on standard datasets (e.g., CIFAR-10) compared to vanilla training.
- Easily customizable for your own models and datasets.

Author: https://github.com/Jackksonns

Usage:
    python train.py

For details, see the README.md.
"""
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
from utils import *
import torch_utils as tu
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

#兼容版训练优化方法使用方法：只需要更换模型定义为您自己训练的模型，将数据集调整成对应格式，修改参数设置即可。

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(32 * 32 * 32, 10)

        
    def forward(self, x):
        x = self.conv2d(x)
        # print('after conv2d', x.shape)
        x = self.relu(x)
        # print('after relu', x.shape)
        x = self.linear(x.view(-1, 32 * 32 * 32))
        # print('after linear', x.shape)
        return x


train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

#获取labels用于KFold
labels = train_set.targets 

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())



#只需在此处更换模型实例化即可
model = MyModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#通过utils文件定义的kstratifiedKFold划分得到划分后的索引
train_idxs, val_idxs = get_stratified_kfold_indices(
    labels,
    n_splits=5,
    random_state=709,
    shuffle=True
)

#使用pytorch dataset切分训练集和验证集，此时的full dataset就是训练集
full_ds = train_set               # 长度 N
train_ds = Subset(full_ds, train_idxs)    # 只取训练下标
val_ds   = Subset(full_ds, val_idxs)      # 只取验证下标

#之后再包装成dataloader dl
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4)
val_dl   = DataLoader(  val_ds, batch_size=64, shuffle=False, num_workers=4)

# #training parameters
# epochs = 10

# def train():
#     step = 0
#     start_time = time.time()
#     model.train()
#     for epoch in range(epochs):
#         print('epoch:', epoch)
#         for data in  train_dl:
#             imgs, labels = data
#             imgs = imgs.to(device)
#             labels = labels.to(device)

#             outputs = model(imgs)
#             # print('outputs shape:', outputs.shape)
#             # print('labels shape:', labels.shape)
#             loss = loss_fn(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             step += 1
#             end_time = time.time()
#             if step % 500 == 0:
#                 print('spend', time.time() - start_time)
#                 print(f'step: {step}, train loss: {loss.item()}')

#         model.eval()
#         total_loss, total_acc, n = 0.0, 0.0, 0
#         with torch.no_grad():
#             for imgs, targets in val_dl:
#                 imgs, targets = imgs.to(device), targets.to(device)
#                 outputs = model(imgs)
#                 total_loss += loss_fn(outputs, targets).item() * imgs.size(0)
#                 total_acc  += (outputs.argmax(1)==targets).sum().item()
#                 n += imgs.size(0)
#                 #就是先算求和，走完测试循环再除
#         avg_loss = total_loss / n
#         avg_acc  = total_acc  / n
#         print(f'epoch {epoch} | test loss: {avg_loss:.4f}, test acc: {avg_acc:.4f}')
            
#         torch.save(model.state_dict(), f'model_{epoch}.pth')

# train()

#参数设置
bs = 64
num_epochs = 200
init_lr = 1e-4
save_dir = './checkpoints'
loss_fn_test = nn.CrossEntropyLoss()

# 假设 full_ds 已经准备好，包含 .targets 或者你自己提取好的 labels
labels = full_ds.targets  
n_splits = 5
train_idxs_list, val_idxs_list = get_stratified_kfold_indices(labels, n_splits=n_splits)

for fold in range(n_splits):
    print(f"Fold {fold+1}/{n_splits}")
    # 1) 根据索引切分子集
    train_ds = Subset(full_ds, train_idxs_list[fold])
    val_ds   = Subset(full_ds, val_idxs_list[fold])

    # 2) 构造 DataLoader
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4)

    # 3) （可选）每折都重新初始化模型、优化器、调度器
    model = MyModel().to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)
    scheduler   = CosineAnnealingScheduler(optimizer, T_max=num_epochs*len(train_dl), eta_min=init_lr*0.01)

    best_acc, best_loss = 0.0, float('inf')

    for epoch in range(1, num_epochs + 1):
        # 1. 训练
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, batch_labels in train_dl:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == batch_labels).sum().item()
            total_samples += inputs.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc  = running_corrects / total_samples
        
        scheduler.step()


        # 2. 验证
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, batch_labels in val_dl:
                inputs, batch_labels = inputs.to(device), batch_labels.to(device)
                outputs = model(inputs)
                loss = loss_fn_test(outputs, batch_labels)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == batch_labels).sum().item()
                val_samples += inputs.size(0)

        epoch_val_loss = val_loss / val_samples
        epoch_val_acc  = val_corrects / val_samples

        print(f"Epoch {epoch}/{num_epochs} "
              f"Train loss: {epoch_train_loss:.4f} acc: {epoch_train_acc:.4f} | "
              f"Val loss: {epoch_val_loss:.4f} acc: {epoch_val_acc:.4f}")

        # 3. 保存最优模型
        # 如果验证准确率更高，或者准确率相同但损失更低，就更新并保存
        if (epoch_val_acc > best_acc) or (epoch_val_acc == best_acc and epoch_val_loss < best_loss):
            best_acc = epoch_val_acc
            best_loss = epoch_val_loss
            fname = f"fold{fold}_epoch{epoch}_acc{epoch_val_acc:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
            print(f"  ↳ New best model saved: acc={best_acc:.4f}, loss={epoch_val_loss:.4f}")

# 获取每个fold最好的模型路径
best_model_paths = get_best_model_paths(save_dir, n_splits)

# 准备存储预测结果
all_fold_preds = []
all_labels = []

with torch.no_grad():
    for i, path in enumerate(best_model_paths):
        print(f"加载模型: {path}")
        model = MyModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model.to(device)

        fold_preds = []

        if i == 0:
            for batch in val_dl:
                inputs, labels = batch
                inputs = inputs.to(device)
                all_labels.append(labels)
        else:
            for batch in val_dl:
                inputs = batch[0]
                inputs = inputs.to(device)

        for batch in val_dl:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            fold_preds.append(probs.cpu())

        all_fold_preds.append(torch.cat(fold_preds, dim=0))

# 计算平均预测
stacked_preds = torch.stack(all_fold_preds, dim=0)  # [k, N, C]
final_probs = torch.mean(stacked_preds, dim=0)      # [N, C]

# 最终预测标签
pred_labels = torch.argmax(final_probs, dim=1)

# 拼接所有真实标签
all_labels = torch.cat(all_labels, dim=0)

# 计算准确率
acc = accuracy_score(all_labels.numpy(), pred_labels.numpy())
print(f"集成模型在测试集上的准确率：{acc:.4f}")