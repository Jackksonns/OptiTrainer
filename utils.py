import os
import re
import random
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def get_best_model_paths(save_dir: str, n_splits: int) -> List[str]:
    files = os.listdir(save_dir)
    fold_best = {}
    pattern = re.compile(r'fold(\d+)_epoch(\d+)_acc([0-9.]+)\.pth')
    for f in files:
        m = pattern.match(f)
        if m:
            fold_idx = int(m.group(1))
            acc = float(m.group(3))
            prev = fold_best.get(fold_idx)
            if (prev is None) or (acc > prev[0]):
                fold_best[fold_idx] = (acc, os.path.join(save_dir, f))
    paths = []
    for fold in range(n_splits):
        if fold not in fold_best:
            raise FileNotFoundError(f"No saved model for fold {fold} in {save_dir}")
        paths.append(fold_best[fold][1])
    return paths

# helper to get dataset with certain transform (train=True)
#此函数完全可以根据实际情况修改，自定义数据集请自定义dataset，再用内置dataloader读取即可（transform is important）
def cifar10_with_transform(transform, root):
    return torchvision.datasets.CIFAR10(root, train=True, download=False, transform=transform)
# 跑一次试试
def train_one_epoch(model, dl, optimizer, device):
    model.train()
    total = 0; loss_sum = 0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total += x.size(0)
        loss_sum += loss.item() * x.size(0)
    return loss_sum / total

def eval_on_dl(model, dl, device):
    model.eval()
    total = 0; correct = 0; loss_sum = 0
    with torch.no_grad():
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            loss_sum += loss.item() * x.size(0)
    return loss_sum/total, correct/total


