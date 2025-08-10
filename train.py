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

"""
Improved Version-20250810:
- Addressed information leakage during the integration phase by utilizing an independent test set.
- Ensured strict alignment between predictions and true labels.
- Implemented support for weighted soft-voting, with weights derived from fold validation set accuracy.
- Incorporated self-contained KFold indexing and best-path parsing utilities.
"""

import os
import re
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 基础配置（按需修改即可
SEED = 709
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# training hyperparams: batch_size, epochs, learning rate, n_splits = K-fold, etc.
bs = 128
num_epochs = 30         # 实际训练可增大
init_lr = 1e-3
n_splits = 5
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

# utils functions(the newest)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def get_stratified_kfold_indices(labels, n_splits=5, random_state=709, shuffle=True):
    """
    返回 (train_idxs_list, val_idxs_list)
    每个元素都是 numpy 索引数组
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    labels = np.array(labels)
    train_idxs_list = []
    val_idxs_list = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_idxs_list.append(train_idx)
        val_idxs_list.append(val_idx)
    return train_idxs_list, val_idxs_list

def get_best_model_paths(save_dir: str, n_splits: int) -> List[str]:
    """
    找到每个 fold 对应的最佳模型文件（按 filename 中的 acc 值选择最大者）。
    假定保存格式为: fold{fold}_epoch{epoch}_acc{acc:.4f}.pth
    如果某 fold 没找到任何文件，会抛出错误提醒。
    """
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
    # ensure folds 0..n_splits-1 exist
    paths = []
    for fold in range(n_splits):
        if fold not in fold_best:
            raise FileNotFoundError(f"No saved model found for fold {fold} in {save_dir}.")
        paths.append(fold_best[fold][1])
    return paths

# 模型定义：可依据实际情况自定义
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((32,32))
        self.linear = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# 数据集与data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载数据（CIFAR-10 示例）
full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# labels 用于分层
labels = full_train.targets  # list length N

# KFold 训练（保存每 fold 最优模型，并记录每 fold 的 best_acc）
train_idxs_list, val_idxs_list = get_stratified_kfold_indices(labels, n_splits=n_splits, random_state=SEED)

best_acc_list = [0.0] * n_splits

for fold in range(n_splits):
    print(f"\n=== Fold {fold}/{n_splits-1} ===")
    # 划分
    train_ds = Subset(full_train, train_idxs_list[fold])
    # 注意：验证集需要用 test_transform 而不是带数据增强的 train_transform
    # 因为 full_train 的 transform 里包含了 train_transform，因此我们要临时包一层来覆盖 transform
    # 简单方法是从原始 CIFAR10 dataset 重建 Subset（不想深入改 dataset 的 transform 时）
    val_indices = val_idxs_list[fold]
    # 直接构造一个带 test_transform 的 CIFAR 数据集，再用 val_indices 切片
    val_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
    val_ds = Subset(val_full, val_indices)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # model & optimizer & scheduler
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    best_acc = 0.0
    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, batch_labels in train_dl:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == batch_labels).sum().item()
            total_samples += inputs.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        with torch.no_grad():
            for inputs, batch_labels in val_dl:
                inputs, batch_labels = inputs.to(device), batch_labels.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, batch_labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_corrects += (preds == batch_labels).sum().item()
                val_samples += inputs.size(0)

        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects / val_samples

        scheduler.step()

        print(f"Fold {fold} Epoch {epoch}/{num_epochs} | "
              f"Train loss {epoch_train_loss:.4f} acc {epoch_train_acc:.4f} | "
              f"Val loss {epoch_val_loss:.4f} acc {epoch_val_acc:.4f}")

        # 保存最优模型（基于 val_acc 优先，acc 相同则 loss 更低）
        if (epoch_val_acc > best_acc) or (epoch_val_acc == best_acc and epoch_val_loss < best_loss):
            best_acc = epoch_val_acc
            best_loss = epoch_val_loss
            best_epoch = epoch
            fname = f"fold{fold}_epoch{epoch}_acc{epoch_val_acc:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
            print(f"  ↳ New best for fold {fold}: acc={best_acc:.4f}, loss={best_loss:.4f}, saved {fname}")

    best_acc_list[fold] = best_acc
    print(f"Fold {fold} finished. Best val acc: {best_acc:.4f} (epoch {best_epoch})")

# 集成推理（在独立 test_set 上）
print("\n=== Ensemble inference on independent test set ===")
test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

# 找到每个 fold 的 best model
best_model_paths = get_best_model_paths(save_dir, n_splits=n_splits)
print("Best model files per fold:", best_model_paths)
print("Best val acc per fold:", best_acc_list)

all_fold_preds = []  # 每个元素是 [N, C] 的 tensor（cpu）
# 也收集真实标签（仅一次）
all_true_labels = []

# collect true labels once (guaranteed same order because test_dl shuffle=False)
with torch.no_grad():
    for inputs, labels in test_dl:
        all_true_labels.append(labels)
all_true_labels = torch.cat(all_true_labels, dim=0).numpy()  # [N]

# 对每个 fold 的最优模型，在 test_dl 上做预测
with torch.no_grad():
    for i, path in enumerate(best_model_paths):
        print(f"Loading model for fold {i}: {path}")
        model = MyModel()
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        fold_probs = []
        for inputs, _ in test_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            fold_probs.append(probs.cpu())
        fold_probs = torch.cat(fold_probs, dim=0)  # [N, C]
        all_fold_preds.append(fold_probs)

# 把 preds 堆叠为 [k, N, C]
stacked_preds = torch.stack(all_fold_preds, dim=0)  # float tensor on cpu

# 选择 soft voting 方式：
use_weighted = True  # 如果想用简单平均改成 False

if use_weighted:
    weights = np.array(best_acc_list, dtype=float)
    if weights.sum() <= 0:
        # 防止全 0（理论上不会），退回等权重
        weights = np.ones(len(best_acc_list), dtype=float)
    weights = weights / weights.sum()
    weights_t = torch.tensor(weights, dtype=stacked_preds.dtype).view(-1, 1, 1)  # [k,1,1]
    final_probs = (stacked_preds * weights_t).sum(dim=0)
else:
    final_probs = torch.mean(stacked_preds, dim=0)

pred_labels = torch.argmax(final_probs, dim=1).numpy()  # [N]

acc = accuracy_score(all_true_labels, pred_labels)
print(f"Ensemble accuracy on independent test set: {acc:.4f}")

# 可查看每 fold 在 test set 上的单模型表现：
per_fold_test_accs = []
for k in range(stacked_preds.shape[0]):
    p = torch.argmax(stacked_preds[k], dim=1).numpy()
    per_fold_test_accs.append(accuracy_score(all_true_labels, p))
print("Per-fold test accuracies:", per_fold_test_accs)

# 打印权重与验证 acc
print("fold validation accs:", best_acc_list)
if use_weighted:
    print("weights used:", weights)

# 保存最终概率结果用于后续分析
np.save(os.path.join(save_dir, "ensemble_final_probs.npy"), final_probs.numpy())
print("Saved ensemble_final_probs.npy")
