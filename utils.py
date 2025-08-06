import os
import torch
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import _LRScheduler
import re

#get_stratified_kfold_indices(labels, …) 只需要提供一个标签列表即可，不管后端数据集是 Dataset 还是自定义 Dataset／DataLoader，都能把索引分好
def get_stratified_kfold_indices(labels, n_splits=5, random_state=42, shuffle=True):
    """
    Generate stratified K-fold train/validation indices.

    Args:
        labels (Sequence): list or array of labels.
        n_splits (int): number of folds.
        random_state (int): random seed.
        shuffle (bool): whether to shuffle before splitting.

    Returns:
        train_indices, val_indices: lists of index arrays for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    train_indices = []
    val_indices = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_indices.append(train_idx)
        val_indices.append(val_idx)
    return train_indices, val_indices

#mixup函数仅用于分类任务
#mixup_data 默认是针对单标签分类（y 为一维整型标签）设计的。如果你的 custom model 用的是多标签分类、回归或检测任务：

# 回归：Mixup 直接对连续 y 也能用，只是你要在 loss 里按照 y_a*lam + y_b*(1-lam) 计算
def mixup_data(x, y, alpha=0.4, device='cuda'):
    """
    Perform MixUp augmentation on a batch.

    Args:
        x (Tensor): input batch of images.
        y (Tensor): target labels (LongTensor).
        alpha (float): mixup alpha parameter.
        device (str): device for tensors.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

#CosineAnnealingScheduler 是继承自 PyTorch 标准的 _LRScheduler，只要你的训练循环在每个 step 或每个 epoch 都调用 scheduler.step()，就能按照设定好的余弦周期自动更新 lr
class CosineAnnealingScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler.

    Args:
        optimizer: torch optimizer.
        T_max (int): maximum number of iterations.
        eta_min (float): minimum learning rate.
        last_epoch (int): the index of last epoch.
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def save_best_model(model, save_dir, model_name, val_acc, val_loss,
                    best_acc, best_loss):
    """
    Save model checkpoint if current metrics are improved.

    Args:
        model (nn.Module): the model to save.
        save_dir (str): directory to save checkpoints.
        model_name (str): base name for saved file.
        val_acc (float): current validation accuracy.
        val_loss (float): current validation loss.
        best_acc (float): best recorded accuracy so far.
        best_loss (float): best recorded loss so far.

    Returns:
        new_best_acc, new_best_loss: updated best metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    new_best_acc, new_best_loss = best_acc, best_loss
    filepath = os.path.join(save_dir, f"{model_name}_best.pth")

    improved = False
    # prioritize accuracy, then loss
    if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
        torch.save(model.state_dict(), filepath)
        new_best_acc = val_acc
        new_best_loss = val_loss
        improved = True

    return new_best_acc, new_best_loss, improved

#k折处理后，找到最优模型

def get_best_model_paths(checkpoint_dir, k=5):
    """
    自动从保存的多个.pth文件中找出每个fold验证准确率最高的模型
    """
    model_paths = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    # 用字典保存每个fold最好的模型路径
    best_models = {}

    # 正则匹配 fold 和 acc
    pattern = r"fold(\d+)_epoch\d+_acc([\d\.]+)\.pth"

    for fname in model_paths:
        match = re.match(pattern, fname)
        if match:
            fold = int(match.group(1))
            acc = float(match.group(2))
            if (fold not in best_models) or (acc > best_models[fold][1]):
                best_models[fold] = (fname, acc)

    # 返回排序好的路径列表
    return [os.path.join(checkpoint_dir, best_models[i][0]) for i in range(k)]


# #使用示例
# from training_utils import (
#     get_stratified_kfold_indices,
#     mixup_data,
#     CosineAnnealingScheduler,
#     save_best_model
# )

# # 1. 划分索引
# train_idx_list, val_idx_list = get_stratified_kfold_indices(csv['label'], n_splits=5)

# # 2. MixUp 扩增
# mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device=device)

# # 3. 创建调度器
# scheduler = CosineAnnealingScheduler(optimizer, T_max=epochs*len(train_dl), eta_min=lr*0.01)

# # 4. 保存最优模型
# best_acc, best_loss = 0.0, float('inf')
# best_acc, best_loss, improved = save_best_model(model, save_dir, model_name,
#                                                 val_acc, val_loss,
#                                                 best_acc, best_loss)
# if improved:
#     print("模型性能提升，已保存最优权重。")
