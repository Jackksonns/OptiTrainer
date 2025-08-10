"""
ensemble_kfold_snapshot.py

Two strategies to choose for different targets：
- method = 'kfold_diverse'  -> K-fold + per-fold diversity (seed + augment variants)
- method = 'snapshot'       -> Snapshot ensemble (single run, save snapshots)

说明：
- 使用 CIFAR-10 做 demo（你可替换为自己的 Dataset & Model）
- 评估在独立 test_set 上（避免信息泄漏）
- ensemble 使用 soft-voting (可选加权)
"""

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

# 配置config
method = 'kfold_diverse'   # 'kfold_diverse' or 'snapshot'
SEED = 2025
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# training hyperparams
bs = 64
num_epochs = 200            # 单 fold 或单 run 的 epoch（snapshot 会在此训练周期内产生若干快照）
init_lr = 3e-3
n_splits = 5               # 用于 k-fold 可理解为n_splits = k
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

# ensemble options
use_weighted = True        # 是否按 val acc 加权（kfold 可用）
ensemble_from = 'probs'    # 'probs' 或 'logits' （若用 logits 平均，最后再 softmax）

# snapshot options (only used when method == 'snapshot')
snap_interval = 10         # 每多少 epoch 保存一个 snapshot
#比如： num_epochs=40,snap_interval=10就会得到4snapshots


# 固定种子
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)


# 模型定义（可修改为custom模型）
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


# 数据 & transform variants（通过数据增强制造不同折的模型的多样性）
# 一组可选的训练数据增强参数样例
augment_variants = [
    transforms.Compose([transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor()]),
    transforms.Compose([transforms.RandomRotation(15),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]),
    transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()]),
    transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
                        transforms.ToTensor()]),
]

test_transform = transforms.Compose([transforms.ToTensor()])

# 下载数据（CIFAR-10 demo）
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
full_train_root = './data'
test_set = torchvision.datasets.CIFAR10(root=full_train_root, train=False, download=True, transform=test_transform)

# helper to get dataset with certain transform (train=True)
def cifar10_with_transform(transform):
    return torchvision.datasets.CIFAR10(root=full_train_root, train=True, download=False, transform=transform)

# utilities
def get_stratified_kfold_indices(labels, n_splits=5, random_state=SEED, shuffle=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    labels = np.array(labels)
    train_idxs_list, val_idxs_list = [], []
    for tr, vl in skf.split(np.zeros(len(labels)), labels):
        train_idxs_list.append(tr)
        val_idxs_list.append(vl)
    return train_idxs_list, val_idxs_list

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

# 跑一次试试
def train_one_epoch(model, dl, optimizer):
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

def eval_on_dl(model, dl):
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


# 优化方法 A: K-fold + diversity
def run_kfold_diverse():
    print("Running K-fold with per-fold diversity...")
    # full train labels for stratified split
    full_train_for_labels = torchvision.datasets.CIFAR10(root=full_train_root, train=True, download=False, transform=test_transform)
    labels = full_train_for_labels.targets

    train_idxs_list, val_idxs_list = get_stratified_kfold_indices(labels, n_splits=n_splits)

    best_acc_list = [0.0] * n_splits
    saved_paths = []

    for fold in range(n_splits):
        seed = SEED + fold * 13
        set_seed(seed)
        aug = augment_variants[fold % len(augment_variants)]
        print(f"\n=== Fold {fold} | seed={seed} | augment_variant={fold % len(augment_variants)} ===")

        # train & val datasets with appropriate transforms
        train_ds = Subset(cifar10_with_transform(aug), train_idxs_list[fold])
        val_ds = Subset(cifar10_with_transform(test_transform), val_idxs_list[fold])

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

        model = MyModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

        best_acc = 0.0; best_loss = float('inf'); best_epoch = -1

        for epoch in range(1, num_epochs+1):
            tr_loss = train_one_epoch(model, train_dl, optimizer)
            val_loss, val_acc = eval_on_dl(model, val_dl)
            scheduler.step()
            print(f"Fold{fold} Epoch {epoch}/{num_epochs} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
            if (val_acc > best_acc) or (val_acc == best_acc and val_loss < best_loss):
                best_acc = val_acc; best_loss = val_loss; best_epoch = epoch
                fname = f"fold{fold}_epoch{epoch}_acc{val_acc:.4f}.pth"
                torch.save(model.state_dict(), os.path.join(save_dir, fname))
                print("  ——————> saved", fname)
        best_acc_list[fold] = best_acc
        print(f"Fold {fold} best val acc {best_acc:.4f} (epoch {best_epoch})")

    # evaluate ensemble on independent test set
    print("\n=== Ensemble evaluation on independent test set ===")
    test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    best_paths = get_best_model_paths(save_dir, n_splits)
    print("best_paths:", best_paths)
    print("best val accs:", best_acc_list)

    all_fold_preds = []
    for i, path in enumerate(best_paths):
        print("Loading", path)
        m = MyModel().to(device)
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt)
        m.eval()
        fold_probs = []
        with torch.no_grad():
            for x,y in test_dl:
                x = x.to(device)
                out = m(x)
                if ensemble_from == 'logits':
                    fold_probs.append(out.cpu())     # store logits
                else:
                    fold_probs.append(F.softmax(out, dim=1).cpu())  # store probs
        fold_tensor = torch.cat(fold_probs, dim=0)  # [N, C] on cpu
        all_fold_preds.append(fold_tensor)

    stacked = torch.stack(all_fold_preds, dim=0)  # [k, N, C] (if logits, it's logits)
    # weighting
    if use_weighted:
        w = np.array(best_acc_list, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        w_t = torch.tensor(w, dtype=stacked.dtype).view(-1,1,1)
        if ensemble_from == 'logits':
            avg_logits = (stacked * w_t).sum(dim=0)
            final_probs = F.softmax(avg_logits, dim=1)
        else:
            final_probs = (stacked * w_t).sum(dim=0)
    else:
        if ensemble_from == 'logits':
            avg_logits = torch.mean(stacked, dim=0)
            final_probs = F.softmax(avg_logits, dim=1)
        else:
            final_probs = torch.mean(stacked, dim=0)

    pred_labels = torch.argmax(final_probs, dim=1).numpy()

    # true labels
    trues = []
    with torch.no_grad():
        for x,y in test_dl:
            trues.append(y)
    trues = torch.cat(trues, dim=0).numpy()

    acc = accuracy_score(trues, pred_labels)
    print("Ensemble test acc:", acc)

    # per-model test acc & pairwise disagreement
    per_fold_accs = []
    pred_np = []
    for k in range(stacked.shape[0]):
        if ensemble_from == 'logits':
            p = torch.argmax(F.softmax(stacked[k], dim=1), dim=1).numpy()
        else:
            p = torch.argmax(stacked[k], dim=1).numpy()
        pred_np.append(p)
        per_fold_accs.append(accuracy_score(trues, p))
    print("Per-fold test accs:", per_fold_accs)

    K = len(pred_np)
    dis = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            dis[i,j] = (pred_np[i] != pred_np[j]).mean()
    print("Pairwise disagreement:\n", dis)
    return


# 方法 B: Snapshot ensemble
def run_snapshot():
    print("Running Snapshot ensemble (single-run snapshots)...")
    # use the same training transform for full train
    train_ds = cifar10_with_transform(augment_variants[0])  # 数据增强方案可视具体情况修改
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_ds = torchvision.datasets.CIFAR10(root=full_train_root, train=True, transform=test_transform, download=False)
    # use a small held-out val for tracking (optional)
    # or evaluate snapshots on independent test set only
    test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = MyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
    # you can use CosineAnnealingWarmRestarts or manual lr schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    snapshot_paths = []
    for epoch in range(1, num_epochs+1):
        tr_loss = train_one_epoch(model, train_dl, optimizer)
        scheduler.step()
        print(f"Epoch {epoch}/{num_epochs} | tr_loss {tr_loss:.4f}")
        # save snapshot at interval
        if epoch % snap_interval == 0:
            fname = f"snapshot_epoch{epoch}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, fname))
            snapshot_paths.append(os.path.join(save_dir, fname))
            print("Saved snapshot:", fname)

    # evaluate snapshots on test set
    all_preds = []
    for p in snapshot_paths:
        print("Loading snapshot", p)
        m = MyModel().to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt)
        m.eval()
        probs_list = []
        with torch.no_grad():
            for x,y in test_dl:
                x = x.to(device)
                out = m(x)
                if ensemble_from == 'logits':
                    probs_list.append(out.cpu())
                else:
                    probs_list.append(F.softmax(out, dim=1).cpu())
        all_preds.append(torch.cat(probs_list, dim=0))

    stacked = torch.stack(all_preds, dim=0)  # [snapshots, N, C]

    if ensemble_from == 'logits':
        avg_logits = torch.mean(stacked, dim=0)
        final_probs = F.softmax(avg_logits, dim=1)
    else:
        final_probs = torch.mean(stacked, dim=0)

    pred_labels = torch.argmax(final_probs, dim=1).numpy()
    trues = []
    with torch.no_grad():
        for x,y in test_dl:
            trues.append(y)
    trues = torch.cat(trues, dim=0).numpy()

    acc = accuracy_score(trues, pred_labels)
    print("Snapshot ensemble test acc:", acc)

    # per-snapshot accs
    per_accs = []
    pred_np = []
    for k in range(stacked.shape[0]):
        if ensemble_from == 'logits':
            p = torch.argmax(F.softmax(stacked[k], dim=1), dim=1).numpy()
        else:
            p = torch.argmax(stacked[k], dim=1).numpy()
        pred_np.append(p)
        per_accs.append(accuracy_score(trues, p))
    print("Per-snapshot accs:", per_accs)

    K = len(pred_np)
    dis = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            dis[i,j] = (pred_np[i] != pred_np[j]).mean()
    print("Pairwise disagreement:\n", dis)
    return

# main
if __name__ == '__main__':
    if method == 'kfold_diverse':
        run_kfold_diverse()
    elif method == 'snapshot':
        run_snapshot()
    else:
        raise ValueError("Invalid method")
