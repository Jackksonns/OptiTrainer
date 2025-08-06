# OptiTrainer: Optimized Training Pipeline for Deep Learning

## Project Overview

OptiTrainer is designed to significantly enhance the training process and performance of deep learning models, especially for image classification tasks. Experiments demonstrate that, on standard datasets such as CIFAR-10, the optimized training pipeline can improve model accuracy by up to 20% compared to conventional training routines.

## Key Features

- **Standard Training Pipeline (`normal_train.py`)**: A baseline single-run training and validation script for easy understanding and comparison.
- **Optimized Training Pipeline (`train.py`)**: Integrates multiple advanced training strategies:
  - K-Fold Cross Validation for robust model evaluation and improved generalization
  - Cosine Annealing Learning Rate Scheduler for better convergence
  - Automatic best model checkpointing for each fold
  - Ensemble inference across folds for further accuracy boost
- Fully customizable for user-defined models and datasets, supporting a wide range of classification tasks.

## Quick Start

### 1. Clone the Repository

```bash
# Clone OptiTrainer
https://github.com/Jackksonns/OptiTrainer.git
cd OptiTrainer
```

### 2. Data Preparation

- CIFAR-10 is used by default; the script will automatically download it on first run.
- For custom datasets, organize your data in a PyTorch-compatible format and modify the data loading section in the scripts accordingly.

### 3. Run Training Scripts

- Standard training:

  ```bash
  python normal_train.py
  ```

- Optimized training:

  ```bash
  python train.py
  ```

## Detailed Feature Description

### Core Components of the Optimized Pipeline (`train.py`)

1. **K-Fold Cross Validation**
   - Utilizes `get_stratified_kfold_indices(labels, n_splits=5, random_state=709, shuffle=True)`
   - Automatically splits the dataset into K folds, iteratively using one fold for validation and the rest for training, enhancing model generalization.
2. **Cosine Annealing Learning Rate Scheduler**
   - `CosineAnnealingScheduler(optimizer, T_max, eta_min)`
   - Dynamically adjusts the learning rate during training for improved convergence.
3. **Automatic Best Model Saving**
   - Saves model weights whenever validation accuracy improves during each fold.
4. **Ensemble Inference**
   - Aggregates predictions from the best models of all folds, averaging results for superior accuracy.

### Main Parameter Explanations

- `get_stratified_kfold_indices(labels, n_splits, random_state, shuffle)`
  - `labels`: List/array of all sample labels
  - `n_splits`: Number of folds (default: 5)
  - `random_state`: Random seed for reproducibility
  - `shuffle`: Whether to shuffle data before splitting
  - **Returns**: `train_idxs_list`, `val_idxs_list` (each a list of K index arrays)
- `CosineAnnealingScheduler(optimizer, T_max, eta_min)`
  - `optimizer`: PyTorch optimizer
  - `T_max`: Maximum number of iterations
  - `eta_min`: Minimum learning rate


## Experimental Results

- On CIFAR-10, OptiTrainer achieves up to 20% higher accuracy compared to the standard training pipeline.

  - **Note:** For experimental validation, we used a simple custom model consisting of a small convolutional layer followed by a linear layer. While the training accuracy of each fold was similar, the ensemble prediction strategy in OptiTrainer provided a significant breakthrough in final test accuracy. This highlights the strength of the ensemble-based approach, even when using basic model architectures.

- ```bash
  #mormal training process  --result
  epoch 199 | test loss: 1.1417, test acc: 0.6009
  
  #OptiTrainer training process  --result
  加载模型: ./checkpoints/fold0_epoch90_acc0.6125.pth
  /root/autodl-tmp/train.py:227: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    model.load_state_dict(torch.load(path, map_location=device))
  加载模型: ./checkpoints/fold1_epoch81_acc0.6133.pth
  加载模型: ./checkpoints/fold2_epoch83_acc0.6245.pth
  加载模型: ./checkpoints/fold3_epoch93_acc0.6074.pth
  加载模型: ./checkpoints/fold4_epoch112_acc0.6115.pth
  集成模型在测试集上的准确率：0.8094
  ```

- Actual improvement may vary depending on the dataset and model architecture.

## Directory Structure

- `train.py`: Optimized training pipeline
- `normal_train.py`: Standard training pipeline
- `utils.py`: Utility functions (K-Fold, scheduler, etc.)
- `TorchUtils/`: Deep learning training tricks and tools
- `data/`: Dataset directory
- `训练优化函数使用说明.md`: Chinese documentation for training optimization functions

## Contact

- Author GitHub: [@Jackksonns](https://github.com/Jackksonns)

## Limitations & Future Work

- **Current limitations:**
  - Primarily designed for image classification; other tasks require adaptation
  - Manual adjustment needed for custom models and datasets
- **Future directions:**
  - Extend support to segmentation, detection, and other tasks
  - Provide more automated data/model adaptation
  - Integrate additional training tricks and visualization tools

---

For questions or suggestions, please contact the author via GitHub or submit an issue.

## Citation & Dependencies

Some utility functions in OptiTrainer are adapted from [TorchUtils](https://github.com/seefun/TorchUtils).

**Special thanks to the author of TorchUtils for their excellent open-source work.**

**Before using OptiTrainer, please clone and install TorchUtils locally:**

```bash
# Clone TorchUtils
https://github.com/seefun/TorchUtils.git
cd TorchUtils
pip install -r requirements.txt
pip install .
```

If you use OptiTrainer or TorchUtils in your research, please consider citing the original repositories.
