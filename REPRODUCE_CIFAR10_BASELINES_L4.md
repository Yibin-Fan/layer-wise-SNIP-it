# 复现 CIFAR10 Baselines（L4）

需要跑的实验只有这三组：

- `SNAPit` with `VGG16 + CIFAR10`
- `SNIPit` with `ResNet18 + CIFAR10` before training
- `SNIPitDuring` with `ResNet18 + CIFAR10` during training

为了计算 `accuracy drop`，需要对应跑两个未剪枝 baseline：

- `VGG16 + EmptyCrit`
- `ResNet18 + EmptyCrit`

## 1. 环境

```bash
cd /path/to/layer-wise-SNIP-it
conda env create -f environment.yml
conda activate snapit-l4
mkdir -p gitignored/data gitignored/results
```

检查 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

## 2. 正式训练

### VGG16 baseline

```bash
python main.py \
  --model VGG16 \
  --data_set CIFAR10 \
  --prune_criterion EmptyCrit \
  --pruning_limit 0.0 \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _vgg16_empty
```

### VGG16 SNAPit

```bash
python main.py \
  --model VGG16 \
  --data_set CIFAR10 \
  --prune_criterion SNAPit \
  --pruning_limit 0.93 \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _vgg16_snapit93
```

### ResNet18 baseline

```bash
python main.py \
  --model ResNet18 \
  --data_set CIFAR10 \
  --prune_criterion EmptyCrit \
  --pruning_limit 0.0 \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _resnet18_empty
```

### ResNet18 SNIPit

```bash
python main.py \
  --model ResNet18 \
  --data_set CIFAR10 \
  --prune_criterion SNIPit \
  --pruning_limit 0.98 \
  --outer_layer_pruning \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _resnet18_snipit98
```

### ResNet18 SNIPitDuring

```bash
python main.py \
  --model ResNet18 \
  --data_set CIFAR10 \
  --prune_criterion SNIPitDuring \
  --pruning_limit 0.98 \
  --outer_layer_pruning \
  --epochs 80 \
  --prune_delay 4 \
  --prune_freq 4 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _resnet18_snipitduring98
```

如果 `batch_size=256` 显存不足，就改成 `128`。

## 3. 找到结果目录

```bash
ls -td gitignored/results/*_vgg16_empty | head -n 1
ls -td gitignored/results/*_vgg16_snapit93 | head -n 1
ls -td gitignored/results/*_resnet18_empty | head -n 1
ls -td gitignored/results/*_resnet18_snipit98 | head -n 1
ls -td gitignored/results/*_resnet18_snipitduring98 | head -n 1
```

## 4. 结果表格

实验跑完后，把提取脚本输出的结果填到下面表格中：

| Experiment | Baseline Acc (%) | Method Acc (%) | Accuracy Drop (pp) | Weight Sparsity (%) | Node Sparsity (%) | FLOPs Reduction (x) |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 + SNAPit |  |  |  |  |  |  |
| ResNet18 + SNIPit |  |  |  |  |  |  |
| ResNet18 + SNIPitDuring |  |  |  |  |  |  |

## 5. 对照目标

- `VGG16 + SNAPit`：`accuracy-drop` 约 `-1%`，`weight sparsity` 约 `99%`，`node sparsity` 约 `93%`
- `ResNet18 + SNIPit`：`accuracy-drop` 约 `-4%`，`weight sparsity` 约 `98%`
- `ResNet18 + SNIPitDuring`：`accuracy-drop` 约 `0%`，`weight sparsity` 约 `98%`
