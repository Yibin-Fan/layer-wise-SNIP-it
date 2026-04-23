# 复现 CIFAR10 Baselines（L4）

需要跑的对比实验有三组：

- `VGG16 + SNAPit` vs `VGG16 + LayerWiseSNAPit`
- `ResNet18 + SNIPit` vs `ResNet18 + LayerWiseSNIPit`
- `ResNet18 + SNIPitDuring` vs `ResNet18 + LayerWiseSNIPitDuring`

为了计算 `accuracy drop`，额外跑两个未剪枝 reference：

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

### 2.1 未剪枝 reference

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

### 2.2 实验一：VGG16 structured pruning

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

```bash
python main.py \
  --model VGG16 \
  --data_set CIFAR10 \
  --prune_criterion LayerWiseSNAPit \
  --pruning_limit 0.93 \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _vgg16_layerwise_snapit93
```

### 2.3 实验二：ResNet18 unstructured pruning before training

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

```bash
python main.py \
  --model ResNet18 \
  --data_set CIFAR10 \
  --prune_criterion LayerWiseSNIPit \
  --pruning_limit 0.98 \
  --outer_layer_pruning \
  --epochs 80 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _resnet18_layerwise_snipit98
```

### 2.4 实验三：ResNet18 unstructured pruning during training

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

```bash
python main.py \
  --model ResNet18 \
  --data_set CIFAR10 \
  --prune_criterion LayerWiseSNIPitDuring \
  --pruning_limit 0.98 \
  --outer_layer_pruning \
  --epochs 80 \
  --prune_delay 4 \
  --prune_freq 4 \
  --device cuda \
  --batch_size 256 \
  --seed 333 \
  --run_name _resnet18_layerwise_snipitduring98
```

如果 `batch_size=256` 显存不足，就改成 `128`。

## 3. 找到结果目录

```bash
ls -td gitignored/results/*_vgg16_empty | head -n 1
ls -td gitignored/results/*_vgg16_snapit93 | head -n 1
ls -td gitignored/results/*_vgg16_layerwise_snapit93 | head -n 1
ls -td gitignored/results/*_resnet18_empty | head -n 1
ls -td gitignored/results/*_resnet18_snipit98 | head -n 1
ls -td gitignored/results/*_resnet18_layerwise_snipit98 | head -n 1
ls -td gitignored/results/*_resnet18_snipitduring98 | head -n 1
ls -td gitignored/results/*_resnet18_layerwise_snipitduring98 | head -n 1
```

## 4. 结果表格

实验跑完后，把结果填到下面表格中：

| Group | Method | Reference Acc (%) | Method Acc (%) | Accuracy Drop (pp) | Weight Sparsity (%) | Node Sparsity (%) | FLOPs Reduction (x) |
|---|---|---:|---:|---:|---:|---:|---:|
| VGG16 structured before training | SNAPit | 88.20 | 84.93 | 3.27 | 99.25 | 93.01 | 9.87 |
| VGG16 structured before training | LayerWiseSNAPit | 88.20 |  |  |  |  |  |
| ResNet18 unstructured before training | SNIPit | 79.62 | 76.27 | 3.35 | 97.96 | 0.00 | 1.00 |
| ResNet18 unstructured before training | LayerWiseSNIPit | 79.62 |  |  |  |  |  |
| ResNet18 unstructured during training | SNIPitDuring | 79.62 | 79.78 | -0.16 | 97.96 | 0.00 | 1.00 |
| ResNet18 unstructured during training | LayerWiseSNIPitDuring | 79.62 |  |  |  |  |  |

## 5. 对照目标

- `VGG16 + SNAPit`：`accuracy-drop` 约 `-1%`，`weight sparsity` 约 `99%`，`node sparsity` 约 `93%`
- `ResNet18 + SNIPit`：`accuracy-drop` 约 `-4%`，`weight sparsity` 约 `98%`
- `ResNet18 + SNIPitDuring`：`accuracy-drop` 约 `0%`，`weight sparsity` 约 `98%`
