# 复现 `VGG16 + CIFAR10` baseline（RTX 4060）

目标是跑通论文仓库在 `VGG16 + CIFAR10` 上的 **structured baseline: `SNAPit`**，并和未剪枝的 `EmptyCrit` 做对比。

## 1. 准备环境

```bash
cd /path/to/layer-wise-SNIP-it
conda env create -f environment.yml
conda activate snapit-4060
```

检查 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 2. 准备目录

```bash
cd /path/to/layer-wise-SNIP-it
mkdir -p gitignored/data gitignored/results
```

## 3. 先做 smoke test

未剪枝：

```bash
python main.py \
  --model VGG16 \
  --data_set CIFAR10 \
  --prune_criterion EmptyCrit \
  --pruning_limit 0.0 \
  --epochs 1 \
  --device cuda \
  --batch_size 128 \
  --seed 333 \
  --run_name _smoke_vgg16_cifar10_empty_seed333
```

`SNAPit`：

```bash
python main.py \
  --model VGG16 \
  --data_set CIFAR10 \
  --prune_criterion SNAPit \
  --pruning_limit 0.93 \
  --epochs 1 \
  --device cuda \
  --batch_size 128 \
  --seed 333 \
  --run_name _smoke_vgg16_cifar10_snapit93_seed333
```

## 4. 正式训练

未剪枝 baseline：

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
  --run_name _vgg16_cifar10_empty_seed333
```

`SNAPit` baseline：

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
  --run_name _vgg16_cifar10_snapit93_seed333
```

如果 `batch_size=256` 显存不足，就改成 `128`。

## 5. 定位结果目录

```bash
ls -td gitignored/results/*_vgg16_cifar10_empty_seed333 | head -n 1
ls -td gitignored/results/*_vgg16_cifar10_snapit93_seed333 | head -n 1
```

## 6. 提取最终指标

把下面脚本中的两个目录名替换成你自己的运行目录：

```bash
PYTHONPATH=. python - <<'PY'
import pickle
from pathlib import Path

def load_run(run_dir: str):
    path = Path(run_dir)
    with open(path / "models" / "Metrics_finished.pickle", "rb") as f:
        state = pickle.load(f)
    data = state["_data"]
    result = {
        "acc_test": float(data["acc/test"][-1]),
        "weight_sparsity": float(data["sparse/weight"][-1]),
        "node_sparsity": float(data["sparse/node"][-1]),
        "flops_per_sample": float(data["time/flops_per_sample"][-1]),
        "flops_log_cum": float(data["time/flops_log_cum"][-1]),
    }
    result["cum_flops"] = 10 ** result["flops_log_cum"]
    return result

baseline_dir = "gitignored/results/<baseline_run_folder>"
snapit_dir = "gitignored/results/<snapit_run_folder>"

baseline = load_run(baseline_dir)
snapit = load_run(snapit_dir)

print(f"baseline acc: {baseline['acc_test'] * 100:.2f}%")
print(f"snapit acc: {snapit['acc_test'] * 100:.2f}%")
print(f"accuracy drop: {(baseline['acc_test'] - snapit['acc_test']) * 100:.2f} percentage points")
print(f"snapit weight sparsity: {snapit['weight_sparsity'] * 100:.2f}%")
print(f"snapit node sparsity: {snapit['node_sparsity'] * 100:.2f}%")
print(f"baseline FLOPs/sample: {baseline['flops_per_sample']:.0f}")
print(f"snapit FLOPs/sample: {snapit['flops_per_sample']:.0f}")
print(f"cumulative training FLOPs reduction: {baseline['cum_flops'] / snapit['cum_flops']:.2f}x")
PY
```

## 7. 结果判断

跑通后重点看：

- `SNAPit` 的 `node sparsity` 是否接近 `93%`
- `weight sparsity` 是否接近 `99%`
- 相对 `EmptyCrit` 的 `accuracy drop` 是否接近 `0` 到 `1` 个百分点
- `cumulative training FLOPs reduction` 是否接近 `8x`

## 8. 两个必要说明

- 当前仓库已经修过路径问题，fork 改名后可以直接从仓库根目录运行。
- 4060 不能实际使用原 README 的 `torch==1.4.0`，所以这里是兼容环境复现，不是原始二进制环境复刻。
