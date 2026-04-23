import pickle
from pathlib import Path


RESULTS_ROOT = Path("gitignored/results")

RUNS = {
    "vgg16_empty": "2026-04-23_10.17.21_vgg16_empty",
    "resnet18_empty": "2026-04-23_10.28.09_resnet18_empty",

    "vgg16_snapit93": "2026-04-23_10.37.39_vgg16_snapit93",
    "vgg16_layerwise_snapit93": "2026-04-23_10.46.45_vgg16_layerwise_snapit93",

    "resnet18_snipit98": "2026-04-23_10.55.03_resnet18_snipit98",
    "resnet18_layerwise_snipit98": "2026-04-23_11.05.21_resnet18_layerwise_snipit98",

    "resnet18_snipitduring98": "2026-04-23_11.16.29_resnet18_snipitduring98",
    "resnet18_layerwise_snipitduring98": "2026-04-23_11.27.07_resnet18_layerwise_snipitduring98",
}


EXPERIMENTS = [
    {
        "group": "VGG16 structured\nbefore training",
        "method": "SNAPit",
        "run_key": "vgg16_snapit93",
        "baseline_key": "vgg16_empty",
    },
    {
        "group": "VGG16 structured\nbefore training",
        "method": "Layer-wise SNAPit",
        "run_key": "vgg16_layerwise_snapit93",
        "baseline_key": "vgg16_empty",
    },
    {
        "group": "ResNet18 unstructured\nbefore training",
        "method": "SNIPit",
        "run_key": "resnet18_snipit98",
        "baseline_key": "resnet18_empty",
    },
    {
        "group": "ResNet18 unstructured\nbefore training",
        "method": "Layer-wise SNIPit",
        "run_key": "resnet18_layerwise_snipit98",
        "baseline_key": "resnet18_empty",
    },
    {
        "group": "ResNet18 unstructured\nduring training",
        "method": "SNIPitDuring",
        "run_key": "resnet18_snipitduring98",
        "baseline_key": "resnet18_empty",
    },
    {
        "group": "ResNet18 unstructured\nduring training",
        "method": "Layer-wise SNIPitDuring",
        "run_key": "resnet18_layerwise_snipitduring98",
        "baseline_key": "resnet18_empty",
    },
]


def safe_last(data, key):
    if key not in data or len(data[key]) == 0:
        return None
    return float(data[key][-1])


def safe_best(data, key):
    if key not in data or len(data[key]) == 0:
        return None
    return float(max(data[key]))


def load_run(run_name: str):
    run_dir = RESULTS_ROOT / run_name
    metrics_file = run_dir / "models" / "Metrics_finished.pickle"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, "rb") as f:
        state = pickle.load(f)

    data = state["_data"]

    flops_log_cum = safe_last(data, "time/flops_log_cum")
    cum_flops = 10 ** flops_log_cum if flops_log_cum is not None else None

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "acc_best": safe_best(data, "acc/test"),
        "acc_final": safe_last(data, "acc/test"),
        "loss_final": safe_last(data, "loss/test"),
        "weight_sparsity": safe_last(data, "sparse/weight"),
        "node_sparsity": safe_last(data, "sparse/node"),
        "flops_per_sample": safe_last(data, "time/flops_per_sample"),
        "flops_log_cum": flops_log_cum,
        "cum_flops": cum_flops,
        "gpu_time": safe_last(data, "time/gpu_time"),
        "batch_time": safe_last(data, "time/batch_time"),
        "ram_footprint": safe_last(data, "cuda/ram_footprint"),
        "disk_log": safe_last(data, "sparse/log_disk_size"),
    }


def pct(x):
    return "N/A" if x is None else f"{x * 100:.2f}"


def flt(x, digits=2):
    return "N/A" if x is None else f"{x:.{digits}f}"


def integer(x):
    return "N/A" if x is None else f"{x:.0f}"


def sci(x):
    return "N/A" if x is None else f"{x:.2e}"


def compare_with_baseline(baseline, method):
    acc_drop_pp = None
    if baseline["acc_best"] is not None and method["acc_best"] is not None:
        acc_drop_pp = (baseline["acc_best"] - method["acc_best"]) * 100.0

    flops_reduction = None
    if baseline["cum_flops"] is not None and method["cum_flops"] is not None and method["cum_flops"] != 0:
        flops_reduction = baseline["cum_flops"] / method["cum_flops"]

    return {
        "reference_acc": baseline["acc_best"],
        "method_acc": method["acc_best"],
        "acc_drop_pp": acc_drop_pp,
        "weight_sparsity": method["weight_sparsity"],
        "node_sparsity": method["node_sparsity"],
        "flops_reduction": flops_reduction,
    }


def print_single_run(title, result):
    print(f"\n===== {title} =====")
    print(f"Run name:            {result['run_name']}")
    print(f"Run dir:             {result['run_dir']}")
    print(f"Best acc (%):        {pct(result['acc_best'])}")
    print(f"Final acc (%):       {pct(result['acc_final'])}")
    print(f"Final loss:          {flt(result['loss_final'], 4)}")
    print(f"Weight sparsity (%): {pct(result['weight_sparsity'])}")
    print(f"Node sparsity (%):   {pct(result['node_sparsity'])}")
    print(f"FLOPs/sample:        {integer(result['flops_per_sample'])}")
    print(f"Cumulative FLOPs:    {sci(result['cum_flops'])}")
    print(f"GPU time (s):        {flt(result['gpu_time'], 2)}")
    print(f"Batch time (s):      {flt(result['batch_time'], 4)}")
    print(
        f"RAM footprint (MB):  {flt(result['ram_footprint'] / 1e6 if result['ram_footprint'] is not None else None, 2)}"
    )
    print(f"Disk log10 size:     {flt(result['disk_log'], 4)}")


def print_grouped_comparison_table(rows):
    print("\n================ FINAL TABLE ================\n")

    headers = [
        "Group",
        "Method",
        "Reference\nAcc (%)",
        "Method\nAcc (%)",
        "Accuracy\nDrop (pp)",
        "Weight\nSparsity (%)",
        "Node\nSparsity (%)",
        "FLOPs\nReduction (x)",
    ]

    widths = [28, 26, 14, 14, 16, 18, 16, 18]

    def print_row(values):
        print("".join(str(v).ljust(w) for v, w in zip(values, widths)))

    print_row(headers)
    print("-" * sum(widths))

    for row in rows:
        print_row([
            row["group"],
            row["method"],
            pct(row["result"]["reference_acc"]),
            pct(row["result"]["method_acc"]),
            flt(row["result"]["acc_drop_pp"], 2),
            pct(row["result"]["weight_sparsity"]),
            pct(row["result"]["node_sparsity"]),
            flt(row["result"]["flops_reduction"], 2),
        ])

    print("\n=============================================\n")


def main():
    # 1) 先读取全部 runs
    loaded = {}
    for key, run_name in RUNS.items():
        loaded[key] = load_run(run_name)

    # 2) 单独输出每组结果
    print("\n########## INDIVIDUAL RUN RESULTS ##########\n")

    print_single_run("BASELINE: VGG16 Empty", loaded["vgg16_empty"])
    print_single_run("BASELINE: ResNet18 Empty", loaded["resnet18_empty"])

    print_single_run("METHOD: VGG16 SNAPit", loaded["vgg16_snapit93"])
    print_single_run("METHOD: VGG16 Layer-wise SNAPit", loaded["vgg16_layerwise_snapit93"])

    print_single_run("METHOD: ResNet18 SNIPit", loaded["resnet18_snipit98"])
    print_single_run("METHOD: ResNet18 Layer-wise SNIPit", loaded["resnet18_layerwise_snipit98"])

    print_single_run("METHOD: ResNet18 SNIPitDuring", loaded["resnet18_snipitduring98"])
    print_single_run("METHOD: ResNet18 Layer-wise SNIPitDuring", loaded["resnet18_layerwise_snipitduring98"])

    # 3) 再做对比表
    rows = []
    for exp in EXPERIMENTS:
        baseline = loaded[exp["baseline_key"]]
        method = loaded[exp["run_key"]]
        cmp_result = compare_with_baseline(baseline, method)

        rows.append({
            "group": exp["group"],
            "method": exp["method"],
            "result": cmp_result,
        })

    print_grouped_comparison_table(rows)


if __name__ == "__main__":
    main()