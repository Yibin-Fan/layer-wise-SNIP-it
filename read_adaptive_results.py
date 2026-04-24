import pickle
from pathlib import Path

# ==================================================
# Root folder containing all experiment result folders
# ==================================================
RESULTS_ROOT = Path("gitignored/results")

# ==================================================
# Three new adaptive layer-wise experiments
# ==================================================
RUNS = {
    "vgg16_adaptive_layerwise_snapit93":
        "2026-04-24_00.19.23_vgg16_adaptive_layerwise_snapit93",

    "resnet18_adaptive_layerwise_snipit98":
        "2026-04-24_00.26.38_resnet18_adaptive_layerwise_snipit98",

    "resnet18_adaptive_layerwise_snipitduring98":
        "2026-04-24_00.35.55_resnet18_adaptive_layerwise_snipitduring98",
}


# ==================================================
# Helpers
# ==================================================
def safe_last(data, key):
    if key not in data or len(data[key]) == 0:
        return None
    return float(data[key][-1])


def safe_best(data, key):
    if key not in data or len(data[key]) == 0:
        return None
    return float(max(data[key]))


def pct(x):
    return "N/A" if x is None else f"{x * 100:.2f}"


def flt(x, digits=2):
    return "N/A" if x is None else f"{x:.{digits}f}"


def integer(x):
    return "N/A" if x is None else f"{x:.0f}"


def sci(x):
    return "N/A" if x is None else f"{x:.2e}"


# ==================================================
# Load one run
# ==================================================
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
        "cum_flops": cum_flops,

        "gpu_time": safe_last(data, "time/gpu_time"),
        "batch_time": safe_last(data, "time/batch_time"),

        "ram_footprint": safe_last(data, "cuda/ram_footprint"),
        "disk_log": safe_last(data, "sparse/log_disk_size"),
    }


# ==================================================
# Pretty print
# ==================================================
def print_result(title, r):
    print("\n==================================================")
    print(title)
    print("==================================================")

    print(f"Run name:             {r['run_name']}")
    print(f"Run dir:              {r['run_dir']}")

    print(f"Best Accuracy (%):    {pct(r['acc_best'])}")
    print(f"Final Accuracy (%):   {pct(r['acc_final'])}")
    print(f"Final Loss:           {flt(r['loss_final'], 4)}")

    print(f"Weight Sparsity (%):  {pct(r['weight_sparsity'])}")
    print(f"Node Sparsity (%):    {pct(r['node_sparsity'])}")

    print(f"FLOPs / sample:       {integer(r['flops_per_sample'])}")
    print(f"Cumulative FLOPs:     {sci(r['cum_flops'])}")

    print(f"GPU Time (s):         {flt(r['gpu_time'])}")
    print(f"Batch Time (s):       {flt(r['batch_time'], 4)}")

    ram_mb = None
    if r["ram_footprint"] is not None:
        ram_mb = r["ram_footprint"] / 1e6

    print(f"RAM Footprint (MB):   {flt(ram_mb)}")
    print(f"Disk log10 size:      {flt(r['disk_log'], 4)}")


# ==================================================
# Main
# ==================================================
def main():
    print("\n########## ADAPTIVE LAYER-WISE RESULTS ##########\n")

    for key, folder in RUNS.items():
        result = load_run(folder)
        print_result(key, result)


if __name__ == "__main__":
    main()