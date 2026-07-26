#!/usr/bin/env python3
"""Extract the cache-hierarchy and occupancy rows needed by the benchmark report."""

import argparse
import csv
from pathlib import Path


METRICS = {
    "Block Size": "block_size",
    "Grid Size": "grid_size",
    "Memory Throughput": "memory_throughput_pct",
    "L1/TEX Cache Throughput": "l1_tex_throughput_pct",
    "L2 Cache Throughput": "l2_throughput_pct",
    "DRAM Throughput": "dram_throughput_pct",
    "L1/TEX Hit Rate": "l1_tex_hit_rate_pct",
    "L2 Hit Rate": "l2_hit_rate_pct",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "Waves Per SM": "waves_per_sm",
}


def load_rows(path):
    lines = path.read_text(errors="replace").splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith('"ID",'))
    return list(csv.DictReader(lines[header:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--kernel-substring",
        default="sweepLevelFusedKernel",
        help="Only extract metrics for kernels whose name contains this substring.",
    )
    args = parser.parse_args()

    result = {"profile": args.input.name}
    for row in load_rows(args.input):
        if args.kernel_substring not in row["Kernel Name"]:
            continue
        metric = row["Metric Name"]
        if metric == "Memory Throughput" and row["Metric Unit"] == "byte/s":
            result["memory_throughput_bytes_s"] = row["Metric Value"]
        elif metric in METRICS:
            result[METRICS[metric]] = row["Metric Value"]
    if len(result) == 1:
        raise SystemExit("No sweepLevelFusedKernel metrics found")
    columns = ["profile"] + sorted(key for key in result if key != "profile")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
