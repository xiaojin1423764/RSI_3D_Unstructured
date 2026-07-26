#!/usr/bin/env python3
"""Collect reproducible SI/RSI CUDA benchmark metrics from case logs and fields."""

import argparse
import csv
import math
import re
from pathlib import Path

from scipy.spatial import cKDTree

CASE_RE = re.compile(
    r"(?P<grid>\d+k)_s(?P<sn>\d+)_r(?P<samples>\d+)"
    r"(?:_b(?P<batch>\d+))?_rep(?P<rep>\d+)"
)
FIELD_RE = re.compile(r"([A-Za-z0-9_]+)=([^,\s]+)")
GIB = 1024 ** 3
TAIL_EXTRA = 10  # Figure 5 uses N through N+10 for the shared RSI-tail chains.


def fields(line):
    return {key: value for key, value in FIELD_RE.findall(line)}


def numeric(values, key, default=float("nan")):
    try:
        return float(values[key])
    except (KeyError, ValueError):
        return default


def read_field(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [(float(row["x"]), float(row["y"]), float(row["z"])) for row in rows],
        [float(row["phi0"]) for row in rows],
    )


def read_phi(path):
    return read_field(path)[1]


def relative_l2(values, reference):
    if len(values) != len(reference):
        return float("nan"), float("nan")
    numerator = math.fsum((value - ref) ** 2 for value, ref in zip(values, reference))
    denominator = math.fsum(ref ** 2 for ref in reference)
    return math.sqrt(numerator / denominator), max(
        (abs(value - ref) for value, ref in zip(values, reference)), default=0.0
    )


def parse_case(case_dir):
    match = CASE_RE.fullmatch(case_dir.name)
    if not match:
        return None
    lines = (case_dir / "run.log").read_text(errors="replace").splitlines()
    grouped = {}
    for line in lines:
        if line.startswith("CUDA streaming timing: si_total="):
            grouped["si"] = fields(line)
        elif line.startswith("CUDA streaming timing: si_plan_breakdown_"):
            grouped["si_plan"] = fields(line)
        elif line.startswith("CUDA streaming timing: rsi_total="):
            grouped["rsi"] = fields(line)
        elif line.startswith("CUDA streaming timing: rsi_plan_breakdown_"):
            grouped["rsi_plan"] = fields(line)
        elif line.startswith("CUDA streaming parallelism:"):
            grouped["parallel"] = fields(line)
        elif line.startswith("CUDA streaming plan_transfer:"):
            grouped["transfer"] = fields(line)
        elif line.startswith("CUDA streaming Figure 5 complete:"):
            grouped["complete"] = fields(line)
    if "si" not in grouped or "rsi" not in grouped:
        return None

    item = match.groupdict()
    item["batch"] = item["batch"] or "128"
    sn = int(item["sn"])
    item.update({"directions": sn * (sn + 2), "case": case_dir.name})
    item.update({
        "si_total_s": numeric(grouped["si"], "si_total"),
        "si_plan_s": numeric(grouped["si"], "si_plan"),
        "si_sweep_s": numeric(grouped["si"], "si_sweep"),
        "rsi_total_s": numeric(grouped["rsi"], "rsi_total"),
        "rsi_plan_s": numeric(grouped["rsi"], "rsi_plan"),
        "rsi_sweep_s": numeric(grouped["rsi"], "rsi_sweep"),
    })
    item.update({key: numeric(grouped.get("parallel", {}), key) for key in (
        "si_sweep_launches", "si_sweep_blocks", "si_sweep_logical_threads",
        "si_sweep_threads_per_block", "rsi_sweep_launches", "rsi_sweep_blocks",
        "rsi_sweep_logical_threads", "rsi_sweep_threads_per_block",
    )})
    item.update({key: numeric(grouped.get("transfer", {}), key) for key in (
        "si_cache_read_bytes", "si_h2d_bytes", "rsi_cache_read_bytes",
        "rsi_h2d_bytes", "rsi_direction_h2d_bytes",
    )})
    si_plan = grouped.get("si_plan", {})
    rsi_plan = grouped.get("rsi_plan", {})
    item["si_cache_s"] = numeric(si_plan, "si_plan_breakdown_cache")
    item["si_h2d_s"] = numeric(si_plan, "si_plan_breakdown_upload")
    item["rsi_cache_s"] = numeric(rsi_plan, "rsi_plan_breakdown_cache")
    item["rsi_cache_wall_s"] = numeric(rsi_plan, "rsi_plan_fixed_cache_wall")
    item["rsi_h2d_s"] = numeric(rsi_plan, "rsi_plan_breakdown_upload")
    item["rsi_cache_hits"] = numeric(rsi_plan, "rsi_plan_fixed_cache_hits")
    item["rsi_cache_misses"] = numeric(rsi_plan, "rsi_plan_fixed_cache_misses")
    item["si_cache_gib_s"] = item["si_cache_read_bytes"] / GIB / item["si_cache_s"] if item["si_cache_s"] else float("nan")
    rsi_cache_denominator = item["rsi_cache_wall_s"] if item["rsi_cache_wall_s"] > 0 else item["rsi_cache_s"]
    item["rsi_cache_gib_s"] = item["rsi_cache_read_bytes"] / GIB / rsi_cache_denominator if rsi_cache_denominator else float("nan")
    item["si_h2d_gib_s"] = item["si_h2d_bytes"] / GIB / item["si_h2d_s"] if item["si_h2d_s"] else float("nan")
    item["rsi_h2d_gib_s"] = item["rsi_h2d_bytes"] / GIB / item["rsi_h2d_s"] if item["rsi_h2d_s"] else float("nan")
    complete = grouped.get("complete", {})
    item["cells"] = numeric(complete, "cells")
    item["si_iterations"] = numeric(complete, "SI_iterations")
    item["rsi_iterations"] = item["si_iterations"] + TAIL_EXTRA
    item["rsi_batch_capacity_actual"] = numeric(complete, "rsi_batch_capacity")
    item["rsi_batch_cap_requested"] = numeric(complete, "rsi_batch_cap_requested")
    item["rsi_samples_per_s"] = int(item["samples"]) / item["rsi_total_s"]
    item["rsi_sweep_samples_per_s"] = int(item["samples"]) / item["rsi_sweep_s"]
    # Same GPU-sweep work metric for both methods. SI evaluates every direction
    # at each SI iteration; RSI evaluates one sampled direction per chain step.
    item["si_mcell_sweeps_per_s"] = (
        item["cells"] * item["directions"] * item["si_iterations"]
        / item["si_sweep_s"] / 1.0e6
    )
    item["rsi_mcell_sweeps_per_s"] = (
        item["cells"] * int(item["samples"]) * item["rsi_iterations"]
        / item["rsi_sweep_s"] / 1.0e6
    )
    # Both methods use the same unit of work: one cell advanced for one
    # direction/sample at one transport iteration.  The ratio is therefore
    # the RSI throughput improvement over SI under one common definition.
    item["rsi_parallel_efficiency_mcell_sweeps_s"] = item["rsi_mcell_sweeps_per_s"]
    item["si_parallel_efficiency_mcell_sweeps_s"] = item["si_mcell_sweeps_per_s"]
    item["rsi_vs_si_parallel_efficiency_gain"] = (
        item["rsi_mcell_sweeps_per_s"] / item["si_mcell_sweeps_per_s"]
    )
    item["rsi_vs_si_speedup"] = item["si_total_s"] / item["rsi_total_s"]
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--reference-case",
        action="append",
        default=[],
        metavar="CASE",
        help=(
            "Case directory under ROOT/cases containing an SI reference. Repeat this "
            "option to provide one reference per grid. A single reference retains the "
            "cross-grid nearest-cell-center comparison."
        ),
    )
    args = parser.parse_args()
    cases = [parse_case(path) for path in sorted((args.root / "cases").glob("*"))]
    cases = [case for case in cases if case]

    if args.reference_case:
        references = []
        for reference_name in args.reference_case:
            reference_path = (
                args.root / "cases" / reference_name / "Cir" / "figure5_SI_fine.csv"
            )
            if not reference_path.exists():
                raise SystemExit(f"Reference SI field not found: {reference_path}")
            reference_case = parse_case(args.root / "cases" / reference_name)
            if not reference_case:
                raise SystemExit(
                    f"Reference case has no complete timing data: {reference_name}"
                )
            reference_coords, reference_phi = read_field(reference_path)
            references.append({
                "case": reference_case,
                "coords": reference_coords,
                "phi": reference_phi,
                "tree": cKDTree(reference_coords),
            })

        references_by_grid = {}
        if len(references) > 1:
            for reference in references:
                grid = reference["case"]["grid"]
                if grid in references_by_grid:
                    raise SystemExit(f"Multiple reference cases provided for grid {grid}")
                references_by_grid[grid] = reference

        mapped_references = {}
        for case in cases:
            reference = (
                references[0]
                if len(references) == 1
                else references_by_grid.get(case["grid"])
            )
            if reference is None:
                raise SystemExit(f"No reference case provided for grid {case['grid']}")
            base = args.root / "cases" / case["case"] / "Cir"
            si_path = base / "figure5_SI_fine.csv"
            rsi_tail_path = base / "figure5_RSI_tail.csv"
            if not si_path.exists() or not rsi_tail_path.exists():
                case["si_rel_l2_vs_reference"] = float("nan")
                case["rsi_tail_rel_l2_vs_reference"] = float("nan")
                case["rsi_tail_max_abs_vs_reference"] = float("nan")
                continue
            coordinates, si_phi = read_field(si_path)
            reference_case = reference["case"]
            if case["grid"] == reference_case["grid"]:
                if len(coordinates) != len(reference["coords"]):
                    raise SystemExit(
                        f"Same-grid reference cell count mismatch for {case['case']}"
                    )
                ref = reference["phi"]
                mean_distance = 0.0
                max_distance = 0.0
                reference_map = "same_mesh_cell_order"
            else:
                map_key = (case["grid"], reference_case["case"])
                if map_key not in mapped_references:
                    distances, indices = reference["tree"].query(coordinates, workers=-1)
                    mapped_references[map_key] = (
                        [reference["phi"][index] for index in indices],
                        math.fsum(distances) / len(distances),
                        max(distances),
                    )
                ref, mean_distance, max_distance = mapped_references[map_key]
                reference_map = "nearest_cell_center"
            case["si_rel_l2_vs_reference"], _ = relative_l2(si_phi, ref)
            case["rsi_tail_rel_l2_vs_reference"], case["rsi_tail_max_abs_vs_reference"] = relative_l2(
                read_phi(rsi_tail_path), ref
            )
            case["reference_case"] = reference_case["case"]
            case["reference_directions"] = reference_case["directions"]
            case["reference_cells"] = reference_case["cells"]
            case["reference_map"] = reference_map
            case["reference_mean_distance"] = mean_distance
            case["reference_max_distance"] = max_distance
    else:
        references = {}
        for case in cases:
            if case["sn"] == "316":
                path = args.root / "cases" / case["case"] / "Cir" / "figure5_SI_fine.csv"
                if path.exists():
                    references[(case["grid"], case["rep"])] = read_phi(path)
        for case in cases:
            ref = references.get((case["grid"], case["rep"]))
            base = args.root / "cases" / case["case"] / "Cir"
            if ref and (base / "figure5_SI_fine.csv").exists():
                case["si_rel_l2_vs_s316"], _ = relative_l2(read_phi(base / "figure5_SI_fine.csv"), ref)
                case["rsi_tail_rel_l2_vs_s316"], case["rsi_tail_max_abs_vs_s316"] = relative_l2(
                    read_phi(base / "figure5_RSI_tail.csv"), ref
                )
            else:
                case["si_rel_l2_vs_s316"] = float("nan")
                case["rsi_tail_rel_l2_vs_s316"] = float("nan")
                case["rsi_tail_max_abs_vs_s316"] = float("nan")

    if not cases:
        raise SystemExit("No completed streaming benchmark cases found")
    columns = sorted({key for case in cases for key in case})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)


if __name__ == "__main__":
    main()
