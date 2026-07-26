#!/usr/bin/env python3
"""Render the 270-case SI/RSI factor matrix as six LaTeX longtables."""

import argparse
import csv
import math
from pathlib import Path


GRIDS = ("10k", "30k", "100k")
SN = {10, 20, 30, 70, 100, 224}
SAMPLES = {8, 64, 512, 1024, 4096}
BATCHES = {64, 128, 256}


def number(value, digits=3):
    value = float(value)
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}"


def integer(value):
    value = float(value)
    return "--" if not math.isfinite(value) else str(int(value))


def rows_for_grid(rows, grid):
    selected = []
    for row in rows:
        if row["grid"] != grid:
            continue
        if int(row["sn"]) not in SN or int(row["samples"]) not in SAMPLES:
            continue
        if int(float(row["rsi_batch_cap_requested"])) not in BATCHES:
            continue
        selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            int(row["directions"]),
            int(row["samples"]),
            int(float(row["rsi_batch_cap_requested"])),
        ),
    )


def timing_row(row):
    error = float(row["rsi_tail_rel_l2_vs_reference"]) * 100.0
    return " & ".join((
        integer(row["directions"]), integer(row["samples"]),
        integer(row["rsi_batch_cap_requested"]), integer(row["rsi_batch_capacity_actual"]),
        number(row["si_total_s"]), number(row["rsi_total_s"]),
        f"{number(row['rsi_vs_si_speedup'], 2)}x",
        number(error),
    )) + r"\\"


def efficiency_row(row):
    return " & ".join((
        integer(row["directions"]), integer(row["samples"]),
        integer(row["rsi_batch_cap_requested"]), integer(row["rsi_batch_capacity_actual"]),
        number(row["si_sweep_s"]), number(row["si_parallel_efficiency_mcell_sweeps_s"], 1),
        number(row["rsi_sweep_s"]), number(row["rsi_parallel_efficiency_mcell_sweeps_s"], 1),
        f"{number(row['rsi_vs_si_parallel_efficiency_gain'], 2)}x",
        number(row["rsi_plan_s"]),
    )) + r"\\"


def longtable(title, label, column_spec, header, rows):
    body = "\n".join(rows)
    return f"""{{\\tiny\\setlength{{\\tabcolsep}}{{2.3pt}}
\\begin{{longtable}}{{{column_spec}}}
\\caption{{{title}}}\\label{{{label}}}\\\\
\\toprule
{header}\\\\
\\midrule
\\endfirsthead
\\multicolumn{{{header.count('&') + 1}}}{{l}}{{\\small 续表：{title}}}\\\\
\\toprule
{header}\\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{body}
\\end{{longtable}}
}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with args.summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    sections = []
    for grid in GRIDS:
        grid_rows = rows_for_grid(rows, grid)
        expected = 89 if grid == "100k" else 90
        if len(grid_rows) != expected:
            raise SystemExit(f"{grid}: expected {expected} factor-matrix rows, found {len(grid_rows)}")
        cells = integer(grid_rows[0]["cells"])
        missing_note = "；缺少 S224/4096/请求 B=256" if grid == "100k" else ""
        sections.append(longtable(
            f"{grid} 网格（{cells} cells）：端到端时间与同网格 S316 参考误差{missing_note}",
            f"tab:factor-time-{grid}",
            "rrrrrrrr",
            "方向数 & 样本数 & 请求 B & 实际 B & SI total/s & RSI total/s & RSI/SI & error (\\%)",
            [timing_row(row) for row in grid_rows],
        ))
    for grid in GRIDS:
        grid_rows = rows_for_grid(rows, grid)
        cells = integer(grid_rows[0]["cells"])
        missing_note = "；缺少 S224/4096/请求 B=256" if grid == "100k" else ""
        sections.append(longtable(
            f"{grid} 网格（{cells} cells）：SI/RSI 并行效率{missing_note}",
            f"tab:factor-efficiency-{grid}",
            "rrrrrrrrrr",
            "方向数 & 样本数 & 请求 B & 实际 B & SI sweep/s & $E_{\\rm SI}$ & RSI sweep/s & $E_{\\rm RSI}$ & RSI/SI & RSI plan/s",
            [efficiency_row(row) for row in grid_rows],
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
