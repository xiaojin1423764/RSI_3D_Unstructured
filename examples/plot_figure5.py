import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import matplotlib.tri as tri

from plot_naming import figure_base, source_prefix


FILES = [
    ("csv_data/figure5_SI_fine.csv", "SI fine", "si_fine"),
    ("csv_data/figure5_SI_coarse.csv", "SI coarse", "si_coarse"),
    ("csv_data/figure5_RSI.csv", "RSI", "rsi"),
    ("csv_data/figure5_RSI_tail.csv", "RSI tail average", "rsi_tail"),
]
Z_SLICES = [0.50]
VMIN = 0.0
VMAX = 2.2
NORM = PowerNorm(gamma=0.55, vmin=VMIN, vmax=VMAX)


os.makedirs("Figures", exist_ok=True)
PREFIX = source_prefix()

for file, title, out_dir in FILES:
    df = pd.read_csv(file)
    base = figure_base(file, PREFIX)
    figure_dir = os.path.join("Figures", out_dir)
    os.makedirs(figure_dir, exist_ok=True)

    for z0 in Z_SLICES:
        df["zdist"] = (df["z"] - z0).abs()
        tol = df["zdist"].quantile(0.12)
        sub = df[df["zdist"] <= tol].copy()
        if len(sub) < 3:
            continue

        triang = tri.Triangulation(sub["x"].to_numpy(), sub["y"].to_numpy())
        val = sub["phi0"].to_numpy()

        plt.figure(figsize=(6, 5))
        plt.tricontourf(triang, val, levels=80, norm=NORM)
        plt.colorbar(label=r"$\phi_0$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{title}, z≈{z0:.2f}")
        plt.axis("equal")
        plt.tight_layout()

        out = os.path.join(figure_dir, f"{base}_z{z0:.2f}.png")
        plt.savefig(out, dpi=300)
        plt.close()
