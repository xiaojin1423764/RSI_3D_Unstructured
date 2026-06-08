import os

import numpy as np
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
VMAX = 2.2
LAYERED_Z_SLICES = [0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82]
LAYERED_VISIBLE_MIN = 0.015
LAYERED_NORM = PowerNorm(gamma=0.42, vmin=LAYERED_VISIBLE_MIN, vmax=VMAX)


def draw_axes(ax):
    edge_color = "0.68"
    corners = [
        ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)),
        ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)),
        ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
        ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
    ]
    for a, b in corners:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=edge_color, lw=0.6, alpha=0.45)
    ax.plot([0, 1.06], [0, 0], [0, 0], color="black", lw=1.2)
    ax.plot([0, 0], [0, 1.06], [0, 0], color="black", lw=1.2)
    ax.plot([0, 0], [0, 0], [0, 1.06], color="black", lw=1.2)
    ax.text(1.10, 0, 0, "x")
    ax.text(0, 1.10, 0, "y")
    ax.text(0, 0, 1.10, "z")


def plot_layered_slices(df, title, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    mappable = None
    for z0 in LAYERED_Z_SLICES:
        sub = df.assign(zdist=(df["z"] - z0).abs())
        tol = sub["zdist"].quantile(0.035)
        sub = sub[sub["zdist"] <= tol].copy()
        if len(sub) < 3:
            continue
        val = sub["phi0"].to_numpy()
        if val.max() < LAYERED_VISIBLE_MIN:
            continue
        triang = tri.Triangulation(sub["x"].to_numpy(), sub["y"].to_numpy())
        mappable = ax.tricontourf(
            triang,
            val,
            levels=np.linspace(LAYERED_VISIBLE_MIN, VMAX, 80),
            norm=LAYERED_NORM,
            zdir="z",
            offset=z0,
            alpha=0.38,
            antialiased=True,
        )

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=r"$\phi_0$", shrink=0.72, pad=0.08)

    draw_axes(ax)
    ax.set_title(f"{title}, layered slices")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=132)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


os.makedirs("Figures", exist_ok=True)
PREFIX = source_prefix()

for file, title, out_dir in FILES:
    df = pd.read_csv(file)
    base = figure_base(file, PREFIX)
    figure_dir = os.path.join("Figures", out_dir)
    os.makedirs(figure_dir, exist_ok=True)
    plot_layered_slices(df, title, os.path.join(figure_dir, f"{base}_3D_layers.png"))
