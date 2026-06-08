import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import PowerNorm
from scipy.interpolate import griddata

from plot_naming import figure_base, source_prefix


FILES = [
    ("csv_data/figure5_SI_fine.csv", "SI fine", "si_fine"),
    ("csv_data/figure5_SI_coarse.csv", "SI coarse", "si_coarse"),
    ("csv_data/figure5_RSI.csv", "RSI", "rsi"),
    ("csv_data/figure5_RSI_tail.csv", "RSI tail average", "rsi_tail"),
]
VMIN = 0.0
VMAX = 2.2
NORM = PowerNorm(gamma=0.55, vmin=VMIN, vmax=VMAX)
VOLUME_GRID_N = 64
VOLUME_VISIBLE_MIN = 0.01
VOLUME_VIEWS = [
    ("iso", 24, -58),
    ("iso_back", 24, 132),
    ("front", 0, -90),
    ("side", 0, 0),
    ("top", 90, -90),
]


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


def plot_3d_field(df, title, out_prefix, views):
    pts = df[["x", "y", "z"]].to_numpy()
    values = df["phi0"].to_numpy()
    edges = np.linspace(0.0, 1.0, VOLUME_GRID_N + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing="ij")

    vol = griddata(pts, values, (gx, gy, gz), method="linear")
    missing = np.isnan(vol)
    if missing.any():
        vol[missing] = griddata(pts, values, (gx, gy, gz), method="nearest")[missing]
    vol = np.clip(vol, 0.0, None)

    filled = vol >= VOLUME_VISIBLE_MIN
    colors = cm.viridis(NORM(vol))
    alpha = np.clip((vol - VOLUME_VISIBLE_MIN) / (0.22 - VOLUME_VISIBLE_MIN), 0.0, 1.0)
    colors[..., 3] = 0.22 + 0.70 * alpha
    colors[~filled, 3] = 0.0
    ex, ey, ez = np.meshgrid(edges, edges, edges, indexing="ij")

    mappable = cm.ScalarMappable(norm=NORM, cmap="viridis")
    mappable.set_array([])

    for view_name, elev, azim in views:
        plot_3d_volume(
            title,
            f"{out_prefix}_{view_name}.png",
            ex,
            ey,
            ez,
            filled,
            colors,
            mappable,
            elev,
            azim,
        )


def plot_3d_volume(title, out_path, ex, ey, ez, filled, colors, mappable, elev, azim):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        ex,
        ey,
        ez,
        filled,
        facecolors=colors,
        edgecolor=(1.0, 1.0, 1.0, 0.018),
        linewidth=0.12,
        shade=False,
    )
    fig.colorbar(mappable, ax=ax, label=r"$\phi_0$", shrink=0.72, pad=0.08)

    draw_axes(ax)
    ax.set_title(f"{title}, 3D")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
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
    views = VOLUME_VIEWS if out_dir == "si_coarse" else [("iso_back", 24, 132)]
    plot_3d_field(df, title, os.path.join(figure_dir, f"{base}_3D"), views)
