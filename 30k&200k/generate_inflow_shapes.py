#!/usr/bin/env python3
"""Plot the actual mesh-discretized circular inflow regions used by the solver."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import meshio
import numpy as np


SOURCE_CENTER = np.array([0.5, 0.5])
SOURCE_RADIUS = 0.2 / np.sqrt(np.pi)
IDEAL_AREA = np.pi * SOURCE_RADIUS**2


def bottom_triangles(mesh_path):
    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :3]
    triangles = np.concatenate(
        [block.data for block in mesh.cells if block.type == "triangle"], axis=0
    )
    on_bottom = np.all(np.abs(points[triangles, 1]) < 1.0e-10, axis=1)
    return points, triangles[on_bottom]


def triangle_areas_xz(vertices):
    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def boundary_segments(points, active_triangles):
    edge_counts = Counter()
    for triangle in active_triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            edge_counts[tuple(sorted((int(a), int(b))))] += 1
    return [points[list(edge), :][:, [0, 2]] for edge, n in edge_counts.items() if n == 1]


def draw_panel(ax, all_vertices, active_vertices, outline, zoom):
    ax.add_collection(
        PolyCollection(
            all_vertices,
            facecolors="#F2F4F5",
            edgecolors="#CAD1D5",
            linewidths=0.18,
            rasterized=True,
        )
    )
    ax.add_collection(
        PolyCollection(
            active_vertices,
            facecolors="#D95F4B",
            edgecolors="#8E3025",
            linewidths=0.32,
            rasterized=True,
        )
    )
    ax.add_collection(LineCollection(outline, colors="#651E17", linewidths=1.5))
    ideal = plt.Circle(
        SOURCE_CENTER,
        SOURCE_RADIUS,
        fill=False,
        color="#176B87",
        linestyle=(0, (5, 3)),
        linewidth=1.7,
        label="Ideal circle",
    )
    ax.add_patch(ideal)
    ax.plot(*SOURCE_CENTER, marker="+", color="#176B87", markersize=7, mew=1.2)
    ax.set_aspect("equal")
    if zoom:
        pad = 1.42 * SOURCE_RADIUS
        ax.set_xlim(SOURCE_CENTER[0] - pad, SOURCE_CENTER[0] + pad)
        ax.set_ylim(SOURCE_CENTER[1] - pad, SOURCE_CENTER[1] + pad)
        ax.set_title("Inflow-region detail")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Complete bottom boundary ($y=0$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.grid(False)


def plot_shape(mesh_path, mesh_label, output_path):
    points, triangles = bottom_triangles(mesh_path)
    vertices = points[triangles][:, :, [0, 2]]
    centroids = vertices.mean(axis=1)
    active = np.sum((centroids - SOURCE_CENTER) ** 2, axis=1) <= SOURCE_RADIUS**2
    active_triangles = triangles[active]
    active_vertices = vertices[active]
    discrete_area = triangle_areas_xz(active_vertices).sum()
    area_error = 100.0 * (discrete_area / IDEAL_AREA - 1.0)
    outline = boundary_segments(points, active_triangles)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.35))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.82, wspace=0.28)
    for ax, zoom in zip(axes, (False, True)):
        draw_panel(ax, vertices, active_vertices, outline, zoom)

    fig.suptitle(
        f"Actual mesh-discretized inflow region: {mesh_label}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.045,
        (
            f"Active bottom faces: {active.sum():,} / {len(triangles):,}     "
            f"Discrete area: {discrete_area:.8f}     "
            f"Ideal-circle area: {IDEAL_AREA:.8f}     "
            f"Area difference: {area_error:+.3f}%"
        ),
        ha="center",
        fontsize=10.5,
    )
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)

    return len(triangles), int(active.sum()), discrete_area, area_error


def main():
    here = Path(__file__).resolve().parent
    root = here.parent
    cases = [
        (root / "gmsh_work/example1.msh", "30k (36,470 cells)", here / "30k_actual_inflow_region.png"),
        (root / "gmsh_work/mesh200k/example1.msh", "200k (164,151 cells)", here / "200k_actual_inflow_region.png"),
    ]
    for mesh_path, label, output_path in cases:
        total, active, area, error = plot_shape(mesh_path, label, output_path)
        print(
            f"{label}: bottom_faces={total}, active_faces={active}, "
            f"area={area:.12g}, area_error={error:+.6f}%, output={output_path}"
        )


if __name__ == "__main__":
    main()
