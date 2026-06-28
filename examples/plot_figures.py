import argparse
import itertools
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rsi")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa-rsi")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from plot_naming import figure_base, source_prefix


BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "csv_data")
FINE_CSV_DIR = os.path.join(BASE_DIR, "csv_data_finemesh_backup")
MESH200K_CSV_DIR = os.path.join(BASE_DIR, "csv_data_mesh200k")
FIGURE_DIR = os.path.join(BASE_DIR, "Figures")
PAPER_SLICE_DIR = os.path.join(BASE_DIR, "..", "Figure5_paper_slices_coarsemesh")
FINE_PAPER_SLICE_DIR = os.path.join(BASE_DIR, "..", "Figure5_paper_slices_finemesh")
MESH200K_PAPER_SLICE_DIR = os.path.join(BASE_DIR, "..", "Figure5_paper_slices_mesh200k")
GAUSS_FIGURE_DIR = os.path.join(BASE_DIR, "..", "Gauss_figures")
MSH_FILE = os.path.join(BASE_DIR, "..", "gmsh_work", "example1.msh")
FINE_MSH_FILE = os.path.join(BASE_DIR, "..", "gmsh_work", "finemesh_backup", "example1.msh")
MESH200K_MSH_FILE = os.path.join(BASE_DIR, "..", "gmsh_work", "mesh200k", "example1.msh")

FIELD_FILES = [
    ("figure5_SI_fine.csv", "SI fine", "si_fine", "S32, M=1088"),
    ("figure5_SI_coarse.csv", "SI coarse", "si_coarse", "S4, M=24"),
    ("figure5_RSI.csv", "RSI", "rsi", "S32, M=1088, samples=512"),
    ("figure5_RSI_tail.csv", "RSI tail average", "rsi_tail", "S32, M=1088, samples=512, tail=10"),
]
SOURCE_CASES = [
    ("Rec", os.path.join(CSV_DIR, "Rec")),
    ("Cir", os.path.join(CSV_DIR, "Cir")),
]
FINE_SOURCE_CASES = [
    ("Rec", os.path.join(FINE_CSV_DIR, "Rec")),
    ("Cir", os.path.join(FINE_CSV_DIR, "Cir")),
]
MESH200K_SOURCE_CASES = [
    ("Rec", os.path.join(MESH200K_CSV_DIR, "Rec")),
    ("Cir", os.path.join(MESH200K_CSV_DIR, "Cir")),
]

Y_SLICES = [0.00, 0.25, 0.50, 0.75, 1.00]
PAPER_Y_SLICE_VALUES = [round(0.01 * i, 2) for i in range(31)]
PAPER_SOURCE_CENTER_Z = 0.5
PAPER_SOURCE_RADIUS = 0.2 / np.sqrt(np.pi)
PAPER_Z_SLICE_VALUES = [
    round(v, 2)
    for v in np.arange(
        PAPER_SOURCE_CENTER_Z - PAPER_SOURCE_RADIUS,
        PAPER_SOURCE_CENTER_Z + PAPER_SOURCE_RADIUS + 0.005,
        0.01,
    )
]
LAYER_STACK_Y_SLICES = [0.025 + 0.05 * i for i in range(20)]

VMIN = 0.0
VMAX = 2.2
FIELD_NORM = PowerNorm(gamma=0.55, vmin=VMIN, vmax=VMAX)
Y_SLICE_NORM = PowerNorm(gamma=0.35, vmin=0.0, vmax=0.12)
PAPER_FIGURE5_NORM = PowerNorm(gamma=0.55, vmin=0.0, vmax=2.7)
PAPER_FIGURE5_LEVELS = np.linspace(0.0, 2.7, 181)
PAPER_FIGURE5_CMAP = "turbo"
PAPER_SLICE_GRID_N = 260
PAPER_GAUSS_GRID_N = 360
PAPER_GAUSS_SIGMA = 1.6
PAPER_SLICE_IDW_K = 96
PAPER_SLICE_IDW_POWER = 1.65
PAPER_SLICE_SMOOTH_SIGMA = 0.0
PAPER_SLICE_TOL = 1.0e-12

VOLUME_GRID_N = 64
LEGACY_VOXEL_GRID_N = 128
VOLUME_VISIBLE_MIN = 0.01
VOLUME_VIEWS = [("iso_back", 24, 132)]
PYVISTA_GRID_N = 96
PYVISTA_LOG_MIN = np.log10(0.0035)
PYVISTA_LOG_MAX = np.log10(0.12)
PYVISTA_VOLUME_GRID_N = 64
PYVISTA_ISOSURFACE_SMOOTH_SIGMA = 2.0
PYVISTA_ISOSURFACE_SMOOTH_PASSES = 80
PYVISTA_ISOSURFACE_RELAXATION = 0.08

LAYER_STACK_ALPHA_LOW = 0.01
LAYER_STACK_ALPHA_HIGH = 0.07
LAYER_STACK_ALPHA_MAX = 0.20


def csv_path(name):
    return os.path.join(CSV_DIR, name)


def source_csv_path(source_dir, name):
    return os.path.join(source_dir, name)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def titled_with_params(title, params):
    return f"{title}\n{params}"


def draw_box_axes(ax):
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


def finish_3d_axis(ax, elev, azim):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")


def plot_figure2(show=False):
    df = pd.read_csv(csv_path("figure2_data.csv"))
    ensure_dir(FIGURE_DIR)
    fit_rows = []

    for scattering in ["isotropic", "anisotropic"]:
        sub = df[df["scattering"] == scattering]
        plt.figure()

        for m_value in sorted(sub["M"].unique()):
            data = sub[sub["M"] == m_value].sort_values("S")
            samples = data["S"].to_numpy(dtype=float)
            err = data["e_RSI_N"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(np.log(samples), np.log(err), 1)
            order = -slope
            fit_rows.append({
                "scattering": scattering,
                "M": m_value,
                "slope": slope,
                "order": order,
            })
            plt.loglog(samples, err, marker="o",
                       label=f"M={m_value}, order={order:.3f}")

        sample_ref = np.array(sorted(sub["S"].unique()), dtype=float)
        err_ref = sub["e_RSI_N"].max()
        plt.loglog(
            sample_ref,
            err_ref * (sample_ref / sample_ref[0]) ** (-0.5),
            linestyle="--",
            label=r"$S^{-0.5}$",
        )

        plt.xlabel("S")
        plt.ylabel(r"$e^{(N)}_{RSI}$")
        plt.title(scattering)
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f"figure2_{scattering}.png"), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    fit_df = pd.DataFrame(fit_rows)
    print("拟合收敛阶：")
    print(fit_df.to_string(index=False))


def slice_near_plane(df, axis, value, quantile=0.12):
    dist = (df[axis] - value).abs()
    tol = max(float(dist.quantile(quantile)), float(dist.nsmallest(3).max()))
    return df[dist <= tol].copy()


def plot_field_slice(df, title, out_path, axis, value):
    sub = slice_near_plane(df, axis, value)
    if len(sub) < 3:
        return False

    if axis == "z":
        horizontal, vertical = "x", "y"
    elif axis == "y":
        horizontal, vertical = "x", "z"
    else:
        raise ValueError("axis must be 'y' or 'z'")

    triang = tri.Triangulation(sub[horizontal].to_numpy(), sub[vertical].to_numpy())
    values = sub["phi0"].to_numpy()

    norm = Y_SLICE_NORM if axis == "y" else FIELD_NORM

    plt.figure(figsize=(6, 5))
    plt.tricontourf(triang, values, levels=80, norm=norm, extend="max")
    plt.colorbar(label=r"$\phi_0$")
    plt.xlabel(horizontal)
    plt.ylabel(vertical)
    plt.title(f"{title}, {axis}~={value:.2f}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def interpolate_strict_plane_idw(df, axis, value):
    if axis == "z":
        horizontal, vertical = "x", "y"
    elif axis == "y":
        horizontal, vertical = "x", "z"
    else:
        raise ValueError("axis must be 'y' or 'z'")

    coords = np.linspace(0.0, 1.0, PAPER_SLICE_GRID_N)
    gh, gv = np.meshgrid(coords, coords, indexing="xy")
    plane_points = np.empty((gh.size, 3), dtype=float)
    plane_points[:, {"x": 0, "y": 1, "z": 2}[axis]] = value
    plane_points[:, {"x": 0, "y": 1, "z": 2}[horizontal]] = gh.ravel()
    plane_points[:, {"x": 0, "y": 1, "z": 2}[vertical]] = gv.ravel()

    points = df[["x", "y", "z"]].to_numpy(dtype=float)
    values = df["phi0"].to_numpy(dtype=float)
    tree = cKDTree(points)
    distances, indices = tree.query(
        plane_points,
        k=min(PAPER_SLICE_IDW_K, len(points)),
        workers=-1,
    )
    distances = np.maximum(distances, 1.0e-12)
    weights = 1.0 / distances ** PAPER_SLICE_IDW_POWER
    field = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    field = field.reshape(gh.shape)
    field = gaussian_filter(field, sigma=PAPER_SLICE_SMOOTH_SIGMA, mode="nearest")
    field = np.clip(field, 0.0, None)
    return coords, coords, field, horizontal, vertical


def project_cell_values_to_vertices(tets, cell_values, cell_volumes, n_points):
    vertex_values = np.zeros(n_points, dtype=float)
    vertex_weights = np.zeros(n_points, dtype=float)
    weights = np.repeat(cell_volumes / 4.0, 4)
    np.add.at(vertex_values, tets.ravel(), np.repeat(cell_values, 4) * weights)
    np.add.at(vertex_weights, tets.ravel(), weights)
    valid = vertex_weights > 0.0
    vertex_values[valid] /= vertex_weights[valid]
    return vertex_values


def slice_tetrahedral_linear_field(points, tets, vertex_values, axis, value):
    if axis == "z":
        axis_idx, horizontal_idx, vertical_idx = 2, 0, 1
        horizontal, vertical = "x", "y"
    elif axis == "y":
        axis_idx, horizontal_idx, vertical_idx = 1, 0, 2
        horizontal, vertical = "x", "z"
    else:
        raise ValueError("axis must be 'y' or 'z'")

    triangles = []
    coords = []
    values = []
    vertex_lookup = {}
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def add_point(point3, point_value):
        key = tuple(np.round(point3[[horizontal_idx, vertical_idx]], 12))
        idx = vertex_lookup.get(key)
        if idx is None:
            idx = len(coords)
            vertex_lookup[key] = idx
            coords.append([point3[horizontal_idx], point3[vertical_idx]])
            values.append(point_value)
        return idx

    for tet in tets:
        ps = points[tet]
        vals = vertex_values[tet]
        signed = ps[:, axis_idx] - value
        if np.all(signed > PAPER_SLICE_TOL) or np.all(signed < -PAPER_SLICE_TOL):
            continue

        poly = []
        seen = set()
        for i, j in edge_pairs:
            di, dj = signed[i], signed[j]
            if abs(di) <= PAPER_SLICE_TOL and abs(dj) <= PAPER_SLICE_TOL:
                for endpoint in (i, j):
                    idx = add_point(ps[endpoint], vals[endpoint])
                    if idx not in seen:
                        seen.add(idx)
                        poly.append(idx)
                continue
            if abs(di) <= PAPER_SLICE_TOL:
                idx = add_point(ps[i], vals[i])
            elif abs(dj) <= PAPER_SLICE_TOL:
                idx = add_point(ps[j], vals[j])
            elif di * dj < 0.0:
                t = di / (di - dj)
                point3 = ps[i] + t * (ps[j] - ps[i])
                point_value = vals[i] + t * (vals[j] - vals[i])
                idx = add_point(point3, point_value)
            else:
                continue

            if idx not in seen:
                seen.add(idx)
                poly.append(idx)

        if len(poly) < 3:
            continue

        poly_coords = np.asarray([coords[i] for i in poly])
        center = poly_coords.mean(axis=0)
        order = np.argsort(np.arctan2(poly_coords[:, 1] - center[1], poly_coords[:, 0] - center[0]))
        poly = [poly[i] for i in order]
        for i in range(1, len(poly) - 1):
            triangles.append([poly[0], poly[i], poly[i + 1]])

    if not triangles:
        return None, None, None, horizontal, vertical

    return (
        np.asarray(coords, dtype=float),
        np.asarray(triangles, dtype=int),
        np.clip(np.asarray(values, dtype=float), 0.0, None),
        horizontal,
        vertical,
    )


def plot_paper_tetra_slice_values(vertex_values, out_path, axis, value, mesh_cache):
    points, tets = mesh_cache
    coords, triangles, values, horizontal, vertical = slice_tetrahedral_linear_field(
        points,
        tets,
        vertex_values,
        axis,
        value,
    )
    if coords is None:
        return False

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(4.1, 3.55))
    mappable = ax.tripcolor(
        triang,
        values,
        shading="gouraud",
        cmap=PAPER_FIGURE5_CMAP,
        norm=PAPER_FIGURE5_NORM,
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, extend="max")
    cbar.set_ticks(np.arange(0.5, 2.6, 0.5))
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(horizontal, fontsize=9)
    ax.set_ylabel(vertical, fontsize=9)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.tick_params(labelsize=8, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def plot_paper_tetra_slice_values_gauss(vertex_values, out_path, axis, value, mesh_cache):
    points, tets = mesh_cache
    coords, triangles, values, horizontal, vertical = slice_tetrahedral_linear_field(
        points,
        tets,
        vertex_values,
        axis,
        value,
    )
    if coords is None:
        return False

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    interpolator = tri.LinearTriInterpolator(triang, values)
    grid = np.linspace(0.0, 1.0, PAPER_GAUSS_GRID_N)
    gh, gv = np.meshgrid(grid, grid, indexing="xy")
    field = interpolator(gh, gv)
    if np.ma.isMaskedArray(field):
        field = field.filled(np.nan)
    field = np.asarray(field, dtype=float)
    missing = ~np.isfinite(field)
    if missing.any():
        nearest = griddata(coords, values, (gh, gv), method="nearest")
        field[missing] = nearest[missing]
    missing = ~np.isfinite(field)
    if missing.any():
        field[missing] = 0.0
    field = gaussian_filter(np.clip(field, 0.0, None), sigma=PAPER_GAUSS_SIGMA, mode="nearest")

    fig, ax = plt.subplots(figsize=(4.1, 3.55))
    mappable = ax.imshow(
        field,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="bilinear",
        aspect="equal",
        cmap=PAPER_FIGURE5_CMAP,
        norm=PAPER_FIGURE5_NORM,
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, extend="max")
    cbar.set_ticks(np.arange(0.5, 2.6, 0.5))
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(horizontal, fontsize=9)
    ax.set_ylabel(vertical, fontsize=9)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.tick_params(labelsize=8, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def plot_paper_tetra_slice(df, out_path, axis, value, mesh_cache):
    points, tets, cell_volumes = mesh_cache
    cell_values = df.sort_values("cell_id")["phi0"].to_numpy(dtype=float)
    if len(cell_values) != len(tets):
        raise ValueError(f"cell value count {len(cell_values)} does not match tetra count {len(tets)}")

    vertex_values = project_cell_values_to_vertices(tets, cell_values, cell_volumes, len(points))
    return plot_paper_tetra_slice_values(vertex_values, out_path, axis, value, (points, tets))


def plot_paper_field_slice(df, out_path, axis, value):
    xvals, yvals, field, horizontal, vertical = interpolate_strict_plane_idw(df, axis, value)

    fig, ax = plt.subplots(figsize=(4.1, 3.55))
    mappable = ax.imshow(
        field,
        origin="lower",
        extent=(float(xvals[0]), float(xvals[-1]), float(yvals[0]), float(yvals[-1])),
        interpolation="bilinear",
        aspect="equal",
        cmap=PAPER_FIGURE5_CMAP,
        norm=PAPER_FIGURE5_NORM,
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, extend="max")
    cbar.set_ticks(np.arange(0.5, 2.6, 0.5))
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.6)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(horizontal, fontsize=9)
    ax.set_ylabel(vertical, fontsize=9)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.tick_params(labelsize=8, length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def available_source_cases(source_cases=SOURCE_CASES, csv_dir=CSV_DIR):
    cases = []
    for prefix, source_dir in source_cases:
        if all(os.path.exists(source_csv_path(source_dir, file_name)) for file_name, *_ in FIELD_FILES):
            cases.append((prefix, source_dir))
    if cases:
        return cases

    prefix = source_prefix()
    if all(os.path.exists(os.path.join(csv_dir, file_name)) for file_name, *_ in FIELD_FILES):
        return [(prefix, csv_dir)]
    return []


def plot_figure5_slices():
    ensure_dir(FIGURE_DIR)
    for prefix, source_dir in available_source_cases():
        for file_name, title, out_dir, params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(FIGURE_DIR, out_dir)
            ensure_dir(out_base_dir)
            plot_title = titled_with_params(title, params)

            for y0 in Y_SLICES:
                plot_field_slice(
                    df,
                    plot_title,
                    os.path.join(out_base_dir, f"{base}_y{y0:.2f}.png"),
                    "y",
                    y0,
                )


def plot_figure5_paper_slices(
    source_cases=SOURCE_CASES,
    csv_dir=CSV_DIR,
    msh_file=MSH_FILE,
    paper_slice_dir=PAPER_SLICE_DIR,
    plotter=plot_paper_tetra_slice_values,
):
    points, tets = read_tetra_mesh(msh_file)
    cell_volumes = tetra_volumes(points, tets)
    mesh_cache = (points, tets)
    for prefix, source_dir in available_source_cases(source_cases, csv_dir):
        for file_name, _title, out_dir, _params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            cell_values = df.sort_values("cell_id")["phi0"].to_numpy(dtype=float)
            if len(cell_values) != len(tets):
                raise ValueError(f"cell value count {len(cell_values)} does not match tetra count {len(tets)}")
            vertex_values = project_cell_values_to_vertices(tets, cell_values, cell_volumes, len(points))
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(paper_slice_dir, out_dir)
            ensure_dir(out_base_dir)

            for axis, values in [("y", PAPER_Y_SLICE_VALUES), ("z", PAPER_Z_SLICE_VALUES)]:
                for value in values:
                    plotter(
                        vertex_values,
                        os.path.join(out_base_dir, f"{base}_{axis}{value:.2f}_paper.png"),
                        axis,
                        value,
                        mesh_cache,
                    )


def plot_figure5_paper_slices_finemesh():
    plot_figure5_paper_slices(
        source_cases=FINE_SOURCE_CASES,
        csv_dir=FINE_CSV_DIR,
        msh_file=FINE_MSH_FILE,
        paper_slice_dir=FINE_PAPER_SLICE_DIR,
    )


def plot_figure5_paper_slices_mesh200k():
    plot_figure5_paper_slices(
        source_cases=MESH200K_SOURCE_CASES,
        csv_dir=MESH200K_CSV_DIR,
        msh_file=MESH200K_MSH_FILE,
        paper_slice_dir=MESH200K_PAPER_SLICE_DIR,
    )


def plot_figure5_paper_slices_gauss():
    configs = [
        ("coarsemesh", SOURCE_CASES, CSV_DIR, MSH_FILE),
        ("finemesh", FINE_SOURCE_CASES, FINE_CSV_DIR, FINE_MSH_FILE),
        ("mesh200k", MESH200K_SOURCE_CASES, MESH200K_CSV_DIR, MESH200K_MSH_FILE),
    ]
    for name, source_cases, csv_dir, msh_file in configs:
        plot_figure5_paper_slices(
            source_cases=source_cases,
            csv_dir=csv_dir,
            msh_file=msh_file,
            paper_slice_dir=os.path.join(GAUSS_FIGURE_DIR, name),
            plotter=plot_paper_tetra_slice_values_gauss,
        )


def interpolate_volume(df, grid_n):
    pts = df[["x", "y", "z"]].to_numpy()
    values = df["phi0"].to_numpy()
    edges = np.linspace(0.0, 1.0, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing="ij")

    volume = griddata(pts, values, (gx, gy, gz), method="linear")
    missing = np.isnan(volume)
    if missing.any():
        volume[missing] = griddata(pts, values, (gx, gy, gz), method="nearest")[missing]
    volume = np.clip(volume, 0.0, None)
    return edges, centers, volume


def interpolate_volume_nearest(df, grid_n):
    pts = df[["x", "y", "z"]].to_numpy()
    values = df["phi0"].to_numpy()
    edges = np.linspace(0.0, 1.0, grid_n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing="ij")
    volume = griddata(pts, values, (gx, gy, gz), method="nearest")
    volume = np.clip(volume, 0.0, None)
    return edges, centers, volume


def plot_3d_field(df, title, out_prefix, views, include_voxel=True):
    stack_done = plot_3d_image_slice_stack(df, title, out_prefix, f"{out_prefix}_layer_stack.png")
    if not stack_done:
        stack_done = plot_3d_ray_slice_stack(df, title, f"{out_prefix}_layer_stack.png")
    if include_voxel:
        plot_3d_voxel_legacy(df, title, f"{out_prefix}_voxel3d", views)

    if stack_done:
        return


def plot_3d_voxel_legacy(df, title, out_prefix, views):
    """Old git-version voxel rendering kept as a comparison output."""
    edges, centers, volume = interpolate_volume(df, LEGACY_VOXEL_GRID_N)

    filled = volume >= VOLUME_VISIBLE_MIN
    colors = cm.viridis(FIELD_NORM(volume))
    alpha = np.clip((volume - VOLUME_VISIBLE_MIN) / (0.22 - VOLUME_VISIBLE_MIN), 0.0, 1.0)
    colors[..., 3] = 0.22 + 0.70 * alpha
    colors[~filled, 3] = 0.0
    ex, ey, ez = np.meshgrid(edges, edges, edges, indexing="ij")

    mappable = cm.ScalarMappable(norm=FIELD_NORM, cmap="viridis")
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


def plot_3d_pyvista_volume(df, title, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    _, centers, volume = interpolate_volume_nearest(df, PYVISTA_VOLUME_GRID_N)
    contrast, lo_value, hi_value = enhanced_log_field(volume)
    hi_tail = max(float(np.quantile(contrast, 0.985)), 1.0e-8)
    contrast = np.clip(contrast / hi_tail, 0.0, 1.0)
    contrast = smoothstep(0.28, 0.92, contrast) ** 1.35

    grid = pv.ImageData()
    grid.dimensions = contrast.shape
    spacing = 1.0 / float(PYVISTA_VOLUME_GRID_N - 1)
    grid.spacing = (spacing, spacing, spacing)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["ray_contrast"] = contrast.ravel(order="F")

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1500))
    plotter.set_background("white")
    plotter.add_volume(
        grid,
        scalars="ray_contrast",
        cmap="viridis",
        clim=(0.0, 1.0),
        opacity=[0.00, 0.00, 0.00, 0.00, 0.008, 0.025, 0.08, 0.24, 0.58, 1.00],
        mapper="smart",
        shade=False,
        show_scalar_bar=False,
    )

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color=(0.22, 0.22, 0.22), line_width=1.8)
    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        line_width=5,
        labels_off=False,
    )
    plotter.add_scalar_bar(
        title=f"phi0, log scale\n[{lo_value:.1e}, {hi_value:.2g}]",
        vertical=True,
        position_x=0.88,
        position_y=0.20,
        width=0.08,
        height=0.60,
        label_font_size=24,
        title_font_size=24,
        color="black",
    )
    plotter.add_text(title + ", PyVista volume rendering", position="upper_edge", font_size=22, color="black")
    plotter.camera_position = [(-0.72, 2.75, 1.10), (0.5, 0.5, 0.50), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.93
    plotter.screenshot(out_path)
    plotter.close()
    return True


def plot_3d_multilevel_isosurfaces(df, title, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    _, centers, volume = interpolate_volume_nearest(df, PYVISTA_VOLUME_GRID_N)
    volume = gaussian_filter(volume, sigma=PYVISTA_ISOSURFACE_SMOOTH_SIGMA, mode="nearest")
    contrast, lo_value, hi_value = enhanced_log_field(volume)
    hi_tail = max(float(np.quantile(contrast, 0.985)), 1.0e-8)
    contrast = np.clip(contrast / hi_tail, 0.0, 1.0)
    contrast = smoothstep(0.28, 0.92, contrast) ** 1.35

    grid = pv.ImageData()
    grid.dimensions = contrast.shape
    spacing = 1.0 / float(PYVISTA_VOLUME_GRID_N - 1)
    grid.spacing = (spacing, spacing, spacing)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["ray_contrast"] = contrast.ravel(order="F")

    positive = contrast[contrast > 0.02]
    if len(positive) == 0:
        return False

    levels = np.unique(np.quantile(positive, [0.70, 0.84, 0.93, 0.98]))
    surfaces = grid.contour(isosurfaces=levels, scalars="ray_contrast")
    if surfaces.n_points > 0:
        surfaces = surfaces.smooth(
            n_iter=PYVISTA_ISOSURFACE_SMOOTH_PASSES,
            relaxation_factor=PYVISTA_ISOSURFACE_RELAXATION,
            boundary_smoothing=True,
            feature_smoothing=True,
        )

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1500))
    plotter.set_background("white")
    plotter.add_mesh(
        surfaces,
        scalars="ray_contrast",
        cmap="viridis",
        clim=(float(levels.min()), float(levels.max())),
        opacity=0.58,
        smooth_shading=True,
        show_scalar_bar=False,
    )

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color=(0.20, 0.20, 0.20), line_width=1.8)
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="z", line_width=5, labels_off=False)
    plotter.add_scalar_bar(
        title=f"phi0, log scale\n[{lo_value:.1e}, {hi_value:.2g}]",
        vertical=True,
        position_x=0.88,
        position_y=0.20,
        width=0.08,
        height=0.60,
        label_font_size=24,
        title_font_size=24,
        color="black",
    )
    plotter.add_text(title + ", multilevel isosurfaces", position="upper_edge", font_size=22, color="black")
    plotter.camera_position = [(-0.72, 2.75, 1.10), (0.5, 0.5, 0.50), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.93
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_path)
    plotter.close()
    return True


def smoothstep(edge0, edge1, values):
    t = np.clip((values - edge0) / max(edge1 - edge0, 1.0e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def write_clean_slice_texture(df, y0, out_path, norm):
    sub = slice_near_plane(df, "y", y0, quantile=0.12)
    if len(sub) < 3:
        return False

    grid_n = 260
    x = np.linspace(0.0, 1.0, grid_n)
    z = np.linspace(0.0, 1.0, grid_n)
    gx, gz = np.meshgrid(x, z, indexing="ij")
    points = sub[["x", "z"]].to_numpy()
    values = sub["phi0"].to_numpy()
    field = griddata(points, values, (gx, gz), method="linear")
    missing = np.isnan(field)
    if missing.any():
        field[missing] = griddata(points, values, (gx, gz), method="nearest")[missing]
    field = np.clip(field, 0.0, None)

    fig = plt.figure(figsize=(5, 5), frameon=False)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    cmap = LinearSegmentedColormap.from_list(
        "viridis_layer_stack",
        plt.get_cmap("viridis")(np.linspace(0.0, 1.0, 256)),
    )
    image_field = field.T
    rgba = cmap(norm(image_field))
    rgba[..., 3] = LAYER_STACK_ALPHA_MAX * smoothstep(
        LAYER_STACK_ALPHA_LOW,
        LAYER_STACK_ALPHA_HIGH,
        image_field,
    )
    ax.imshow(
        rgba,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="bilinear",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return True


def plot_3d_image_slice_stack(df, title, out_prefix, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    base_prefix = out_prefix[:-3] if out_prefix.endswith("_3D") else out_prefix
    texture_dir = os.path.join(os.path.dirname(base_prefix), "_slice_textures")
    ensure_dir(texture_dir)
    texture_paths = []
    layer_values = df["phi0"].to_numpy(dtype=float)
    layer_vmax = max(float(np.quantile(layer_values, 0.98)), LAYER_STACK_ALPHA_HIGH)
    layer_norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=layer_vmax)
    for y0 in LAYER_STACK_Y_SLICES:
        texture_path = os.path.join(texture_dir, f"{os.path.basename(base_prefix)}_texture_y{y0:.2f}.png")
        if not write_clean_slice_texture(df, y0, texture_path, layer_norm):
            return False
        texture_paths.append((y0, texture_path))

    plotter = pv.Plotter(off_screen=True, window_size=(1900, 1500))
    plotter.set_background("white")

    for y0, image_path in texture_paths:
        plane = pv.Plane(
            center=(0.5, y0, 0.5),
            direction=(0, 1, 0),
            i_size=1.0,
            j_size=1.0,
            i_resolution=1,
            j_resolution=1,
        )
        texture = pv.read_texture(image_path)
        plotter.add_mesh(
            plane,
            texture=texture,
            lighting=False,
            opacity=1.0,
            show_edges=True,
            edge_color=(0.62, 0.62, 0.62),
            line_width=1.0,
        )

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color=(0.25, 0.25, 0.25), line_width=1.4)
    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        line_width=5,
        labels_off=False,
    )
    plotter.add_text(
        title + f", {len(LAYER_STACK_Y_SLICES)} y-slices ({LAYER_STACK_Y_SLICES[0]:.2f}-{LAYER_STACK_Y_SLICES[-1]:.2f})",
        position="upper_edge",
        font_size=22,
        color="black",
    )
    plotter.camera_position = [(-0.72, 2.75, 1.10), (0.5, 0.5, 0.50), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.95
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_path)
    plotter.close()
    return True


def enhanced_log_field(values):
    positive = np.clip(np.asarray(values, dtype=float), 1.0e-8, None)
    lo_value = max(1.0e-4, float(np.quantile(positive, 0.005)))
    hi_value = max(float(np.quantile(positive, 0.98)), 1.0e-3)
    lo = np.log10(lo_value)
    hi = np.log10(hi_value)
    mapped = np.clip((np.log10(positive) - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    return mapped ** 0.72, lo_value, hi_value


def plot_3d_ray_slice_stack(df, title, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    _, lo_value, hi_value = enhanced_log_field(df["phi0"].to_numpy())
    grid_n = 140
    x = np.linspace(0.0, 1.0, grid_n)
    z = np.linspace(0.0, 1.0, grid_n)
    gx, gz = np.meshgrid(x, z, indexing="ij")

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    plotter.set_background("white")

    for y0 in [0.25, 0.50, 0.75]:
        sub = slice_near_plane(df, "y", y0, quantile=0.075)
        points = sub[["x", "z"]].to_numpy()
        values = sub["phi0"].to_numpy()
        field = griddata(points, values, (gx, gz), method="linear")
        missing = np.isnan(field)
        if missing.any():
            field[missing] = griddata(points, values, (gx, gz), method="nearest")[missing]
        contrast, _, _ = enhanced_log_field(np.clip(field, 0.0, None))
        contrast = 0.08 + 0.92 * contrast

        plane = pv.StructuredGrid(gx, np.full_like(gx, y0), gz)
        plane.point_data["ray_contrast"] = contrast.ravel(order="F")
        plotter.add_mesh(
            plane,
            scalars="ray_contrast",
            cmap="turbo",
            clim=(0.0, 1.0),
            opacity=0.96,
            smooth_shading=False,
            lighting=False,
            show_edges=False,
            show_scalar_bar=False,
        )

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color=(0.18, 0.18, 0.18), line_width=2.0)
    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        line_width=5,
        labels_off=False,
    )
    plotter.add_scalar_bar(
        title=f"phi0, log scale\n[{lo_value:.1e}, {hi_value:.2g}]",
        vertical=False,
        position_x=0.20,
        position_y=0.055,
        width=0.60,
        height=0.08,
        label_font_size=24,
        title_font_size=24,
        color="black",
    )
    plotter.add_text(title + ", y-slice stack", position="upper_edge", font_size=22, color="black")
    plotter.camera_position = [(-0.65, 2.65, 1.22), (0.5, 0.5, 0.50), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.86
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_path)
    plotter.close()
    return True


def plot_3d_tetra_field(df, title, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    points, tets = read_tetra_mesh()
    values = df.sort_values("cell_id")["phi0"].to_numpy(dtype=float)
    if len(values) != len(tets):
        raise RuntimeError(
            f"Figure 5 数据行数({len(values)})与四面体单元数({len(tets)})不一致"
        )

    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64), tets]).ravel()
    cell_types = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    positive = np.clip(values, 1.0e-8, None)
    lo = np.log10(max(1.0e-4, float(np.quantile(positive, 0.005))))
    hi = np.log10(max(float(np.quantile(positive, 0.98)), 1.0e-3))
    enhanced = np.clip((np.log10(positive) - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    enhanced = enhanced ** 0.72
    grid.cell_data["ray_contrast"] = enhanced

    surface = grid.extract_surface()

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    plotter.set_background("white")
    plotter.add_mesh(
        surface,
        scalars="ray_contrast",
        cmap="turbo",
        clim=(0.0, 1.0),
        opacity=0.94,
        smooth_shading=False,
        lighting=False,
        show_edges=False,
        show_scalar_bar=False,
    )

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color="black", line_width=2.4)
    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        line_width=5,
        labels_off=False,
    )
    plotter.add_scalar_bar(
        title="enhanced phi0",
        vertical=True,
        position_x=0.88,
        position_y=0.20,
        width=0.08,
        height=0.60,
        label_font_size=24,
        title_font_size=28,
    )
    plotter.add_text(title + ", tetrahedral mesh", position="upper_edge", font_size=22, color="black")
    plotter.camera_position = [(-0.75, 3.15, 1.35), (0.5, 0.5, 0.50), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.94
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_path)
    plotter.close()
    return True


def plot_3d_surface_slice(df, title, out_path):
    try:
        import pyvista as pv
    except ImportError:
        return False

    sub = slice_near_plane(df, "y", 0.50, quantile=0.10)
    grid_n = 90
    x = np.linspace(0.0, 1.0, grid_n)
    z = np.linspace(0.0, 1.0, grid_n)
    gx, gz = np.meshgrid(x, z, indexing="ij")
    points = sub[["x", "z"]].to_numpy()
    values = sub["phi0"].to_numpy()
    field = griddata(points, values, (gx, gz), method="linear")
    missing = np.isnan(field)
    if missing.any():
        field[missing] = griddata(points, values, (gx, gz), method="nearest")[missing]
    field = np.clip(field, 0.0, None)

    height = 0.50 + 0.34 * np.log10(1.0 + 85.0 * field) / np.log10(1.0 + 85.0 * max(field.max(), 1.0e-12))
    mesh = pv.StructuredGrid(gx, np.full_like(gx, 0.50), height)
    mesh.point_data["phi0"] = field.ravel(order="F")
    wire = mesh.extract_all_edges()

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1500))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="phi0",
        cmap="turbo",
        clim=(0.0, max(0.12, float(np.quantile(field, 0.995)))),
        smooth_shading=True,
        show_scalar_bar=False,
    )
    plotter.add_mesh(wire, color="black", line_width=0.45, opacity=0.22)

    outline = pv.Box(bounds=(0, 1, 0, 1, 0, 1)).outline()
    plotter.add_mesh(outline, color=(0.70, 0.70, 0.70), line_width=2)
    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        line_width=5,
        labels_off=False,
    )
    plotter.add_scalar_bar(
        title="phi0",
        vertical=True,
        position_x=0.88,
        position_y=0.20,
        width=0.08,
        height=0.60,
        label_font_size=24,
        title_font_size=28,
    )
    plotter.add_text(title + ", y=0.50 surface", position="upper_edge", font_size=22, color="black")
    plotter.camera_position = [(-0.75, 3.15, 1.35), (0.5, 0.5, 0.55), (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.93
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_path)
    plotter.close()
    return True


def plot_3d_volume(title, out_path, ex, ey, ez, filled, colors, mappable, elev, azim):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        ex,
        ey,
        ez,
        filled,
        facecolors=colors,
        edgecolor=(1.0, 1.0, 1.0, 0.0),
        linewidth=0.0,
        shade=False,
    )
    fig.colorbar(mappable, ax=ax, label=r"$\phi_0$", shrink=0.72, pad=0.08)

    draw_box_axes(ax)
    ax.set_title(f"{title}, 3D")
    finish_3d_axis(ax, elev, azim)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_figure5_3d(include_layers=True, include_voxel=True):
    ensure_dir(FIGURE_DIR)
    for prefix, source_dir in available_source_cases():
        for file_name, title, out_dir, params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(FIGURE_DIR, out_dir)
            ensure_dir(out_base_dir)
            plot_title = titled_with_params(title, params)
            if include_layers:
                plot_3d_field(
                    df,
                    plot_title,
                    os.path.join(out_base_dir, base),
                    VOLUME_VIEWS,
                    include_voxel=include_voxel,
                )
            elif include_voxel:
                plot_3d_voxel_legacy(df, plot_title, os.path.join(out_base_dir, f"{base}_voxel3d"), VOLUME_VIEWS)


def plot_figure5_outputs():
    plot_figure5_slices()
    plot_figure5_3d(include_layers=True, include_voxel=False)


def plot_figure5_voxel3d():
    plot_figure5_3d(include_layers=False, include_voxel=True)


def plot_figure5_volume3d():
    ensure_dir(FIGURE_DIR)
    for prefix, source_dir in available_source_cases():
        for file_name, title, out_dir, params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(FIGURE_DIR, out_dir)
            ensure_dir(out_base_dir)
            out_path = os.path.join(out_base_dir, f"{base}_volume3d_iso_back.png")
            print(f"Writing {out_path}", flush=True)
            plot_3d_pyvista_volume(
                df,
                titled_with_params(title, params),
                out_path,
            )


def plot_figure5_isosurfaces():
    ensure_dir(FIGURE_DIR)
    for prefix, source_dir in available_source_cases():
        for file_name, title, out_dir, params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(FIGURE_DIR, out_dir)
            ensure_dir(out_base_dir)
            out_path = os.path.join(out_base_dir, f"{base}_isosurface_iso_back.png")
            print(f"Writing {out_path}", flush=True)
            plot_3d_multilevel_isosurfaces(
                df,
                titled_with_params(title, params),
                out_path,
            )


def read_tetra_mesh(msh_file=MSH_FILE):
    mesh = meshio.read(msh_file)
    points = mesh.points[:, :3]
    tets = None
    for block in mesh.cells:
        if block.type == "tetra":
            tets = block.data
            break
    if tets is None:
        raise RuntimeError("没有找到 tetra 四面体单元")
    return points, tets


def tetra_edges(tets):
    edges = set()
    for tet in tets:
        for a, b in itertools.combinations(tet, 2):
            edges.add(tuple(sorted((int(a), int(b)))))
    return np.array(sorted(edges), dtype=int)


def tetra_boundary_faces(tets):
    face_count = {}
    for tet in tets:
        for face in itertools.combinations(tet, 3):
            key = tuple(sorted(int(v) for v in face))
            face_count[key] = face_count.get(key, 0) + 1
    return np.array([face for face, count in face_count.items() if count == 1], dtype=int)


def tetra_volumes(points, tets):
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.abs(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6.0


def edge_lengths(points, edges):
    return np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)


def plot_tetra_mesh():
    points, tets = read_tetra_mesh()
    edges = tetra_edges(tets)
    boundary_faces = tetra_boundary_faces(tets)
    volumes = tetra_volumes(points, tets)
    lengths = edge_lengths(points, edges)
    mesh_stats = (
        f"{len(points):,} vertices, {len(tets):,} tetrahedra, {len(boundary_faces):,} boundary faces\n"
        f"mean cell volume {volumes.mean():.2e}, mean edge length {lengths.mean():.3f}"
    )
    out_dir = os.path.join(FIGURE_DIR, "mesh")
    ensure_dir(out_dir)

    edge_segments = [(points[a], points[b]) for a, b in edges]
    for view_name, elev, azim in [("iso", 24, -58), ("iso_back", 24, 132), ("top", 90, -90)]:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Line3DCollection(edge_segments, colors="0.15", linewidths=0.12, alpha=0.22))
        ax.plot_trisurf(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            triangles=boundary_faces,
            color="white",
            edgecolor="0.20",
            linewidth=0.25,
            alpha=0.08,
            shade=False,
        )
        draw_box_axes(ax)
        ax.set_title(f"Tetrahedral mesh\n{mesh_stats}", fontsize=11)
        finish_3d_axis(ax, elev, azim)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tetra_mesh_{view_name}.png"), dpi=300)
        plt.close()


def level_symmetric_sn(sn_order):
    if sn_order < 2 or sn_order % 2 != 0:
        raise ValueError("S_N order must be an even integer >= 2")
    levels = sn_order // 2
    level_sum = levels + 2
    directions = []
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            k = level_sum - i - j
            if k < 1 or k > levels:
                continue
            norm = np.sqrt(i * i + j * j + k * k)
            base = np.array([i / norm, j / norm, k / norm])
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        directions.append(base * np.array([sx, sy, sz]))
    directions = np.array(directions)
    expected = sn_order * (sn_order + 2)
    if len(directions) != expected:
        raise RuntimeError(f"S{sn_order} direction count error: {len(directions)} != {expected}")
    return directions


def plot_angle_quadrature(sn_orders=(4, 16)):
    out_dir = os.path.join(FIGURE_DIR, "angles")
    ensure_dir(out_dir)
    sphere_u = np.linspace(0, 2 * np.pi, 64)
    sphere_v = np.linspace(0, np.pi, 32)
    sx = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    sy = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    sz = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))

    for sn_order in sn_orders:
        directions = level_symmetric_sn(sn_order)
        direction_count = len(directions)
        octant_count = direction_count // 8
        angular_stats = f"M={direction_count} directions, {octant_count} per octant"

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(sx, sy, sz, color="white", edgecolor="0.85",
                        linewidth=0.25, alpha=0.18, shade=False)
        ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
                   s=26 if sn_order <= 4 else 12, c=directions[:, 2],
                   cmap="coolwarm", depthshade=False)
        ax.set_title(f"Level-symmetric S{sn_order} ordinates\n{angular_stats}", fontsize=11)
        ax.set_xlabel(r"$\Omega_x$")
        ax.set_ylabel(r"$\Omega_y$")
        ax.set_zlabel(r"$\Omega_z$")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=132)
        ax.set_proj_type("ortho")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"angle_S{sn_order}_sphere.png"), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(directions[:, 0], directions[:, 1],
                        s=30 if sn_order <= 4 else 12,
                        c=directions[:, 2], cmap="coolwarm")
        ax.set_xlabel(r"$\Omega_x$")
        ax.set_ylabel(r"$\Omega_y$")
        ax.set_title(f"Level-symmetric S{sn_order}, x-y projection\n{angular_stats}", fontsize=11)
        ax.text(
            0.02,
            0.98,
            f"order N={sn_order}\nM=N(N+2)={direction_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85, "pad": 3},
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        fig.colorbar(sc, ax=ax, label=r"$\Omega_z$")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"angle_S{sn_order}_xy.png"), dpi=300)
        plt.close()


def plot_all(show_figure2=False):
    plot_figure2(show=show_figure2)
    plot_figure5_outputs()
    plot_tetra_mesh()
    plot_angle_quadrature()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate all RSI example figures.")
    parser.add_argument(
        "--only",
        choices=[
            "all", "figure2", "figure5", "slices", "3d", "voxel3d",
            "volume3d", "isosurfaces", "paper-slices", "paper-slices-gauss", "mesh", "angles",
        ],
        default="all",
        help="Select a subset of figures to generate.",
    )
    parser.add_argument("--show-figure2", action="store_true",
                        help="Show Figure 2 windows after saving.")
    parser.add_argument(
        "--paper-mesh",
        choices=["coarse", "fine", "mesh200k"],
        default="coarse",
        help="Select mesh/data set for --only paper-slices.",
    )
    args = parser.parse_args(argv)

    if args.only == "all":
        plot_all(show_figure2=args.show_figure2)
    elif args.only == "figure2":
        plot_figure2(show=args.show_figure2)
    elif args.only == "figure5":
        plot_figure5_outputs()
    elif args.only == "slices":
        plot_figure5_slices()
    elif args.only == "3d":
        plot_figure5_3d()
    elif args.only == "voxel3d":
        plot_figure5_voxel3d()
    elif args.only == "volume3d":
        plot_figure5_volume3d()
    elif args.only == "isosurfaces":
        plot_figure5_isosurfaces()
    elif args.only == "paper-slices":
        if args.paper_mesh == "fine":
            plot_figure5_paper_slices_finemesh()
        elif args.paper_mesh == "mesh200k":
            plot_figure5_paper_slices_mesh200k()
        else:
            plot_figure5_paper_slices()
    elif args.only == "paper-slices-gauss":
        plot_figure5_paper_slices_gauss()
    elif args.only == "mesh":
        plot_tetra_mesh()
    elif args.only == "angles":
        plot_angle_quadrature()


if __name__ == "__main__":
    main()
