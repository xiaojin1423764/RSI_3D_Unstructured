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
from scipy.interpolate import griddata

from plot_naming import figure_base, source_prefix


BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "csv_data")
FIGURE_DIR = os.path.join(BASE_DIR, "Figures")
MSH_FILE = os.path.join(BASE_DIR, "..", "gmsh_work", "example1.msh")

FIELD_FILES = [
    ("figure5_SI_fine.csv", "SI fine", "si_fine", "S16, M=288"),
    ("figure5_SI_coarse.csv", "SI coarse", "si_coarse", "S4, M=24"),
    ("figure5_RSI.csv", "RSI", "rsi", "S16, M=288, samples=256"),
    ("figure5_RSI_tail.csv", "RSI tail average", "rsi_tail", "S16, M=288, samples=256, tail=10"),
]
SOURCE_CASES = [
    ("Rec", os.path.join(CSV_DIR, "Rec")),
    ("Cir", os.path.join(CSV_DIR, "Cir")),
]

Y_SLICES = [0.00, 0.25, 0.50, 0.75, 1.00]
LAYER_STACK_Y_SLICES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

VMIN = 0.0
VMAX = 2.2
FIELD_NORM = PowerNorm(gamma=0.55, vmin=VMIN, vmax=VMAX)
Y_SLICE_NORM = PowerNorm(gamma=0.35, vmin=0.0, vmax=0.12)

VOLUME_GRID_N = 64
LEGACY_VOXEL_GRID_N = 128
VOLUME_VISIBLE_MIN = 0.01
VOLUME_VIEWS = [("iso_back", 24, 132)]
PYVISTA_GRID_N = 96
PYVISTA_LOG_MIN = np.log10(0.0035)
PYVISTA_LOG_MAX = np.log10(0.12)

LAYERED_Z_SLICES = [0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82]
LAYERED_VISIBLE_MIN = 0.015
LAYERED_NORM = PowerNorm(gamma=0.42, vmin=LAYERED_VISIBLE_MIN, vmax=VMAX)
LAYER_STACK_ALPHA_LOW = 0.018
LAYER_STACK_ALPHA_HIGH = 0.08
LAYER_STACK_ALPHA_MAX = 0.34


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


def available_source_cases():
    cases = []
    for prefix, source_dir in SOURCE_CASES:
        if all(os.path.exists(source_csv_path(source_dir, file_name)) for file_name, *_ in FIELD_FILES):
            cases.append((prefix, source_dir))
    if cases:
        return cases

    prefix = source_prefix()
    if all(os.path.exists(csv_path(file_name)) for file_name, *_ in FIELD_FILES):
        return [(prefix, CSV_DIR)]
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
        plt.get_cmap("viridis")(np.linspace(0.12, 1.0, 256)),
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
    layer_norm = PowerNorm(gamma=0.38, vmin=0.0, vmax=layer_vmax)
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


def plot_layered_slices(df, title, out_path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    mappable = None
    for z0 in LAYERED_Z_SLICES:
        sub = slice_near_plane(df, "z", z0, quantile=0.035)
        if len(sub) < 3:
            continue
        values = sub["phi0"].to_numpy()
        if values.max() < LAYERED_VISIBLE_MIN:
            continue
        visible = values >= LAYERED_VISIBLE_MIN
        if visible.sum() < 3:
            continue
        triang = tri.Triangulation(sub["x"].to_numpy(), sub["y"].to_numpy())
        point_visible = visible[triang.triangles]
        triang.set_mask(~point_visible.all(axis=1))
        mappable = ax.tricontourf(
            triang,
            values,
            levels=np.linspace(LAYERED_VISIBLE_MIN, VMAX, 80),
            norm=LAYERED_NORM,
            zdir="z",
            offset=z0,
            alpha=0.38,
            antialiased=True,
        )

    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=r"$\phi_0$", shrink=0.72, pad=0.08)

    draw_box_axes(ax)
    ax.set_title(f"{title}, layered slices")
    finish_3d_axis(ax, 18, 132)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_figure5_layers():
    ensure_dir(FIGURE_DIR)
    for prefix, source_dir in available_source_cases():
        for file_name, title, out_dir, params in FIELD_FILES:
            file_path = source_csv_path(source_dir, file_name)
            df = pd.read_csv(file_path)
            base = figure_base(file_path, prefix)
            out_base_dir = os.path.join(FIGURE_DIR, out_dir)
            ensure_dir(out_base_dir)
            plot_layered_slices(df, titled_with_params(title, params), os.path.join(out_base_dir, f"{base}_z_layer_stack.png"))


def plot_figure5_outputs():
    plot_figure5_slices()
    plot_figure5_3d(include_layers=True, include_voxel=False)


def plot_figure5_voxel3d():
    plot_figure5_3d(include_layers=False, include_voxel=True)


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


def plot_tetra_mesh():
    points, tets = read_tetra_mesh()
    edges = tetra_edges(tets)
    boundary_faces = tetra_boundary_faces(tets)
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
        ax.set_title("Tetrahedral mesh")
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

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(sx, sy, sz, color="white", edgecolor="0.85",
                        linewidth=0.25, alpha=0.18, shade=False)
        ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
                   s=26 if sn_order <= 4 else 12, c=directions[:, 2],
                   cmap="coolwarm", depthshade=False)
        ax.set_title(f"Level-symmetric S{sn_order} ordinates")
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
        ax.set_title(f"Level-symmetric S{sn_order}, x-y projection")
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
        choices=["all", "figure2", "figure5", "slices", "3d", "voxel3d", "layers", "mesh", "angles"],
        default="all",
        help="Select a subset of figures to generate.",
    )
    parser.add_argument("--show-figure2", action="store_true",
                        help="Show Figure 2 windows after saving.")
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
    elif args.only == "layers":
        plot_figure5_layers()
    elif args.only == "mesh":
        plot_tetra_mesh()
    elif args.only == "angles":
        plot_angle_quadrature()


if __name__ == "__main__":
    main()
