import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rsi")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


BASE_DIR = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(BASE_DIR, "csv_data", "Rec", "figure5_RSI.csv")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "structured_grid")


def parse_grid_n(value):
    parts = [int(v) for v in value.replace(",", " ").split()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError("--grid-n must be one integer or three integers")


def parse_bounds(values):
    if values is None:
        return None
    nums = [float(v) for value in values for v in value.replace(",", " ").split()]
    if len(nums) != 6:
        raise argparse.ArgumentTypeError(
            "--bounds must contain six numbers: xmin xmax ymin ymax zmin zmax"
        )
    return np.array([[nums[0], nums[1]], [nums[2], nums[3]], [nums[4], nums[5]]], dtype=float)


def default_output_base(csv_file, out_dir):
    parent = os.path.basename(os.path.dirname(csv_file))
    stem = os.path.splitext(os.path.basename(csv_file))[0]
    if not parent or parent == "csv_data":
        parent = "field"
    return os.path.join(out_dir, f"{parent}_{stem}_grid")


def interpolate_csv(csv_file, output_base, grid_n=(101, 101, 101), bounds=None):
    df = pd.read_csv(csv_file)
    required = {"x", "y", "z", "phi0"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{csv_file} 缺少列: {', '.join(sorted(missing))}")

    points = df[["x", "y", "z"]].to_numpy(dtype=float)
    values = df["phi0"].to_numpy(dtype=float)

    if bounds is None:
        bounds = np.array(
            [
                [points[:, 0].min(), points[:, 0].max()],
                [points[:, 1].min(), points[:, 1].max()],
                [points[:, 2].min(), points[:, 2].max()],
            ],
            dtype=float,
        )

    xq = np.linspace(bounds[0, 0], bounds[0, 1], grid_n[0])
    yq = np.linspace(bounds[1, 0], bounds[1, 1], grid_n[1])
    zq = np.linspace(bounds[2, 0], bounds[2, 1], grid_n[2])
    x_grid, y_grid, z_grid = np.meshgrid(xq, yq, zq, indexing="ij")

    linear = LinearNDInterpolator(points, values, fill_value=np.nan)
    phi0_grid = linear(x_grid, y_grid, z_grid)

    missing_mask = np.isnan(phi0_grid)
    if np.any(missing_mask):
        nearest = NearestNDInterpolator(points, values)
        phi0_grid[missing_mask] = nearest(
            x_grid[missing_mask], y_grid[missing_mask], z_grid[missing_mask]
        )

    out_dir = os.path.dirname(output_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    npz_file = output_base + ".npz"
    mid_y_csv = output_base + "_y_mid.csv"
    mid_y_png = output_base + "_y_mid.png"

    np.savez_compressed(
        npz_file,
        xq=xq,
        yq=yq,
        zq=zq,
        phi0_grid=phi0_grid,
        bounds=bounds,
        grid_n=np.array(grid_n, dtype=np.int64),
        csv_file=csv_file,
        method="LinearNDInterpolator with NearestNDInterpolator fill",
    )

    mid_y = len(yq) // 2
    slice_values = phi0_grid[:, mid_y, :]
    rows = []
    for i, x in enumerate(xq):
        for k, z in enumerate(zq):
            rows.append((x, z, slice_values[i, k]))
    pd.DataFrame(rows, columns=["x", "z", "phi0"]).to_csv(mid_y_csv, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        slice_values.T,
        origin="lower",
        extent=(xq[0], xq[-1], zq[0], zq[-1]),
        aspect="equal",
        cmap="turbo",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Interpolated phi0, y = {yq[mid_y]:.4g}")
    fig.colorbar(im, ax=ax, label="phi0")
    fig.tight_layout()
    fig.savefig(mid_y_png, dpi=300)
    plt.close(fig)

    return {
        "npz": npz_file,
        "slice_csv": mid_y_csv,
        "slice_png": mid_y_png,
        "grid_n": grid_n,
        "bounds": bounds,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate unstructured RSI cell-center CSV data to a Cartesian grid."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input figure5 field CSV.")
    parser.add_argument(
        "--out",
        help="Output base path without extension. Defaults to examples/structured_grid/<case>_<field>_grid.",
    )
    parser.add_argument(
        "--grid-n",
        default=(101, 101, 101),
        type=parse_grid_n,
        help="One integer or three integers, for example 101 or '121 81 121'.",
    )
    parser.add_argument(
        "--bounds",
        nargs="+",
        help="Six numbers: xmin xmax ymin ymax zmin zmax. Defaults to CSV point bounds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_base = args.out or default_output_base(args.csv, DEFAULT_OUT_DIR)
    result = interpolate_csv(args.csv, output_base, grid_n=args.grid_n, bounds=parse_bounds(args.bounds))
    print(f"写出 {result['npz']}")
    print(f"写出 {result['slice_csv']}")
    print(f"写出 {result['slice_png']}")
    print(f"grid_n = {result['grid_n']}")
    print(f"bounds = {result['bounds'].tolist()}")


if __name__ == "__main__":
    main()
