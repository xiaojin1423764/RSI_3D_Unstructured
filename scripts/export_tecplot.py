import argparse
import csv
import os
from glob import glob

import meshio


SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MSH_FILE = os.path.join(REPO_DIR, "Data", "gmsh", "example1.msh")
DEFAULT_CSV_DIR = os.path.join(REPO_DIR, "Data", "csv_data")
DEFAULT_OUT_DIR = os.path.join(REPO_DIR, "Data", "tecplot")


def read_tetra_mesh(msh_file):
    mesh = meshio.read(msh_file)
    points = mesh.points[:, :3]
    tets = None
    for block in mesh.cells:
        if block.type == "tetra":
            tets = block.data
            break
    if tets is None:
        raise RuntimeError("没有找到 tetra 四面体单元，请确认 Gmsh 生成了 3D 四面体网格")
    return points, tets


def read_cell_field(csv_file, element_count):
    values_by_cell = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        required = {"cell_id", "phi0"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_file} 缺少列: {', '.join(sorted(missing))}")

        for row in reader:
            cell_id = int(row["cell_id"])
            values_by_cell[cell_id] = float(row["phi0"])

    if len(values_by_cell) != element_count:
        raise RuntimeError(
            f"{csv_file} 数据行数({len(values_by_cell)})与四面体单元数({element_count})不一致"
        )

    missing_ids = [i for i in range(element_count) if i not in values_by_cell]
    if missing_ids:
        preview = ", ".join(str(i) for i in missing_ids[:8])
        raise RuntimeError(f"{csv_file} 缺少 cell_id: {preview}")

    return [values_by_cell[i] for i in range(element_count)]


def write_block_values(f, values, per_line=6):
    for i in range(0, len(values), per_line):
        f.write(" ".join(f"{v:.16g}" for v in values[i:i + per_line]))
        f.write("\n")


def write_tecplot_dat(csv_file, msh_file, out_file, title=None):
    points, tets = read_tetra_mesh(msh_file)
    phi0 = read_cell_field(csv_file, len(tets))
    cell_ids = list(range(len(tets)))
    title = title or os.path.splitext(os.path.basename(csv_file))[0]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write('VARIABLES = "X" "Y" "Z" "phi0" "cell_id"\n')
        f.write(
            f'ZONE T="{title}", N={len(points)}, E={len(tets)}, '
            'ZONETYPE=FETETRAHEDRON, DATAPACKING=BLOCK, '
            'VARLOCATION=([4,5]=CELLCENTERED)\n'
        )
        write_block_values(f, points[:, 0])
        write_block_values(f, points[:, 1])
        write_block_values(f, points[:, 2])
        write_block_values(f, phi0)
        write_block_values(f, cell_ids)

        for tet in tets:
            nodes = tet + 1
            f.write(f"{nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]}\n")


def default_output_path(csv_file, csv_dir, out_dir):
    rel = os.path.relpath(csv_file, csv_dir)
    stem, _ = os.path.splitext(rel)
    return os.path.join(out_dir, stem + ".dat")


def discover_field_csvs(csv_dir):
    # The current workflow stores valid Figure 5 data under Rec/ and Cir/.
    # Older root-level files may belong to a different mesh, so skip them by default.
    pattern = os.path.join(csv_dir, "*", "figure5_*.csv")
    return sorted(glob(pattern))


def output_title(out_file, out_dir):
    out_abs = os.path.abspath(out_file)
    out_dir_abs = os.path.abspath(out_dir)
    try:
        rel = os.path.relpath(out_abs, out_dir_abs)
    except ValueError:
        rel = os.path.basename(out_file)
    if rel.startswith(".."):
        rel = os.path.basename(out_file)
    return os.path.splitext(rel)[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RSI tetrahedral cell fields to Tecplot ASCII DAT files."
    )
    parser.add_argument(
        "--msh",
        default=DEFAULT_MSH_FILE,
        help="Input Gmsh .msh file containing tetrahedral connectivity.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        help="One Figure 5 field CSV to export. Can be passed multiple times.",
    )
    parser.add_argument(
        "--csv-dir",
        default=DEFAULT_CSV_DIR,
        help="Directory searched recursively for figure5_*.csv when --csv is not given.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated Tecplot .dat files.",
    )
    parser.add_argument(
        "--out",
        help="Output .dat file. Only valid when exactly one --csv is provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = args.csv or discover_field_csvs(args.csv_dir)
    if not csv_files:
        raise RuntimeError(f"没有找到可导出的 figure5_*.csv: {args.csv_dir}")
    if args.out and len(csv_files) != 1:
        raise RuntimeError("--out 只能和单个 --csv 一起使用")

    for csv_file in csv_files:
        out_file = args.out or default_output_path(csv_file, args.csv_dir, args.out_dir)
        title = output_title(out_file, args.out_dir)
        write_tecplot_dat(csv_file, args.msh, out_file, title=title)
        print(f"写出 {out_file}")


if __name__ == "__main__":
    main()
