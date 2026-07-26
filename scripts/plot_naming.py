import os


SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def source_prefix(shape_path=None):
    if shape_path is None:
        shape_path = os.path.join(
            REPO_DIR,
            "Data",
            "csv_data",
            "source_shape.txt",
        )

    if not os.path.exists(shape_path):
        return "Rec"

    with open(shape_path, newline="") as f:
        shape = f.readline().strip()

    if shape == "circle":
        return "Cir"

    return "Rec"


def figure_base(csv_file, prefix=None):
    if prefix is None:
        prefix = source_prefix()

    base = os.path.basename(csv_file).replace(".csv", "")
    if base.startswith("figure5_"):
        return prefix + "_" + base[len("figure5_"):]

    return prefix + "_" + base
