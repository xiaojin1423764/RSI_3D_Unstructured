#!/usr/bin/env python3
"""Rebuild a browsable archive of generated RSI images and data.

The archive uses hard links so it is cheap to rebuild and does not duplicate the
large figure directories. Re-run this script after generating new outputs.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "organized_outputs"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
DATA_EXTS = {".csv", ".npz", ".dat", ".mat", ".txt"}
SKIP_DIRS = {".git", ".agents", ".codex", "__pycache__", "organized_outputs"}


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def classify(path: Path) -> tuple[str, Path] | None:
    rel = path.relative_to(ROOT)
    ext = path.suffix.lower()

    if ext in IMAGE_EXTS:
        category = "images"
        root = image_group(rel)
    elif ext in DATA_EXTS:
        category = "data"
        root = data_group(rel)
    else:
        return None

    return category, root / rel.name


def image_group(rel: Path) -> Path:
    parts = rel.parts

    if parts[0] == "examples":
        if len(parts) > 2 and parts[1] == "structured_grid_circle_scaled_60k":
            return Path("current") / "scaled_circle_60k" / "structured_grid"
        if len(parts) > 2 and parts[1] == "structured_grid":
            return Path("current") / "structured_grid"
        if len(parts) > 2 and parts[1] == "Figures":
            return Path("current") / "figures" / Path(*parts[2:-1])
        return Path("examples_misc") / Path(*parts[1:-1])

    if parts[0] == "Gauss_figures":
        return Path("paper_slices") / "gauss" / Path(*parts[1:-1])

    if parts[0] == "No_Gasuss_figures":
        return Path("paper_slices") / "no_gauss" / Path(*parts[1:-1])

    if parts[0].startswith("Figure5_paper_slices"):
        return Path("paper_slices") / "legacy" / Path(*parts[:-1])

    return Path("misc") / Path(*parts[:-1])


def data_group(rel: Path) -> Path:
    parts = rel.parts

    if parts[0] == "examples":
        if len(parts) > 2 and parts[1] == "csv_data_circle_scaled_60k":
            return Path("current") / "scaled_circle_60k" / "csv" / Path(*parts[2:-1])
        if len(parts) > 2 and parts[1] == "structured_grid_circle_scaled_60k":
            return Path("current") / "scaled_circle_60k" / "structured_grid"
        if len(parts) > 2 and parts[1].startswith("csv_data"):
            return Path("csv") / parts[1] / Path(*parts[2:-1])
        if len(parts) > 2 and parts[1] == "structured_grid":
            return Path("structured_grid") / Path(*parts[2:-1])
        if len(parts) > 2 and parts[1] == "tecplot":
            return Path("tecplot") / Path(*parts[2:-1])
        return Path("examples_misc") / Path(*parts[1:-1])

    if parts[0] == "gmsh_work":
        return Path("mesh") / Path(*parts[1:-1])

    return Path("misc") / Path(*parts[:-1])


def hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def rebuild() -> tuple[Counter[str], list[Path]]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    counts: Counter[str] = Counter()
    linked: list[Path] = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or is_skipped(path):
            continue

        classified = classify(path)
        if classified is None:
            continue

        category, grouped_rel = classified
        dst = OUT / category / grouped_rel
        hardlink(path, dst)
        counts[category] += 1
        linked.append(dst)

    write_manifest(counts, linked)
    return counts, linked


def write_manifest(counts: Counter[str], linked: list[Path]) -> None:
    lines = [
        "Organized RSI outputs",
        "",
        "This directory is rebuilt by tools/organize_outputs.py.",
        "Files are hard links to the generated outputs, so existing script paths remain unchanged.",
        "Re-run the script after generating new figures or data.",
        "",
        "Top-level layout:",
        "  images/current/: current generated figures",
        "  images/paper_slices/: paper-style slice figure archives",
        "  data/current/: current generated CSV/structured-grid data",
        "  data/csv/: source CSV archives by dataset",
        "  data/mesh/: mesh CSV files",
        "  data/tecplot/: Tecplot exports",
        "",
        f"Image files: {counts['images']}",
        f"Data files: {counts['data']}",
        "",
        "Current scaled-circle 60k outputs:",
        "  images/current/scaled_circle_60k/structured_grid/",
        "  data/current/scaled_circle_60k/csv/Cir/",
        "  data/current/scaled_circle_60k/structured_grid/",
        "",
        "File list:",
    ]

    for path in sorted(linked):
        lines.append(f"  {path.relative_to(OUT)}")

    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    write_readme()


def write_readme() -> None:
    lines = [
        "Organized RSI outputs",
        "=====================",
        "",
        "This directory is rebuilt by `tools/organize_outputs.py`.",
        "",
        "The files here are hard links to generated outputs elsewhere in the repository,",
        "so existing script paths under `examples/`, `Gauss_figures/`, `No_Gasuss_figures/`,",
        "and `gmsh_work/` remain unchanged.",
        "",
        "Rebuild after generating new outputs:",
        "",
        "```bash",
        "python3 tools/organize_outputs.py",
        "```",
        "",
        "Layout:",
        "",
        "- `images/current/`: current generated figures.",
        "- `images/current/scaled_circle_60k/`: latest 60k scaled-circle structured-grid figures.",
        "- `images/paper_slices/`: paper-style slice archives.",
        "- `data/current/`: current generated CSV/structured-grid data.",
        "- `data/current/scaled_circle_60k/`: latest 60k scaled-circle Figure 5 data.",
        "- `data/csv/`: source CSV archives by dataset.",
        "- `data/mesh/`: mesh CSV files.",
        "- `data/tecplot/`: Tecplot exports.",
        "",
        "See `MANIFEST.txt` for a full file list.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    counts, _ = rebuild()
    print(f"organized_outputs rebuilt: images={counts['images']} data={counts['data']}")


if __name__ == "__main__":
    main()
