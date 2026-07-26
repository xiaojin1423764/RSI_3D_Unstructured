import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rsi")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSV_DIR = os.path.join(REPO_DIR, "Data", "csv_data")
FIGURE_DIR = os.path.join(REPO_DIR, "results", "Figures")


def csv_path(name):
    return os.path.join(CSV_DIR, name)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_figure2(show=False, input_csv=None, output_prefix="figure2"):
    csv_file = input_csv if input_csv else csv_path("figure2_data.csv")
    df = pd.read_csv(csv_file)
    ensure_dir(FIGURE_DIR)
    fit_rows = []

    for scattering in ["isotropic", "anisotropic"]:
        sub = df[df["scattering"] == scattering]
        if sub.empty:
            continue
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
        plt.savefig(os.path.join(FIGURE_DIR, f"{output_prefix}_{scattering}.png"), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    fit_df = pd.DataFrame(fit_rows)
    print("拟合收敛阶：")
    print(fit_df.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate RSI Figure 2 plots.")
    parser.add_argument(
        "--only",
        choices=["figure2"],
        default="figure2",
        help="Only Figure 2 plotting is supported; Figure 5 is exported for Tecplot.",
    )
    parser.add_argument(
        "--show-figure2",
        action="store_true",
        help="Show Figure 2 windows after saving.",
    )
    parser.add_argument(
        "--input",
        help="CSV file to plot. Defaults to Data/csv_data/figure2_data.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        default="figure2",
        help="Output image prefix under results/Figures.",
    )
    args = parser.parse_args(argv)

    if args.only == "figure2":
        plot_figure2(
            show=args.show_figure2,
            input_csv=args.input,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()
