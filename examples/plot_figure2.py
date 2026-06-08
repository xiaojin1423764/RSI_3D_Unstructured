import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 Figure 2 数据
df = pd.read_csv("csv_data/figure2_data.csv")

fit_rows = []

for scattering in ["isotropic", "anisotropic"]:
    sub = df[df["scattering"] == scattering]

    plt.figure()

    for M in sorted(sub["M"].unique()):
        d = sub[sub["M"] == M].sort_values("S")

        S = d["S"].to_numpy(dtype=float)
        err = d["e_RSI_N"].to_numpy(dtype=float)

        # 拟合：
        # log(error) = a + b log(S)
        # 收敛阶 order = -b
        b, a = np.polyfit(np.log(S), np.log(err), 1)
        order = -b

        fit_rows.append({
            "scattering": scattering,
            "M": M,
            "slope": b,
            "order": order
        })

        plt.loglog(
            S,
            err,
            marker="o",
            label=f"M={M}, order={order:.3f}"
        )

    # 参考线 S^-0.5
    S_ref = np.array(sorted(sub["S"].unique()), dtype=float)
    e_ref = sub["e_RSI_N"].max()

    plt.loglog(
        S_ref,
        e_ref * (S_ref / S_ref[0]) ** (-0.5),
        linestyle="--",
        label=r"$S^{-0.5}$"
    )

    plt.xlabel("S")
    plt.ylabel(r"$e^{(N)}_{RSI}$")
    plt.title(scattering)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # 图片保存到 examples/Figures/
    plt.savefig(f"Figures/figure2_{scattering}.png", dpi=300)
    plt.show()
fit_df = pd.DataFrame(fit_rows)

print("拟合收敛阶：")
print(fit_df.to_string(index=False))